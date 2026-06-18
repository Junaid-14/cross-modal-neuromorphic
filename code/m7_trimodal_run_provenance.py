#!/usr/bin/env python3
"""M7 Tri-Modal Continual Learning — with provenance, proper metrics, and aggregation."""

from __future__ import annotations

import json
import os
import sys
import time
import copy
import dataclasses
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from scipy import stats

import snntorch as snn
from snntorch import surrogate
import tonic

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    seed: int
    task_order: List[str]
    optimizer: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 12
    gradient_clip: float = 1.0
    beta: float = 0.9
    dropout: float = 0.2
    hidden_dim: int = 512
    shared_dim: int = 512
    num_patterns: int = 100
    hopfield_temperature: float = 1.0
    num_gru_layers: int = 2
    num_workers: int = 0
    time_steps: int = 25
    pin_memory: bool = True
    use_contrastive: bool = True
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 0.1
    use_replay: bool = False
    replay_batch: int = 32
    replay_weight: float = 0.3
    replay_samples_per_class: int = 50
    device: str = "auto"
    conditions: List[str] = dataclasses.field(default_factory=lambda: ["no_replay", "replay"])
    seeds: List[int] = dataclasses.field(default_factory=lambda: [42, 123, 456, 789, 101112])
    # alignment flags for future ablations
    align_cross_modal: bool = False
    align_joint_training: bool = False
    align_mixed_batches: bool = False
    # cross-modal alignment hyperparameters
    align_shared_dim: int = 128
    align_weight: float = 0.5
    align_temperature: float = 0.1
    align_use_orthogonal: bool = False
    align_orthogonal_weight: float = 0.01
    nmnist_train_per_class: int = 1500

    def to_json(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Path):
        with open(path) as f:
            return cls(**json.load(f))


# ═══════════════════════════════════════════════════════════════════════════════
#  MPS/CPU compatibility helpers
# ═══════════════════════════════════════════════════════════════════════════════
def safe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device(cfg: ExperimentConfig):
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  COLLATE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def nmnist_collate_fn(batch):
    events, labels = zip(*batch)
    fixed = []
    for e in events:
        if not isinstance(e, torch.Tensor):
            e = torch.from_numpy(e).float()
        if e.shape[0] > 25:
            e = e[:25]
        elif e.shape[0] < 25:
            e = torch.cat([e, torch.zeros(25 - e.shape[0], *e.shape[1:])], dim=0)
        fixed.append(e)
    return torch.stack(fixed), torch.tensor(labels)


def shd_collate_fn(batch):
    events, labels = zip(*batch)
    return torch.stack(events), torch.tensor(labels)


def dvs_collate_fn(batch):
    events, labels = zip(*batch)
    fixed = []
    for e in events:
        if not isinstance(e, torch.Tensor):
            e = torch.from_numpy(e).float()
        if e.shape[0] > 25:
            e = e[:25]
        elif e.shape[0] < 25:
            e = torch.cat([e, torch.zeros(25 - e.shape[0], *e.shape[1:])], dim=0)
        if e.shape[-2:] != (34, 34):
            e = F.adaptive_avg_pool2d(
                e.view(-1, e.shape[-3], e.shape[-2], e.shape[-1]),
                (34, 34)
            ).view(25, e.shape[-3], 34, 34)
        fixed.append(e)
    return torch.stack(fixed), torch.tensor(labels)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
spike_grad = surrogate.atan()


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return -mean_log_prob.mean()


class CrossModalSupConLoss(nn.Module):
    """
    Supervised contrastive loss (Khosla et al. 2020) computed on the SHARED
    embedding subspace across BOTH modalities.

    Positives: all in-batch samples with the same digit identity, regardless of modality.
    Negatives: all other in-batch samples.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, shared_features, labels):
        """
        Args:
            shared_features: (B, align_shared_dim) — L2-normalised shared embeddings
            labels:          (B,) — digit identity labels (0-9), same for both modalities
        Returns:
            scalar loss
        """
        features = F.normalize(shared_features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        # Positive mask: same label, exclude self
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
        # Exclude self from positives (diagonal)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits)
        # Positives include ALL same-label samples (including cross-modal)
        pos_mask = mask
        # For denominator, use ALL samples (including self)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        # SupCon: average over positives (including self gives 0 contribution since mask*log_prob on diagonal is log(1)=0 after normalization)
        mean_log_prob = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
        return -mean_log_prob.mean()


def orthogonal_penalty(shared, private):
    """
    Compute ||shared^T @ private||_F^2  to discourage redundancy between
    shared and private subspaces.

    Args:
        shared:  (B, D_s)
        private: (B, D_p)
    Returns:
        scalar penalty
    """
    # shared^T @ private  →  (D_s, D_p)
    cross = torch.matmul(shared.T, private)
    return cross.pow(2).sum()


class ModernHopfieldLayer(nn.Module):
    def __init__(self, dim=512, num_patterns=256, temperature=1.0):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_patterns, dim) * 0.02)
        self.temperature = temperature
        self.scale = np.sqrt(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn = torch.matmul(x, self.memory.T) / self.temperature / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, self.memory)
        return self.norm(x + out)


class ImprovedHGRNGate(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.W_r = nn.Linear(dim * 2, dim)
        self.W_z = nn.Linear(dim * 2, dim)
        self.W_h = nn.Linear(dim * 2, dim)
        self.norm_r = nn.LayerNorm(dim)
        self.norm_z = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)

    def forward(self, x, h_prev=None):
        batch_size = x.size(0)
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.W_r.weight.size(1) // 2, device=x.device)
        concat = torch.cat([x, h_prev], dim=-1)
        r = torch.sigmoid(self.norm_r(self.W_r(concat)))
        z = torch.sigmoid(self.norm_z(self.W_z(concat)))
        h_tilde = torch.tanh(self.norm_h(self.W_h(torch.cat([x, r * h_prev], dim=-1))))
        return (1 - z) * h_prev + z * h_tilde


class SNN_Backbone(nn.Module):
    def __init__(self, input_channels=2, num_classes=10, beta=0.9, shared_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(1024, shared_dim)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(shared_dim, num_classes)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        B, T = x.size(0), x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        spk4_rec, spk5_rec = [], []
        for t in range(T):
            xt = x[:, t]
            c1 = self.pool1(self.conv1(xt))
            s1, mem1 = self.lif1(c1, mem1)
            c2 = self.pool2(self.conv2(s1))
            s2, mem2 = self.lif2(c2, mem2)
            f = s2.view(B, -1)
            f1 = self.fc1(f)
            s3, mem3 = self.lif3(f1, mem3)
            f2 = self.fc2(s3)
            s4, mem4 = self.lif4(f2, mem4)
            spk4_rec.append(s4)
            f3 = self.fc3(s4)
            s5, mem5 = self.lif5(f3, mem5)
            spk5_rec.append(s5)
        return torch.stack(spk5_rec, dim=1).sum(dim=1), torch.stack(spk4_rec, dim=1).sum(dim=1)


class M7_ContinualAdaptiveModel(nn.Module):
    def __init__(self, tasks, cfg: ExperimentConfig):
        super().__init__()
        self.tasks = tasks
        self.cfg = cfg
        self.backbones = nn.ModuleDict()
        for name, info in tasks.items():
            if info["type"] == "audio":
                self.backbones[name] = nn.ModuleDict({
                    "fc1": nn.Linear(700, 1024),
                    "lif1": snn.Leaky(beta=cfg.beta, spike_grad=spike_grad),
                    "fc2": nn.Linear(1024, 1024),
                    "lif2": snn.Leaky(beta=cfg.beta, spike_grad=spike_grad),
                    "fc3": nn.Linear(1024, cfg.shared_dim),
                    "lif3": snn.Leaky(beta=cfg.beta, spike_grad=spike_grad),
                    "fc4": nn.Linear(cfg.shared_dim, info["num_classes"]),
                    "lif4": snn.Leaky(beta=cfg.beta, spike_grad=spike_grad),
                })
            else:
                self.backbones[name] = SNN_Backbone(
                    input_channels=2, num_classes=info["num_classes"], beta=cfg.beta, shared_dim=cfg.shared_dim
                )
        self.hopfield = ModernHopfieldLayer(
            dim=cfg.shared_dim, num_patterns=cfg.num_patterns, temperature=cfg.hopfield_temperature
        )
        self.hgrn = ImprovedHGRNGate(dim=cfg.shared_dim)
        self.adaptive_gate = nn.Sequential(
            nn.Linear(cfg.shared_dim, 128), nn.ReLU(), nn.Linear(128, 2), nn.Softmax(dim=-1)
        )
        self.heads = nn.ModuleDict()
        for name, info in tasks.items():
            self.heads[name] = nn.Sequential(
                nn.LayerNorm(cfg.shared_dim), nn.Linear(cfg.shared_dim, info["num_classes"])
            )

        # Cross-modal alignment: shared / private subspace projections
        self.align_shared_dim = cfg.align_shared_dim
        self.proj_shared = nn.Linear(cfg.shared_dim, cfg.align_shared_dim)
        if cfg.align_shared_dim < cfg.shared_dim:
            self.proj_private = nn.Linear(cfg.shared_dim, cfg.shared_dim - cfg.align_shared_dim)
        else:
            self.proj_private = None

    def _get_features(self, x, modality):
        bb = self.backbones[modality]
        if isinstance(bb, SNN_Backbone):
            _, features = bb(x)
        else:
            B, T = x.size(0), x.size(1)
            mem1 = bb["lif1"].init_leaky()
            mem2 = bb["lif2"].init_leaky()
            mem3 = bb["lif3"].init_leaky()
            feats = []
            for t in range(T):
                xt = x[:, t].view(B, -1)
                s1, mem1 = bb["lif1"](bb["fc1"](xt), mem1)
                s2, mem2 = bb["lif2"](bb["fc2"](s1), mem2)
                s3, mem3 = bb["lif3"](bb["fc3"](s2), mem3)
                feats.append(s3)
            features = torch.stack(feats, dim=1).sum(dim=1)
        return features.unsqueeze(1)

    def _split_shared_private(self, features):
        """Project 512-d backbone features into shared + private subspaces."""
        shared = self.proj_shared(features)
        if self.align_shared_dim == self.cfg.shared_dim:
            # No split: align on full feature space (private is empty)
            return shared, None, shared
        private = self.proj_private(features)
        full = torch.cat([shared, private], dim=-1)
        return shared, private, full

    def forward_with_shared(self, x, modality, task_name):
        """
        Full forward pass that also returns the shared subspace embedding.
        Used during cross-modal alignment training.
        Returns: (logits, gate, full_features, shared_features)
        """
        bb = self.backbones[modality]
        if isinstance(bb, SNN_Backbone):
            _, features = bb(x)
        else:
            B, T = x.size(0), x.size(1)
            mem1 = bb["lif1"].init_leaky()
            mem2 = bb["lif2"].init_leaky()
            mem3 = bb["lif3"].init_leaky()
            feats = []
            for t in range(T):
                xt = x[:, t].view(B, -1)
                s1, mem1 = bb["lif1"](bb["fc1"](xt), mem1)
                s2, mem2 = bb["lif2"](bb["fc2"](s1), mem2)
                s3, mem3 = bb["lif3"](bb["fc3"](s2), mem3)
                feats.append(s3)
            features = torch.stack(feats, dim=1).sum(dim=1)
        shared, private, full = self._split_shared_private(features)
        gate = self.adaptive_gate(full)
        hop = self.hopfield(full)
        hgrn = self.hgrn(full)
        combined = gate[:, 0:1] * hop + gate[:, 1:2] * hgrn
        return self.heads[task_name](combined), gate, full, shared

    def forward(self, x, modality, task_name):
        """
        Standard forward pass (backbone-compatible signature).
        Returns: (logits, gate, full_features)
        """
        logits, gate, full, _ = self.forward_with_shared(x, modality, task_name)
        return logits, gate, full

    def forward_from_embeddings(self, embs, task_name):
        gate = self.adaptive_gate(embs)
        hop = self.hopfield(embs)
        hgrn = self.hgrn(embs)
        combined = gate[:, 0:1] * hop + gate[:, 1:2] * hgrn
        return self.heads[task_name](combined)


class EmbeddingReplayBuffer:
    def __init__(self, feature_dim=512, samples_per_class=50, device=None):
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.samples_per_class = samples_per_class
        self.device = device
        self._store = {}
        self._task_labels = {}

    def populate(self, model, loader, modality, task_id, label_offset=0):
        model.eval()
        class_samples = {}
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(next(model.parameters()).device)
                feats = model._get_features(data, modality).squeeze(1)
                for f, lbl in zip(feats, labels):
                    c = int(lbl.item()) + label_offset
                    if c not in class_samples:
                        class_samples[c] = []
                    class_samples[c].append(f.cpu())
        for c, samples in class_samples.items():
            if len(samples) > self.samples_per_class:
                indices = torch.randperm(len(samples))[:self.samples_per_class]
                samples = [samples[i] for i in indices]
            self._store[c] = torch.stack(samples)
            self._task_labels[c] = task_id

    def sample(self, batch_size, task_ids=None):
        if task_ids is None:
            task_ids = list(self._store.keys())
        available = [c for c in task_ids if c in self._store]
        if not available:
            return None, None
        classes = np.random.choice(available, size=min(batch_size, len(available)), replace=False)
        embs, lbls = [], []
        for c in classes:
            idx = np.random.randint(len(self._store[c]))
            embs.append(self._store[c][idx])
            lbls.append(c)
        return torch.stack(embs).to(self.device), torch.tensor(lbls, dtype=torch.long, device=self.device)

    def stats(self):
        return {
            "total": sum(len(v) for v in self._store.values()),
            "num_classes": len(self._store),
        }


class RawInputReplayBuffer:
    """Stores raw input tensors + labels for rehearsal through the full model pipeline."""
    def __init__(self, samples_per_class=200):
        self.samples_per_class = samples_per_class
        self._store = {}  # task_name -> list of (data_tensor, label_int)

    def populate(self, loader, task_name, max_batches=None):
        class_samples = {}
        for i, (data, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            for d, lbl in zip(data, labels):
                c = int(lbl.item())
                if c not in class_samples:
                    class_samples[c] = []
                class_samples[c].append(d)

        stored = []
        for c in sorted(class_samples):
            samples = class_samples[c]
            if len(samples) > self.samples_per_class:
                indices = torch.randperm(len(samples))[:self.samples_per_class]
                samples = [samples[i] for i in indices]
            for s in samples:
                stored.append((s, c))

        self._store[task_name] = stored
        n_classes = len(class_samples)
        print(f"   Raw buffer: {task_name} — {len(stored)} samples "
              f"({n_classes} classes × {len(stored)//n_classes} samples)")

    def sample(self, batch_size, task_names=None):
        if task_names is None:
            task_names = list(self._store.keys())

        result = {}
        for task_name in task_names:
            if task_name not in self._store:
                continue
            samples = self._store[task_name]
            if len(samples) == 0:
                continue

            n = min(batch_size, len(samples))
            indices = torch.randint(0, len(samples), (n,))
            batch_data = []
            batch_labels = []
            for idx in indices:
                d, lbl = samples[idx]
                batch_data.append(d)
                batch_labels.append(lbl)

            result[task_name] = (torch.stack(batch_data), torch.tensor(batch_labels))

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  CONTINUAL LEARNER — with proper metrics
# ═══════════════════════════════════════════════════════════════════════════════
class ContinualLearner:
    def __init__(self, model, buffer, cfg: ExperimentConfig, device=None):
        if device is None:
            device = get_device(cfg)
        self.model = model.to(device)
        self.buffer = buffer
        self.cfg = cfg
        self.device = device
        self.acc_matrix = {}
        self.history = {}
        self.completed_tasks = set()

    @torch.no_grad()
    def evaluate(self, loader, modality, task_name):
        self.model.eval()
        correct = total = 0
        for data, labels in loader:
            data, labels = data.to(self.device), labels.to(self.device)
            out, _, _ = self.model(data, modality, task_name)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    def _freeze_completed_tasks(self):
        """Freeze backbones and heads for all completed tasks; keep shared components trainable."""
        for p in self.model.parameters():
            p.requires_grad = True
        frozen_groups = []
        for task_name in sorted(self.completed_tasks):
            for prefix in [f"backbones.{task_name}", f"heads.{task_name}"]:
                count = 0
                for name, param in self.model.named_parameters():
                    if name.startswith(prefix):
                        param.requires_grad = False
                        count += 1
                if count > 0:
                    frozen_groups.append(f"{prefix} ({count} params)")
        if frozen_groups:
            print(f"   🔒 Frozen old tasks: {', '.join(frozen_groups)}")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"   📝 Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")

    def train_task(self, task_name, modality, train_loader, val_loader, test_loader,
                   num_epochs=50, patience=12, lr=1e-3, use_replay=False, replay_task_ids=None,
                   use_scl=True, scl_weight=0.1, step_id=0,
                   freeze_old=True, raw_replay_buffer=None, raw_replay_weight=1.0):
        if freeze_old:
            self._freeze_completed_tasks()
        else:
            for p in self.model.parameters():
                p.requires_grad = True
            print("   📝 All params trainable (no freezing)")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        ce = nn.CrossEntropyLoss()
        scl = SupervisedContrastiveLoss(temperature=self.cfg.contrastive_temperature).to(self.device)

        best_val = 0
        patience_counter = 0
        history = {
            "train_acc": [], "val_acc": [], "train_loss": [],
            "gate_hop": [], "gate_hgrn": [],
        }

        for epoch in range(num_epochs):
            self.model.train()
            train_correct = train_total = 0
            train_loss_sum = 0.0
            gate_hop_sum = gate_hgrn_sum = gate_count = 0

            total_batches = len(train_loader)
            print_every = max(1, total_batches // 10)
            for i, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits, gate, features = self.model(data, modality, task_name)
                loss = ce(logits, labels)

                if use_scl:
                    loss = loss + scl_weight * scl(features, labels)

                if use_replay and self.buffer.stats()["total"] > 0:
                    replay_embs, replay_labels = self.buffer.sample(self.cfg.replay_batch, replay_task_ids)
                    if replay_embs is not None:
                        replay_logits = self.model.forward_from_embeddings(replay_embs, task_name)
                        loss = loss + self.cfg.replay_weight * ce(replay_logits, replay_labels)

                # Raw-input rehearsal of completed tasks
                if raw_replay_buffer is not None and len(raw_replay_buffer._store) > 0:
                    replay_batches = raw_replay_buffer.sample(
                        self.cfg.replay_batch,
                        task_names=list(raw_replay_buffer._store.keys())
                    )
                    for replay_task, (replay_data, replay_labels) in replay_batches.items():
                        replay_data = replay_data.to(self.device)
                        replay_labels = replay_labels.to(self.device)
                        replay_logits, _, _ = self.model(replay_data, replay_task, replay_task)
                        loss = loss + raw_replay_weight * ce(replay_logits, replay_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
                optimizer.step()

                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
                train_loss_sum += loss.item()
                gate_hop_sum += gate[:, 0].sum().item()
                gate_hgrn_sum += gate[:, 1].sum().item()
                gate_count += gate.size(0)
                if (i + 1) % print_every == 0 or i == total_batches - 1:
                    print(
                        f"    batch {i+1:4d}/{total_batches} | "
                        f"loss={train_loss_sum/(i+1):.3f} | acc={100*train_correct/train_total:.1f}%"
                    )

            scheduler.step()
            train_acc = 100.0 * train_correct / train_total
            val_acc = self.evaluate(val_loader, modality, task_name)

            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_loss"].append(train_loss_sum / len(train_loader))
            history["gate_hop"].append(gate_hop_sum / gate_count)
            history["gate_hgrn"].append(gate_hgrn_sum / gate_count)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  [S{step_id}] Epoch {epoch+1:02d}/{num_epochs} | "
                    f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
                    f"hop={history['gate_hop'][-1]:.3f} hgrn={history['gate_hgrn'][-1]:.3f}"
                )

            if val_acc > best_val:
                best_val = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        test_acc = self.evaluate(test_loader, modality, task_name)
        self.completed_tasks.add(task_name)
        return {"test_acc": test_acc, "best_val": best_val, "history": history}

    def align_cross_modal(self, paired_loader, num_epochs=5, lr=1e-3,
                          use_scl=True, scl_weight=0.1,
                          align_weight=0.5, step_id=0,
                          freeze_old=True, raw_replay_buffer=None, raw_replay_weight=1.0):
        """
        Dedicated alignment phase using the PairedDigitLoader.

        Each batch yields (visual_batch, audio_batch, shared_labels).
        Both modalities are fed through their respective backbones;
        the shared subspace is trained with cross-modal SupCon,
        while the full representation still sees task CE + within-modality SCL.

        Total loss = CE(visual) + CE(audio)
                     + scl_weight * (SCL(visual) + SCL(audio))
                     + align_weight * CrossModalSupCon(shared_visual + shared_audio)
                     + optional orthogonal penalty
                     + raw_replay_weight * CE(replayed tasks)
        """
        if freeze_old:
            self._freeze_completed_tasks()
        else:
            for p in self.model.parameters():
                p.requires_grad = True
            print("   📝 All params trainable (no freezing)")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        ce = nn.CrossEntropyLoss()
        scl = SupervisedContrastiveLoss(temperature=self.cfg.contrastive_temperature).to(self.device)
        xmscl = CrossModalSupConLoss(temperature=self.cfg.align_temperature).to(self.device)

        history = {"total_loss": [], "ce_loss": [], "scl_loss": [], "xmscl_loss": [], "orth_loss": []}

        for epoch in range(num_epochs):
            self.model.train()
            epoch_total = epoch_ce = epoch_scl = epoch_xmscl = epoch_orth = 0.0
            n_batches = 0

            for vis_data, aud_data, shared_labels in paired_loader:
                vis_data = vis_data.to(self.device)
                aud_data = aud_data.to(self.device)
                shared_labels = shared_labels.to(self.device)

                optimizer.zero_grad()

                # Forward both modalities with shared subspace exposed
                logits_vis, gate_vis, full_vis, shared_vis = self.model.forward_with_shared(
                    vis_data, "nmnist", "nmnist"
                )
                logits_aud, gate_aud, full_aud, shared_aud = self.model.forward_with_shared(
                    aud_data, "shd", "shd"
                )

                # Task CE on both modalities
                loss_ce = ce(logits_vis, shared_labels) + ce(logits_aud, shared_labels)

                # Within-modality SCL on full representations
                loss_scl = torch.tensor(0.0, device=self.device)
                if use_scl:
                    loss_scl = scl(full_vis, shared_labels) + scl(full_aud, shared_labels)

                # Cross-modal SupCon on shared subspace
                shared_cat = torch.cat([shared_vis, shared_aud], dim=0)
                labels_cat = torch.cat([shared_labels, shared_labels], dim=0)
                loss_xmscl = xmscl(shared_cat, labels_cat)

                # Total loss
                loss = loss_ce + scl_weight * loss_scl + align_weight * loss_xmscl

                # Optional orthogonal penalty
                loss_orth = torch.tensor(0.0, device=self.device)
                if self.cfg.align_use_orthogonal:
                    private_vis = full_vis[:, self.cfg.align_shared_dim:]
                    private_aud = full_aud[:, self.cfg.align_shared_dim:]
                    loss_orth = (
                        orthogonal_penalty(shared_vis, private_vis)
                        + orthogonal_penalty(shared_aud, private_aud)
                    )
                    loss = loss + self.cfg.align_orthogonal_weight * loss_orth

                # Raw-input rehearsal of completed tasks
                if raw_replay_buffer is not None and len(raw_replay_buffer._store) > 0:
                    replay_batches = raw_replay_buffer.sample(
                        self.cfg.replay_batch,
                        task_names=list(raw_replay_buffer._store.keys())
                    )
                    for replay_task, (replay_data, replay_labels) in replay_batches.items():
                        replay_data = replay_data.to(self.device)
                        replay_labels = replay_labels.to(self.device)
                        replay_logits, _, _, _ = self.model.forward_with_shared(
                            replay_data, replay_task, replay_task
                        )
                        loss = loss + raw_replay_weight * ce(replay_logits, replay_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
                optimizer.step()

                epoch_total += loss.item()
                epoch_ce += loss_ce.item()
                epoch_scl += loss_scl.item()
                epoch_xmscl += loss_xmscl.item()
                epoch_orth += loss_orth.item()
                n_batches += 1

            scheduler.step()

            history["total_loss"].append(epoch_total / n_batches)
            history["ce_loss"].append(epoch_ce / n_batches)
            history["scl_loss"].append(epoch_scl / n_batches)
            history["xmscl_loss"].append(epoch_xmscl / n_batches)
            history["orth_loss"].append(epoch_orth / n_batches)

            print(
                f"  [Align S{step_id}] Epoch {epoch+1:02d}/{num_epochs} | "
                f"total={history['total_loss'][-1]:.3f} "
                f"ce={history['ce_loss'][-1]:.3f} "
                f"scl={history['scl_loss'][-1]:.3f} "
                f"xmscl={history['xmscl_loss'][-1]:.3f}"
                + (f" orth={history['orth_loss'][-1]:.3f}" if self.cfg.align_use_orthogonal else "")
            )

        return history


# ═══════════════════════════════════════════════════════════════════════════════
#  RANDOM-INIT BASELINE
# ═══════════════════════════════════════════════════════════════════════════════
def random_init_baseline(tasks, task_name, test_loader, cfg: ExperimentConfig, device, n_seeds=3):
    """
    Evaluate a freshly initialized model on a task, averaged over n_seeds.
    This is the random-chance baseline for proper FWT calculation.
    """
    accs = []
    modality = task_name
    for s in range(n_seeds):
        torch.manual_seed(cfg.seed + 1000 + s)
        model = M7_ContinualAdaptiveModel(tasks=tasks, cfg=cfg).to(device)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                out, _, _ = model(data, modality, task_name)
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
        accs.append(100.0 * correct / total if total > 0 else 0.0)
    return float(np.mean(accs))


# ═══════════════════════════════════════════════════════════════════════════════
#  SANITY PRINTOUT
# ═══════════════════════════════════════════════════════════════════════════════
def print_task_sanity(task_name, num_classes, chance, zero_shot_acc, after_training_acc,
                      learning_gain, fwt, random_init_baseline_acc):
    print(f"\n  📊 SANITY — Task: {task_name} ({num_classes} classes)")
    print(f"     chance          = {chance:.2f}%")
    print(f"     random_init     = {random_init_baseline_acc:.2f}%")
    print(f"     zero_shot       = {zero_shot_acc:.2f}%  (delta vs chance = {zero_shot_acc - chance:+.2f}%)")
    print(f"     after_training  = {after_training_acc:.2f}%")
    print(f"     learning_gain   = {learning_gain:.2f}%  (after - zero_shot)")
    print(f"     FWT             = {fwt:.2f}%  (zero_shot - random_init)")
    # Assertions / warnings
    delta_vs_chance = zero_shot_acc - chance
    if abs(delta_vs_chance) > 5.0:
        print(f"     ⚠️  WARNING: zero_shot deviates from chance by {delta_vs_chance:+.2f}pp (>5pp)")
    else:
        print(f"     ✅ zero_shot within ~1-5pp of chance (expected ~{chance:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-SEED RESULTS SAVE
# ═══════════════════════════════════════════════════════════════════════════════
def save_seed_result(result: dict, cfg: ExperimentConfig, results_dir: Path):
    """Save one seed's result to a per-seed CSV and append to master CSV."""
    csv_path = results_dir / "cl_trimodal_results.csv"
    df = pd.DataFrame([result])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)

    # Also save per-seed CSV for clean aggregation
    seed_csv = results_dir / f"cl_results_seed{result['seed']}_{result['condition']}.csv"
    df.to_csv(seed_csv, index=False)

    # Save config JSON next to it
    cfg_path = results_dir / f"config_seed{result['seed']}_{result['condition']}.json"
    cfg.to_json(cfg_path)

    print(f"   💾 Saved results to {csv_path} and {seed_csv}")
    print(f"   💾 Saved config to {cfg_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════════
def aggregate_results(results_dir: Path, metrics: List[str]) -> pd.DataFrame:
    """
    Ingest per-seed CSVs, compute mean ± std, 95% CI, and paired Wilcoxon
    signed-rank test for replay vs no_replay within each seed.
    """
    master_csv = results_dir / "cl_trimodal_results.csv"
    if not master_csv.exists():
        print("No results to aggregate.")
        return pd.DataFrame()

    df = pd.read_csv(master_csv)
    # Keep only the full-result rows (those with 'T1_after_T1' populated)
    df = df[df["T1_after_T1"].notna()].copy()

    rows = []
    for metric in metrics:
        for condition in ["no_replay", "replay"]:
            sub = df[df["condition"] == condition][metric].dropna()
            if len(sub) == 0:
                continue
            mean = sub.mean()
            std = sub.std(ddof=1)
            n = len(sub)
            ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            rows.append({
                "metric": metric,
                "condition": condition,
                "n": n,
                "mean": mean,
                "std": std,
                "ci95_lower": mean - ci95,
                "ci95_upper": mean + ci95,
            })

    summary = pd.DataFrame(rows)

    # Wilcoxon signed-rank: paired per seed
    wilcoxon_rows = []
    seeds = df["seed"].unique()
    for metric in metrics:
        no_rep = df[df["condition"] == "no_replay"][["seed", metric]].dropna().set_index("seed")
        rep = df[df["condition"] == "replay"][["seed", metric]].dropna().set_index("seed")
        paired = no_rep.join(rep, lsuffix="_nr", rsuffix="_r", how="inner")
        if len(paired) > 1:
            diff = paired[f"{metric}_r"] - paired[f"{metric}_nr"]
            # Wilcoxon requires nonzero differences
            if (diff != 0).any():
                stat, pval = stats.wilcoxon(diff)
                wilcoxon_rows.append({
                    "metric": metric,
                    "wilcoxon_statistic": stat,
                    "wilcoxon_pvalue": pval,
                    "n_paired": len(paired),
                    "mean_diff_replay_minus_no_replay": diff.mean(),
                })
            else:
                wilcoxon_rows.append({
                    "metric": metric,
                    "wilcoxon_statistic": np.nan,
                    "wilcoxon_pvalue": np.nan,
                    "n_paired": len(paired),
                    "mean_diff_replay_minus_no_replay": 0.0,
                    "note": "all differences zero",
                })

    wilcoxon_df = pd.DataFrame(wilcoxon_rows)

    summary_path = results_dir / "cl_trimodal_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n💾 Aggregated summary saved to {summary_path}")

    if not wilcoxon_df.empty:
        wilcoxon_path = results_dir / "cl_trimodal_wilcoxon.csv"
        wilcoxon_df.to_csv(wilcoxon_path, index=False)
        print(f"💾 Wilcoxon test results saved to {wilcoxon_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  TRI-MODAL CL EXPERIMENT — per seed
# ═══════════════════════════════════════════════════════════════════════════════
TASKS = {
    "nmnist": {"num_classes": 10, "type": "visual"},
    "shd":    {"num_classes": 20, "type": "audio"},
    "dvs":    {"num_classes": 11, "type": "visual_motion"},
}


def run_one_seed(cfg: ExperimentConfig, data_loaders: dict, results_dir: Path, ckpt_dir: Path):
    condition = "replay" if cfg.use_replay else "no_replay"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"\n{'='*80}")
    print(f"  SEED {cfg.seed} | {condition.upper()}")
    print(f"{'='*80}")

    device = get_device(cfg)
    model = M7_ContinualAdaptiveModel(tasks=TASKS, cfg=cfg).to(device)
    buffer = EmbeddingReplayBuffer(
        feature_dim=cfg.shared_dim,
        samples_per_class=cfg.replay_samples_per_class,
        device=str(device),
    )
    trainer = ContinualLearner(model=model, buffer=buffer, cfg=cfg, device=device)

    # Pre-compute random-init baselines for FWT
    print("\n  🔧 Computing random-init baselines...")
    random_init_baselines = {}
    for task_name in cfg.task_order:
        ri = random_init_baseline(TASKS, task_name, data_loaders[task_name]["test"], cfg, device, n_seeds=3)
        random_init_baselines[task_name] = ri
        print(f"     {task_name}: {ri:.2f}%")

    # ═══════════════════════════════════════════════════════════════════════
    #  T1
    # ═══════════════════════════════════════════════════════════════════════
    task1 = cfg.task_order[0]
    print(f"\n📌 STEP 1: {task1.upper()}")
    ckpt_t1 = ckpt_dir / f"final_{task1}_{task1}_seed{cfg.seed}_{condition}.pt"
    if ckpt_t1.exists():
        print(f"   ⏩ Skipping T1 training (checkpoint found)")
        model.load_state_dict(torch.load(ckpt_t1, map_location=str(device))["model_state_dict"])
    else:
        trainer.train_task(
            task1, task1, *data_loaders[task1]["train_val_test"],
            num_epochs=15, patience=5, lr=cfg.lr,
            use_replay=False, use_scl=cfg.use_contrastive,
            scl_weight=cfg.contrastive_weight, step_id=1,
        )
        torch.save({"model_state_dict": model.state_dict()}, ckpt_t1)
        # Save config next to checkpoint
        cfg.to_json(ckpt_dir / f"config_{task1}_seed{cfg.seed}_{condition}.json")

    acc_t1_after = trainer.evaluate(data_loaders[task1]["test"], task1, task1)
    print(f"   T1 Test: {acc_t1_after:.2f}%")

    if cfg.use_replay:
        buffer.populate(model, data_loaders[task1]["train"], task1, task_id=1, label_offset=0)
        print(f"   Buffer T1: {buffer.stats()['total']} embeddings")

    # ═══════════════════════════════════════════════════════════════════════
    #  ALIGNMENT (between T1 and T2)
    # ═══════════════════════════════════════════════════════════════════════
    if cfg.align_cross_modal and "paired" in data_loaders:
        print("\n🔗 ALIGNMENT: cross-modal SupCon on shared subspace")
        align_history = trainer.align_cross_modal(
            paired_loader=data_loaders["paired"],
            num_epochs=5,
            lr=cfg.lr,
            use_scl=cfg.use_contrastive,
            scl_weight=cfg.contrastive_weight,
            align_weight=cfg.align_weight,
            step_id=1,
        )
        print(f"   Final xmscl_loss: {align_history['xmscl_loss'][-1]:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    #  T2
    # ═══════════════════════════════════════════════════════════════════════
    task2 = cfg.task_order[1]
    print(f"\n📌 STEP 2: {task2.upper()}")

    # Zero-shot BEFORE training T2
    acc_t2_zero_shot = trainer.evaluate(data_loaders[task2]["test"], task2, task2)
    chance_t2 = 100.0 / TASKS[task2]["num_classes"]
    print(f"   T2 zero-shot: {acc_t2_zero_shot:.2f}% (chance={chance_t2:.2f}%)")

    ckpt_t2 = ckpt_dir / f"final_{task2}_{task2}_seed{cfg.seed}_{condition}.pt"
    if ckpt_t2.exists():
        print(f"   ⏩ Skipping T2 training (checkpoint found)")
        model.load_state_dict(torch.load(ckpt_t2, map_location=str(device))["model_state_dict"])
    else:
        trainer.train_task(
            task2, task2, *data_loaders[task2]["train_val_test"],
            num_epochs=15, patience=5, lr=cfg.lr,
            use_replay=cfg.use_replay,
            replay_task_ids=[1] if cfg.use_replay else None,
            use_scl=cfg.use_contrastive,
            scl_weight=cfg.contrastive_weight, step_id=2,
        )
        torch.save({"model_state_dict": model.state_dict()}, ckpt_t2)
        cfg.to_json(ckpt_dir / f"config_{task2}_seed{cfg.seed}_{condition}.json")

    acc_t2_after = trainer.evaluate(data_loaders[task2]["test"], task2, task2)
    print(f"   T2 Test: {acc_t2_after:.2f}%")

    # Re-evaluate T1 after T2
    acc_t1_after_t2 = trainer.evaluate(data_loaders[task1]["test"], task1, task1)
    print(f"   T1 after T2: {acc_t1_after_t2:.2f}%")

    # Sanity printout for T2
    learning_gain_t2 = acc_t2_after - acc_t2_zero_shot
    fwt_t2 = acc_t2_zero_shot - random_init_baselines[task2]
    print_task_sanity(
        task_name=task2,
        num_classes=TASKS[task2]["num_classes"],
        chance=chance_t2,
        zero_shot_acc=acc_t2_zero_shot,
        after_training_acc=acc_t2_after,
        learning_gain=learning_gain_t2,
        fwt=fwt_t2,
        random_init_baseline_acc=random_init_baselines[task2],
    )

    if cfg.use_replay:
        buffer.populate(model, data_loaders[task2]["train"], task2, task_id=2, label_offset=0)
        print(f"   Buffer T2: {buffer.stats()['total']} embeddings")

    # ═══════════════════════════════════════════════════════════════════════
    #  T3
    # ═══════════════════════════════════════════════════════════════════════
    task3 = cfg.task_order[2]
    print(f"\n📌 STEP 3: {task3.upper()}")

    # Zero-shot BEFORE training T3
    acc_t3_zero_shot = trainer.evaluate(data_loaders[task3]["test"], task3, task3)
    chance_t3 = 100.0 / TASKS[task3]["num_classes"]
    print(f"   T3 zero-shot: {acc_t3_zero_shot:.2f}% (chance={chance_t3:.2f}%)")

    ckpt_t3 = ckpt_dir / f"final_{task3}_{task3}_seed{cfg.seed}_{condition}.pt"
    if ckpt_t3.exists():
        print(f"   ⏩ Skipping T3 training (checkpoint found)")
        model.load_state_dict(torch.load(ckpt_t3, map_location=str(device))["model_state_dict"])
    else:
        trainer.train_task(
            task3, task3, *data_loaders[task3]["train_val_test"],
            num_epochs=15, patience=5, lr=cfg.lr,
            use_replay=cfg.use_replay,
            replay_task_ids=[1, 2] if cfg.use_replay else None,
            use_scl=cfg.use_contrastive,
            scl_weight=cfg.contrastive_weight, step_id=3,
        )
        torch.save({"model_state_dict": model.state_dict()}, ckpt_t3)
        cfg.to_json(ckpt_dir / f"config_{task3}_seed{cfg.seed}_{condition}.json")

    acc_t3_after = trainer.evaluate(data_loaders[task3]["test"], task3, task3)
    print(f"   T3 Test: {acc_t3_after:.2f}%")

    # Re-evaluate T1 and T2 after T3
    acc_t1_after_t3 = trainer.evaluate(data_loaders[task1]["test"], task1, task1)
    acc_t2_after_t3 = trainer.evaluate(data_loaders[task2]["test"], task2, task2)
    print(f"   T1 after T3: {acc_t1_after_t3:.2f}%")
    print(f"   T2 after T3: {acc_t2_after_t3:.2f}%")

    # Sanity printout for T3
    learning_gain_t3 = acc_t3_after - acc_t3_zero_shot
    fwt_t3 = acc_t3_zero_shot - random_init_baselines[task3]
    print_task_sanity(
        task_name=task3,
        num_classes=TASKS[task3]["num_classes"],
        chance=chance_t3,
        zero_shot_acc=acc_t3_zero_shot,
        after_training_acc=acc_t3_after,
        learning_gain=learning_gain_t3,
        fwt=fwt_t3,
        random_init_baseline_acc=random_init_baselines[task3],
    )

    # Forgetting metrics
    forgetting_t1_t2 = acc_t1_after - acc_t1_after_t2
    forgetting_t1_t3 = acc_t1_after_t2 - acc_t1_after_t3
    forgetting_t2_t3 = acc_t2_after - acc_t2_after_t3

    result = {
        "seed": cfg.seed,
        "condition": condition,
        "T1_after_T1": acc_t1_after,
        "T2_after_T2": acc_t2_after,
        "T3_after_T3": acc_t3_after,
        "T1_after_T2": acc_t1_after_t2,
        "T1_after_T3": acc_t1_after_t3,
        "T2_after_T3": acc_t2_after_t3,
        "T2_zero_shot": acc_t2_zero_shot,
        "T3_zero_shot": acc_t3_zero_shot,
        "T2_chance": chance_t2,
        "T3_chance": chance_t3,
        "T2_learning_gain": learning_gain_t2,
        "T3_learning_gain": learning_gain_t3,
        "T2_random_init_baseline": random_init_baselines[task2],
        "T3_random_init_baseline": random_init_baselines[task3],
        "T2_fwt": fwt_t2,
        "T3_fwt": fwt_t3,
        "forgetting_T1_T2": forgetting_t1_t2,
        "forgetting_T1_T3": forgetting_t1_t3,
        "forgetting_T2_T3": forgetting_t2_t3,
    }

    save_seed_result(result, cfg, results_dir)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING (lazy — called once per script invocation)
# ═══════════════════════════════════════════════════════════════════════════════
def build_data_loaders(cfg: ExperimentConfig):
    data_dir = Path("./ICONS_M7/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

    # N-MNIST
    sensor_size = tonic.datasets.NMNIST.sensor_size
    nmnist_transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])
    nmnist_train = tonic.datasets.NMNIST(save_to=str(data_dir), train=True, transform=nmnist_transform)
    nmnist_test = tonic.datasets.NMNIST(save_to=str(data_dir), train=False, transform=nmnist_transform)
    cached_nmnist_train = tonic.DiskCachedDataset(nmnist_train, cache_path=str(data_dir / "nmnist_train_cache"))
    n_val = int(0.1 * len(cached_nmnist_train))
    n_train = len(cached_nmnist_train) - n_val
    train_split, val_split = random_split(
        cached_nmnist_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    nmnist_loaders = {
        "train": DataLoader(train_split, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=nmnist_collate_fn),
        "val":   DataLoader(val_split,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=nmnist_collate_fn),
        "test":  DataLoader(nmnist_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=nmnist_collate_fn),
        "train_val_test": (
            DataLoader(train_split, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=nmnist_collate_fn),
            DataLoader(val_split,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=nmnist_collate_fn),
            DataLoader(nmnist_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=nmnist_collate_fn),
        ),
    }

    # SHD
    shd_train = tonic.datasets.SHD(save_to=str(data_dir), train=True)
    shd_test = tonic.datasets.SHD(save_to=str(data_dir), train=False)
    def _events_to_dense(events, label):
        time_bins, channels = 100, 700
        dense = torch.zeros(time_bins, channels)
        if len(events) > 0:
            max_time = events["t"].max() if len(events) > 0 else 1
            time_indices = (events["t"] / max_time * (time_bins - 1)).astype(int)
            channel_indices = events["x"].astype(int)
            for t, c in zip(time_indices, channel_indices):
                if 0 <= t < time_bins and 0 <= c < channels:
                    dense[t, c] = 1.0
        return dense.unsqueeze(1).unsqueeze(1), label

    shd_train_data = [_events_to_dense(e, l) for e, l in tqdm(shd_train, desc="  SHD train")]
    shd_test_data = [_events_to_dense(e, l) for e, l in tqdm(shd_test, desc="  SHD test")]
    n_val_shd = int(0.1 * len(shd_train_data))
    shd_train_split, shd_val_split = random_split(
        shd_train_data, [len(shd_train_data) - n_val_shd, n_val_shd],
        generator=torch.Generator().manual_seed(42))
    shd_loaders = {
        "train": DataLoader(shd_train_split, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=shd_collate_fn),
        "val":   DataLoader(shd_val_split,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn),
        "test":  DataLoader(shd_test_data,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn),
        "train_val_test": (
            DataLoader(shd_train_split, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=shd_collate_fn),
            DataLoader(shd_val_split,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn),
            DataLoader(shd_test_data,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn),
        ),
    }

    # DVS-Gesture
    dvs_transform = tonic.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, time_window=50000),
    ])
    dvs_train = tonic.datasets.DVSGesture(save_to=str(data_dir), train=True, transform=dvs_transform)
    dvs_test = tonic.datasets.DVSGesture(save_to=str(data_dir), train=False, transform=dvs_transform)
    n_val_dvs = int(0.1 * len(dvs_train))
    dvs_train_split, dvs_val_split = random_split(
        dvs_train, [len(dvs_train) - n_val_dvs, n_val_dvs],
        generator=torch.Generator().manual_seed(42))
    dvs_loaders = {
        "train": DataLoader(dvs_train_split, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=dvs_collate_fn),
        "val":   DataLoader(dvs_val_split,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=dvs_collate_fn),
        "test":  DataLoader(dvs_test,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=dvs_collate_fn),
        "train_val_test": (
            DataLoader(dvs_train_split, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=dvs_collate_fn),
            DataLoader(dvs_val_split,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=dvs_collate_fn),
            DataLoader(dvs_test,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=dvs_collate_fn),
        ),
    }

    # Paired cross-modal loader (N-MNIST + SHD)
    from Dataloaders.paired_digit_loader import get_paired_digit_loaders

    def _visual_transform(x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        if x.shape[0] > 25:
            x = x[:25]
        elif x.shape[0] < 25:
            x = torch.cat([x, torch.zeros(25 - x.shape[0], *x.shape[1:])], dim=0)
        return x

    paired_loader = get_paired_digit_loaders(
        visual_dataset=train_split,      # N-MNIST train Subset
        audio_dataset=shd_train_split,   # SHD train Subset
        batch_size=30,
        samples_per_class=min(500, len(train_split) // 10, len(shd_train_split) // 10),
        num_workers=0,
        seed=42,
        visual_transform=_visual_transform,
        audio_transform=None,
    )

    return {
        "nmnist": nmnist_loaders,
        "shd": shd_loaders,
        "dvs": dvs_loaders,
        "paired": paired_loader,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    base_dir = Path("./ICONS_M7")
    results_dir = base_dir / "results"
    ckpt_dir = results_dir / "checkpoints"
    plot_dir = results_dir / "plots"
    for d in [results_dir, ckpt_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Default config
    base_cfg = ExperimentConfig(
        seed=42,
        task_order=["nmnist", "shd", "dvs"],
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        max_epochs=50,
        patience=12,
        gradient_clip=1.0,
        beta=0.9,
        hidden_dim=512,
        shared_dim=512,
        num_patterns=100,
        hopfield_temperature=1.0,
        num_gru_layers=2,
        num_workers=0,
        time_steps=25,
        pin_memory=True,
        use_contrastive=True,
        contrastive_temperature=0.07,
        contrastive_weight=0.1,
        use_replay=False,
        replay_batch=32,
        replay_weight=0.3,
        replay_samples_per_class=50,
        device="auto",
        conditions=["no_replay", "replay"],
        seeds=[42, 123, 456, 789, 101112],
        align_cross_modal=False,
        align_shared_dim=128,
        align_weight=0.5,
        align_temperature=0.1,
        align_use_orthogonal=False,
        align_orthogonal_weight=0.01,
    )

    print("="*80)
    print("🚀 M7 TRI-MODAL CONTINUAL LEARNING — Provenance Edition")
    print("="*80)
    print(f"Device: {get_device(base_cfg)}")
    print(f"Seeds: {base_cfg.seeds}")
    print(f"Conditions: {base_cfg.conditions}")
    print()

    # Build data loaders once
    print("Loading datasets...")
    data_loaders = build_data_loaders(base_cfg)
    print("Done.\n")

    all_results = []
    for seed in base_cfg.seeds:
        for condition in base_cfg.conditions:
            cfg = dataclasses.replace(
                base_cfg,
                seed=seed,
                use_replay=(condition == "replay"),
            )
            result = run_one_seed(cfg, data_loaders, results_dir, ckpt_dir)
            all_results.append(result)

    # Summary
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    df = pd.DataFrame(all_results)
    cols = [c for c in df.columns if c not in ["T2_chance", "T3_chance", "T2_random_init_baseline", "T3_random_init_baseline"]]
    print(df[cols].to_string(index=False))

    # Aggregate
    metrics = [
        "T1_after_T1", "T2_after_T2", "T3_after_T3",
        "T1_after_T2", "T1_after_T3", "T2_after_T3",
        "T2_zero_shot", "T3_zero_shot",
        "T2_learning_gain", "T3_learning_gain",
        "T2_fwt", "T3_fwt",
        "forgetting_T1_T2", "forgetting_T1_T3", "forgetting_T2_T3",
    ]
    aggregate_results(results_dir, metrics)

    # Save master config
    base_cfg.to_json(results_dir / "master_config.json")
    print(f"\n💾 Master config saved to {results_dir / 'master_config.json'}")
    print("\n✅ Tri-modal CL complete!")


if __name__ == "__main__":
    main()

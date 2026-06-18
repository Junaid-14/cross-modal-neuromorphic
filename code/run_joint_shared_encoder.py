#!/usr/bin/env python3
"""
Joint Shared Encoder Experiment
───────────────────────────────────────────────────────────────────────────────
Test whether a SHARED encoder trained JOINTLY on both modalities learns a
cross-modal digit geometry that GENERALIZES to held-out classes.

Architecture (modality-specific stems -> SHARED core -> shared space):
  visual_stem:  raw N-MNIST frames -> d_stem (256)
  audio_stem:   SHD dense (T,700) -> d_stem (256)
  shared_core:  d_stem -> d_embed (128), L2-normalised  [SHARED weights]
  classifier:   Linear(d_embed, 6) for digits 0-5        [SHARED weights]

Training: joint on digits 0-5 only.  Loss = CrossModalSupCon + CE(both modalities).
Held-out: digits 6-9 NEVER seen during training.

Outputs: results/joint_shared_encoder/
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import snntorch as snn
from snntorch import surrogate
import tonic

# ── import existing utilities ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from m7_trimodal_run_provenance import (
    CrossModalSupConLoss,
    spike_grad,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results/joint_shared_encoder")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("./ICONS_M7/datasets")

TRAIN_CLASSES = (0, 1, 2, 3, 4, 5)
HELD_OUT_CLASSES = (6, 7, 8, 9)
NUM_TRAIN_CLASSES = len(TRAIN_CLASSES)
NUM_HELD_OUT_CLASSES = len(HELD_OUT_CLASSES)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    d_stem: int = 256
    d_embed: int = 128
    beta: float = 0.9
    batch_size: int = 30          # divisible by 6 for class balance
    samples_per_class: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    supcon_temperature: float = 0.1
    epochs: int = 100
    patience: int = 10
    num_workers: int = 0
    seeds: Tuple[int, ...] = (42, 123, 456, 789, 1011)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class ClassFilteredDataset(Dataset):
    """Wrap a dataset and expose only samples whose label is in `keep_classes`."""

    def __init__(self, base_dataset: Dataset, keep_classes: Tuple[int, ...], label_fn=None):
        super().__init__()
        self.base_dataset = base_dataset
        self.keep_classes = set(keep_classes)
        self.label_fn = label_fn or (lambda x: int(x))
        self.indices: List[int] = []
        for i in range(len(base_dataset)):
            _, lbl = base_dataset[i]
            if self.label_fn(lbl) in self.keep_classes:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


class PairedDigitDatasetFiltered(Dataset):
    """
    Yokes N-MNIST + SHD by shared digit identity, but ONLY for a subset of classes.
    Mirrors the logic of Dataloaders.paired_digit_loader.PairedDigitDataset
    but generalised to an arbitrary class list.
    """

    def __init__(
        self,
        visual_dataset: Dataset,
        audio_dataset: Dataset,
        classes: Tuple[int, ...] = TRAIN_CLASSES,
        samples_per_class: int = 500,
        visual_transform=None,
        audio_transform=None,
        seed: int = 42,
    ):
        self.visual_dataset = visual_dataset
        self.audio_dataset = audio_dataset
        self.classes = tuple(sorted(classes))
        self.num_classes = len(self.classes)
        self.samples_per_class = samples_per_class
        self.visual_transform = visual_transform
        self.audio_transform = audio_transform

        # Index visual by label
        self.visual_by_id: Dict[int, List[int]] = {c: [] for c in self.classes}
        for idx in range(len(visual_dataset)):
            _, lbl = visual_dataset[idx]
            lbl = int(lbl)
            if lbl in self.visual_by_id:
                self.visual_by_id[lbl].append(idx)

        # Index audio by label
        self.audio_by_id: Dict[int, List[int]] = {c: [] for c in self.classes}
        for idx in range(len(audio_dataset)):
            _, lbl = audio_dataset[idx]
            lbl = int(lbl)
            if lbl in self.audio_by_id:
                self.audio_by_id[lbl].append(idx)

        # Feasibility clamp
        min_vis = min(len(v) for v in self.visual_by_id.values()) if self.visual_by_id else 0
        min_aud = min(len(v) for v in self.audio_by_id.values()) if self.audio_by_id else 0
        self._feasible_spc = min(min_vis, min_aud, samples_per_class)
        if self._feasible_spc < samples_per_class:
            print(
                f"⚠️  PairedDigitDatasetFiltered: requested spc={samples_per_class}, "
                f"limiting to {self._feasible_spc} (vis={min_vis}, aud={min_aud})"
            )
            self.samples_per_class = self._feasible_spc

        rng = np.random.default_rng(seed)
        self.visual_indices: Dict[int, np.ndarray] = {}
        self.audio_indices: Dict[int, np.ndarray] = {}

        for c in self.classes:
            vis_pool = np.array(self.visual_by_id[c])
            aud_pool = np.array(self.audio_by_id[c])
            rng.shuffle(vis_pool)
            rng.shuffle(aud_pool)
            vis_rep = np.tile(vis_pool, math.ceil(self.samples_per_class / len(vis_pool)))
            aud_rep = np.tile(aud_pool, math.ceil(self.samples_per_class / len(aud_pool)))
            self.visual_indices[c] = vis_rep[: self.samples_per_class]
            self.audio_indices[c] = aud_rep[: self.samples_per_class]

        # Build class-balanced flat index
        slots = []
        for s in range(self.samples_per_class):
            class_order = rng.permutation(self.num_classes)
            slots.append([int(self.classes[c] + s * self.num_classes) for c in class_order])
        rng.shuffle(slots)
        self._flat_indices: List[int] = [idx for slot in slots for idx in slot]

    def __len__(self):
        return self.samples_per_class * self.num_classes

    def _decode_flat(self, flat_idx: int):
        c = flat_idx % self.num_classes
        s = flat_idx // self.num_classes
        return s, self.classes[c]

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for size {len(self)}")
        flat_idx = self._flat_indices[idx]
        s, c = self._decode_flat(flat_idx)
        vis_idx = int(self.visual_indices[c][s])
        aud_idx = int(self.audio_indices[c][s])
        vis_data, _ = self.visual_dataset[vis_idx]
        aud_data, _ = self.audio_dataset[aud_idx]
        if self.visual_transform is not None:
            vis_data = self.visual_transform(vis_data)
        if self.audio_transform is not None:
            aud_data = self.audio_transform(aud_data)
        if not isinstance(vis_data, torch.Tensor):
            vis_data = torch.as_tensor(vis_data, dtype=torch.float32)
        if not isinstance(aud_data, torch.Tensor):
            aud_data = torch.as_tensor(aud_data, dtype=torch.float32)

        # Normalize N-MNIST time dimension to fixed length
        if vis_data.ndim == 4:
            T = vis_data.shape[0]
            if T > NMNIST_TIME_STEPS:
                vis_data = vis_data[:NMNIST_TIME_STEPS]
            elif T < NMNIST_TIME_STEPS:
                pad = torch.zeros(NMNIST_TIME_STEPS - T, *vis_data.shape[1:], dtype=torch.float32)
                vis_data = torch.cat([vis_data, pad], dim=0)

        # Normalize SHD audio to fixed shape (squeeze extra dims from _events_to_dense)
        if aud_data.ndim == 4 and aud_data.shape[1:] == (1, 1, 700):
            aud_data = aud_data.squeeze(1).squeeze(1)  # (100, 700)

        return vis_data, aud_data, c


def paired_collate_fn(batch):
    visuals, audios, labels = zip(*batch)
    return torch.stack(visuals, dim=0), torch.stack(audios, dim=0), torch.tensor(labels, dtype=torch.long)


# ── N-MNIST collate ──────────────────────────────────────────────────────────
NMNIST_TIME_STEPS = 25


def nmnist_collate_fn(batch):
    frames, labels = zip(*batch)
    frames_list = []
    for f in frames:
        f_tensor = torch.as_tensor(f, dtype=torch.float32)
        if f_tensor.shape[0] >= NMNIST_TIME_STEPS:
            frames_list.append(f_tensor[:NMNIST_TIME_STEPS])
        else:
            padding = torch.zeros(
                NMNIST_TIME_STEPS - f_tensor.shape[0], *f_tensor.shape[1:], dtype=torch.float32
            )
            frames_list.append(torch.cat([f_tensor, padding], dim=0))
    return torch.stack(frames_list, dim=0), torch.tensor(labels, dtype=torch.long)


# ── SHD dense conversion & collate ───────────────────────────────────────────

def _events_to_dense(events):
    time_bins, channels = 100, 700
    dense = torch.zeros(time_bins, channels)
    if len(events) > 0:
        max_time = events["t"].max() if len(events) > 0 else 1
        time_indices = (events["t"] / max_time * (time_bins - 1)).astype(int)
        channel_indices = events["x"].astype(int)
        for t, c in zip(time_indices, channel_indices):
            if 0 <= t < time_bins and 0 <= c < channels:
                dense[t, c] = 1.0
    return dense.unsqueeze(1).unsqueeze(1)   # (100, 1, 1, 700)


def shd_collate_fn(batch):
    spikes, labels = zip(*batch)
    return torch.stack(spikes, dim=0).squeeze(2).squeeze(2), torch.tensor(labels, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class VisualStem(nn.Module):
    """N-MNIST frames -> d_stem (time-summed spikes)."""

    def __init__(self, d_stem: int = 256, beta: float = 0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(1024, 512)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc_proj = nn.Linear(512, d_stem)
        self.lif_proj = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        B, T = x.size(0), x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem_proj = self.lif_proj.init_leaky()
        spk_rec = []
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
            fp = self.fc_proj(s4)
            sp, mem_proj = self.lif_proj(fp, mem_proj)
            spk_rec.append(sp)
        return torch.stack(spk_rec, dim=1).sum(dim=1)   # (B, d_stem)


class AudioStem(nn.Module):
    """SHD dense (B,T,700) -> d_stem (time-summed spikes)."""

    def __init__(self, d_stem: int = 256, beta: float = 0.9):
        super().__init__()
        self.fc1 = nn.Linear(700, 1024)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(1024, 1024)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(1024, 512)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc_proj = nn.Linear(512, d_stem)
        self.lif_proj = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        B, T = x.size(0), x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_proj = self.lif_proj.init_leaky()
        spk_rec = []
        for t in range(T):
            xt = x[:, t].view(B, -1)
            s1, mem1 = self.lif1(self.fc1(xt), mem1)
            s2, mem2 = self.lif2(self.fc2(s1), mem2)
            s3, mem3 = self.lif3(self.fc3(s2), mem3)
            fp = self.fc_proj(s3)
            sp, mem_proj = self.lif_proj(fp, mem_proj)
            spk_rec.append(sp)
        return torch.stack(spk_rec, dim=1).sum(dim=1)   # (B, d_stem)


class SharedCore(nn.Module):
    """d_stem -> d_embed, L2-normalised.  SHARED across both modalities."""

    def __init__(self, d_stem: int = 256, d_embed: int = 128, beta: float = 0.9):
        super().__init__()
        self.fc = nn.Linear(d_stem, d_embed)
        self.act = nn.GELU()

    def forward(self, x):
        return F.normalize(self.act(self.fc(x)), dim=1)


class JointSharedEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.visual_stem = VisualStem(d_stem=cfg.d_stem, beta=cfg.beta)
        self.audio_stem = AudioStem(d_stem=cfg.d_stem, beta=cfg.beta)
        self.shared_core = SharedCore(d_stem=cfg.d_stem, d_embed=cfg.d_embed, beta=cfg.beta)
        self.classifier = nn.Linear(cfg.d_embed, NUM_TRAIN_CLASSES)

    def forward_visual(self, x):
        stem = self.visual_stem(x)
        embed = self.shared_core(stem)
        logits = self.classifier(embed)
        return embed, logits

    def forward_audio(self, x):
        stem = self.audio_stem(x)
        embed = self.shared_core(stem)
        logits = self.classifier(embed)
        return embed, logits


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def build_paired_train_val_loaders(cfg: Config, seed: int):
    """Return (train_loader, val_loader) for digits 0-5."""
    # N-MNIST
    sensor_size = tonic.datasets.NMNIST.sensor_size
    nmnist_transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])
    nmnist_full = tonic.datasets.NMNIST(
        save_to=str(DATA_DIR), train=True, transform=nmnist_transform
    )
    cached_nmnist = tonic.DiskCachedDataset(
        nmnist_full, cache_path=str(DATA_DIR / "nmnist_train_cache")
    )

    # SHD (cached dense conversion)
    shd_cache_path = DATA_DIR / "shd_train_dense.pkl"
    if shd_cache_path.exists():
        with open(shd_cache_path, "rb") as f:
            shd_dense = pickle.load(f)
    else:
        shd_full = tonic.datasets.SHD(save_to=str(DATA_DIR), train=True)
        shd_dense = [(_events_to_dense(e), int(l) % 10) for e, l in tqdm(shd_full, desc="  SHD dense", leave=False)]
        with open(shd_cache_path, "wb") as f:
            pickle.dump(shd_dense, f)

    # Filter to train classes
    nmnist_filtered = ClassFilteredDataset(cached_nmnist, TRAIN_CLASSES)
    shd_filtered = ClassFilteredDataset(shd_dense, TRAIN_CLASSES)

    # Build paired dataset
    paired_ds = PairedDigitDatasetFiltered(
        visual_dataset=nmnist_filtered,
        audio_dataset=shd_filtered,
        classes=TRAIN_CLASSES,
        samples_per_class=cfg.samples_per_class,
        seed=seed,
    )

    n_val = int(0.1 * len(paired_ds))
    n_train = len(paired_ds) - n_val
    train_split, val_split = random_split(
        paired_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_split, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, collate_fn=paired_collate_fn
    )
    val_loader = DataLoader(
        val_split, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, drop_last=False, collate_fn=paired_collate_fn
    )
    return train_loader, val_loader


def build_test_loaders(batch_size: int = 64):
    """Return test loaders for ALL digits (we filter by class downstream)."""
    # N-MNIST test
    sensor_size = tonic.datasets.NMNIST.sensor_size
    nmnist_transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])
    nmnist_test = tonic.datasets.NMNIST(
        save_to=str(DATA_DIR), train=False, transform=nmnist_transform
    )
    nmnist_test_loader = DataLoader(
        nmnist_test, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=nmnist_collate_fn
    )

    # SHD test (cached dense conversion)
    shd_test_cache_path = DATA_DIR / "shd_test_dense.pkl"
    if shd_test_cache_path.exists():
        with open(shd_test_cache_path, "rb") as f:
            shd_test_dense = pickle.load(f)
    else:
        shd_test_raw = tonic.datasets.SHD(save_to=str(DATA_DIR), train=False)
        shd_test_dense = [(_events_to_dense(e), int(l)) for e, l in tqdm(shd_test_raw, desc="  SHD test dense", leave=False)]
        with open(shd_test_cache_path, "wb") as f:
            pickle.dump(shd_test_dense, f)
    shd_test_loader = DataLoader(
        shd_test_dense, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=shd_collate_fn
    )
    return {"nmnist": nmnist_test_loader, "shd": shd_test_loader}


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, xmscl, optimizer, device, shuffled_labels: bool = False):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_scl = 0.0
    n_batches = 0

    for vis, aud, labels in loader:
        vis = vis.to(device)
        aud = aud.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        vis_embed, vis_logits = model.forward_visual(vis)
        aud_embed, aud_logits = model.forward_audio(aud)

        # CE on both modalities
        ce_vis = F.cross_entropy(vis_logits, labels)
        ce_aud = F.cross_entropy(aud_logits, labels)
        ce_loss = ce_vis + ce_aud

        # SupCon: concatenate both modalities
        all_embed = torch.cat([vis_embed, aud_embed], dim=0)
        all_labels = torch.cat([labels, labels], dim=0)

        if shuffled_labels:
            perm = torch.randperm(all_labels.size(0))
            all_labels = all_labels[perm]

        scl_loss = xmscl(all_embed, all_labels)

        loss = ce_loss + scl_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_scl += scl_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "ce": total_ce / n_batches,
        "scl": total_scl / n_batches,
    }


@torch.no_grad()
def validate(model, loader, xmscl, device, shuffled_labels: bool = False):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_scl = 0.0
    n_batches = 0
    correct_vis = 0
    correct_aud = 0
    total_samples = 0

    for vis, aud, labels in loader:
        vis = vis.to(device)
        aud = aud.to(device)
        labels = labels.to(device)

        vis_embed, vis_logits = model.forward_visual(vis)
        aud_embed, aud_logits = model.forward_audio(aud)

        ce_vis = F.cross_entropy(vis_logits, labels)
        ce_aud = F.cross_entropy(aud_logits, labels)
        ce_loss = ce_vis + ce_aud

        all_embed = torch.cat([vis_embed, aud_embed], dim=0)
        all_labels = torch.cat([labels, labels], dim=0)
        if shuffled_labels:
            perm = torch.randperm(all_labels.size(0))
            all_labels = all_labels[perm]
        scl_loss = xmscl(all_embed, all_labels)

        loss = ce_loss + scl_loss
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_scl += scl_loss.item()
        n_batches += 1

        correct_vis += (vis_logits.argmax(dim=1) == labels).sum().item()
        correct_aud += (aud_logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "loss": total_loss / n_batches,
        "ce": total_ce / n_batches,
        "scl": total_scl / n_batches,
        "acc_vis": correct_vis / total_samples,
        "acc_aud": correct_aud / total_samples,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION: extract embeddings for test sets
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_test_embeddings(model, test_loaders, device):
    model.eval()
    out = {}
    for modality, loader in test_loaders.items():
        all_embeds = []
        all_labels = []
        for data, labels in tqdm(loader, desc=f"Extract {modality}", leave=False):
            data = data.to(device)
            if modality == "nmnist":
                embed, _ = model.forward_visual(data)
            else:
                embed, _ = model.forward_audio(data)
            all_embeds.append(embed.cpu())
            all_labels.append(labels)
        out[modality] = (
            torch.cat(all_embeds, dim=0),   # (N, d_embed)
            torch.cat(all_labels, dim=0),   # (N,)
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS (copied faithfully from v2)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_in_domain_metrics(vis_embed, vis_labels, aud_embed, aud_labels, classes=TRAIN_CLASSES):
    """Compute cross-modal retrieval and prototype alignment for in-domain classes."""
    vis_embed = vis_embed.numpy()
    aud_embed = aud_embed.numpy()
    vis_labels = vis_labels.numpy()
    aud_labels = aud_labels.numpy()

    # Prototypes
    pv = {c: vis_embed[vis_labels == c].mean(axis=0) for c in classes}
    pa = {c: aud_embed[aud_labels == c].mean(axis=0) for c in classes}
    pv = {c: v / (np.linalg.norm(v) + 1e-8) for c, v in pv.items()}
    pa = {c: a / (np.linalg.norm(a) + 1e-8) for c, a in pa.items()}

    # Prototype alignment
    same_sims = [np.dot(pv[c], pa[c]) for c in classes]
    proto_align = float(np.mean(same_sims))

    # V2A retrieval
    v2a_correct = 0
    for c in classes:
        sims = {d: np.dot(pv[c], pa[d]) for d in classes}
        if max(sims, key=sims.get) == c:
            v2a_correct += 1
    v2a_retrieval = v2a_correct / len(classes)

    # A2V retrieval
    a2v_correct = 0
    for c in classes:
        sims = {d: np.dot(pa[c], pv[d]) for d in classes}
        if max(sims, key=sims.get) == c:
            a2v_correct += 1
    a2v_retrieval = a2v_correct / len(classes)

    return {
        "in_domain_proto_align": proto_align,
        "in_domain_v2a_retrieval": v2a_retrieval,
        "in_domain_a2v_retrieval": a2v_retrieval,
    }


def compute_held_out_metrics(vis_embed, vis_labels, aud_embed, aud_labels, held_out=HELD_OUT_CLASSES):
    """Deterministic null + retrieval on held-out classes."""
    vis_embed = vis_embed.numpy()
    aud_embed = aud_embed.numpy()
    vis_labels = vis_labels.numpy()
    aud_labels = aud_labels.numpy()

    # Prototypes
    pv = {c: vis_embed[vis_labels == c].mean(axis=0) for c in held_out}
    pa = {c: aud_embed[aud_labels == c].mean(axis=0) for c in held_out}
    pv = {c: v / (np.linalg.norm(v) + 1e-8) for c, v in pv.items()}
    pa = {c: a / (np.linalg.norm(a) + 1e-8) for c, a in pa.items()}

    # Same-class alignment
    same_sims = [np.dot(pv[c], pa[c]) for c in held_out]
    same_align = float(np.mean(same_sims))

    # Null: mean of all off-diagonal pairs
    off_sims = []
    for c in held_out:
        for d in held_out:
            if c != d:
                off_sims.append(np.dot(pv[c], pa[d]))
    null_align = float(np.mean(off_sims)) if off_sims else 0.0
    gap = same_align - null_align

    # V2A retrieval
    v2a_correct = 0
    for c in held_out:
        sims = {d: np.dot(pv[c], pa[d]) for d in held_out}
        if max(sims, key=sims.get) == c:
            v2a_correct += 1
    v2a_retrieval = v2a_correct / len(held_out)

    # A2V retrieval
    a2v_correct = 0
    for c in held_out:
        sims = {d: np.dot(pa[c], pv[d]) for d in held_out}
        if max(sims, key=sims.get) == c:
            a2v_correct += 1
    a2v_retrieval = a2v_correct / len(held_out)

    return {
        "held_out_same_align": same_align,
        "held_out_null_align": null_align,
        "held_out_gap": gap,
        "held_out_v2a_retrieval": v2a_retrieval,
        "held_out_a2v_retrieval": a2v_retrieval,
    }


def run_procrustes(vis_embed, vis_labels, aud_embed, aud_labels,
                   train_classes=TRAIN_CLASSES, held_out_classes=HELD_OUT_CLASSES):
    """Parameter-free orthogonal alignment corroboration in embedding space."""
    vis_embed = vis_embed.numpy()
    aud_embed = aud_embed.numpy()
    vis_labels = vis_labels.numpy()
    aud_labels = aud_labels.numpy()

    V_proto = {c: vis_embed[vis_labels == c].mean(axis=0) for c in list(train_classes) + list(held_out_classes)}
    A_proto = {c: aud_embed[aud_labels == c].mean(axis=0) for c in list(train_classes) + list(held_out_classes)}

    V_train = np.stack([V_proto[c] for c in train_classes])
    A_train = np.stack([A_proto[c] for c in train_classes])
    mean_v = V_train.mean(axis=0)
    mean_a = A_train.mean(axis=0)

    Vc = V_train - mean_v
    Ac = A_train - mean_a

    M = Vc.T @ Ac
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    all_classes = list(train_classes) + list(held_out_classes)
    A_centered = {c: A_proto[c] - mean_a for c in all_classes}

    correct = 0
    same_sims = []
    off_sims = []
    for c in held_out_classes:
        v_transformed = (V_proto[c] - mean_v) @ R
        v_transformed = v_transformed / (np.linalg.norm(v_transformed) + 1e-8)
        sims = {
            d: np.dot(v_transformed, A_centered[d] / (np.linalg.norm(A_centered[d]) + 1e-8))
            for d in all_classes
        }
        best = max(sims, key=sims.get)
        if best == c:
            correct += 1
        same_sims.append(sims[c])
        for d in all_classes:
            if d != c:
                off_sims.append(sims[d])

    retrieval = correct / len(held_out_classes)
    return {
        "procrustes_retrieval": retrieval,
        "procrustes_same_align": float(np.mean(same_sims)),
        "procrustes_null_align": float(np.mean(off_sims)),
        "procrustes_gap": float(np.mean(same_sims)) - float(np.mean(off_sims)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE PER SEED
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_seed(seed: int, cfg: Config, shuffled_labels: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"  SEED {seed}" + ("  [SHUFFLED CONTROL]" if shuffled_labels else ""))
    print(f"{'='*60}")

    # ── Data ──
    print("Building paired loaders (digits 0-5)...")
    train_loader, val_loader = build_paired_train_val_loaders(cfg, seed)

    print("Building test loaders (all digits)...")
    test_loaders = build_test_loaders(batch_size=64)

    # ── Leakage assert ──
    for _, _, labels in train_loader:
        for lbl in labels:
            assert int(lbl.item()) in TRAIN_CLASSES, f"LEAKAGE: label {lbl.item()} in training!"
    for _, _, labels in val_loader:
        for lbl in labels:
            assert int(lbl.item()) in TRAIN_CLASSES, f"LEAKAGE: label {lbl.item()} in validation!"
    print("Leakage assert PASSED.")

    # ── Model ──
    model = JointSharedEncoder(cfg).to(DEVICE)
    xmscl = CrossModalSupConLoss(temperature=cfg.supcon_temperature).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ── Untrained baseline ──
    print("Extracting untrained embeddings...")
    untrained_embs = extract_test_embeddings(model, test_loaders, DEVICE)
    untrained_metrics = compute_held_out_metrics(
        untrained_embs["nmnist"][0], untrained_embs["nmnist"][1],
        untrained_embs["shd"][0], untrained_embs["shd"][1],
    )
    print(f"  Untrained held-out V2A retrieval: {untrained_metrics['held_out_v2a_retrieval']:.4f}")

    # ── Training ──
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, xmscl, optimizer, DEVICE, shuffled_labels)
        val_stats = validate(model, val_loader, xmscl, DEVICE, shuffled_labels)
        epoch_time = time.time() - t0

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_acc_vis={val_stats['acc_vis']:.4f} val_acc_aud={val_stats['acc_aud']:.4f} "
            f"({epoch_time:.1f}s)"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best
    model.load_state_dict(best_state)

    # ── Extract trained embeddings ──
    print("Extracting trained embeddings...")
    trained_embs = extract_test_embeddings(model, test_loaders, DEVICE)

    # Filter test embeddings to relevant classes
    vis_embed_all, vis_lbl_all = trained_embs["nmnist"]
    aud_embed_all, aud_lbl_all = trained_embs["shd"]
    aud_lbl_all = aud_lbl_all % 10

    # In-domain metrics (0-5)
    mask_vis_in = torch.tensor([l.item() in TRAIN_CLASSES for l in vis_lbl_all])
    mask_aud_in = torch.tensor([l.item() in TRAIN_CLASSES for l in aud_lbl_all])
    in_domain = compute_in_domain_metrics(
        vis_embed_all[mask_vis_in], vis_lbl_all[mask_vis_in],
        aud_embed_all[mask_aud_in], aud_lbl_all[mask_aud_in],
        classes=TRAIN_CLASSES,
    )

    # Held-out metrics (6-9)
    mask_vis_ho = torch.tensor([l.item() in HELD_OUT_CLASSES for l in vis_lbl_all])
    mask_aud_ho = torch.tensor([l.item() in HELD_OUT_CLASSES for l in aud_lbl_all])
    held_out = compute_held_out_metrics(
        vis_embed_all[mask_vis_ho], vis_lbl_all[mask_vis_ho],
        aud_embed_all[mask_aud_ho], aud_lbl_all[mask_aud_ho],
        held_out=HELD_OUT_CLASSES,
    )

    # Procrustes (all classes)
    procrustes = run_procrustes(
        vis_embed_all, vis_lbl_all, aud_embed_all, aud_lbl_all,
        train_classes=TRAIN_CLASSES, held_out_classes=HELD_OUT_CLASSES,
    )

    # Classifier accuracy on held-out (should be ~chance since classifier never saw 6-9)
    with torch.no_grad():
        vis_logits_ho = model.classifier(vis_embed_all[mask_vis_ho].to(DEVICE))
        aud_logits_ho = model.classifier(aud_embed_all[mask_aud_ho].to(DEVICE))
        # Map held-out labels 6-9 to random guesses since classifier has 6 outputs
        vis_acc_ho = (vis_logits_ho.argmax(dim=1) == (vis_lbl_all[mask_vis_ho] - 6).clamp(0, 5).to(DEVICE)).float().mean().item()
        aud_acc_ho = (aud_logits_ho.argmax(dim=1) == (aud_lbl_all[mask_aud_ho] - 6).clamp(0, 5).to(DEVICE)).float().mean().item()

    result = {
        "seed": seed,
        "shuffled": int(shuffled_labels),
        "best_epoch": len(history),
        "val_loss": best_val_loss,
        "untrained_held_out_v2a": untrained_metrics["held_out_v2a_retrieval"],
        **in_domain,
        **held_out,
        **procrustes,
        "held_out_classifier_acc_vis": vis_acc_ho,
        "held_out_classifier_acc_aud": aud_acc_ho,
    }

    # Save per-seed CSV
    suffix = "shuffled" if shuffled_labels else "normal"
    df = pd.DataFrame([result])
    df.to_csv(RESULTS_DIR / f"seed{seed}_{suffix}.csv", index=False)

    print(f"  In-domain V2A retrieval:  {result['in_domain_v2a_retrieval']:.4f}")
    print(f"  Held-out V2A retrieval:   {result['held_out_v2a_retrieval']:.4f}")
    print(f"  Held-out gap:             {result['held_out_gap']:.4f}")
    print(f"  Procrustes retrieval:     {result['procrustes_retrieval']:.4f}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = Config()

    # Save config
    with open(RESULTS_DIR / "config.json", "w") as f:
        f.write(cfg.to_json())

    all_results = []

    # Normal runs
    for seed in cfg.seeds:
        result = run_one_seed(seed, cfg, shuffled_labels=False)
        all_results.append(result)

    # Shuffled control (1 seed is enough to verify chance behaviour)
    print("\n" + "=" * 60)
    print("  SHUFFLED LABEL CONTROL")
    print("=" * 60)
    result_shuffled = run_one_seed(cfg.seeds[0], cfg, shuffled_labels=True)
    all_results.append(result_shuffled)

    # Summaries
    normal_results = [r for r in all_results if not r["shuffled"]]
    shuffled_results = [r for r in all_results if r["shuffled"]]

    def _summarise(results):
        keys = [k for k in results[0].keys() if k not in ("seed", "shuffled")]
        rows = []
        for k in keys:
            vals = [r[k] for r in results]
            rows.append({"metric": k, "mean": np.mean(vals), "std": np.std(vals)})
        return pd.DataFrame(rows)

    summary_normal = _summarise(normal_results)
    summary_normal.to_csv(RESULTS_DIR / "summary_normal.csv", index=False)

    summary_shuffled = _summarise(shuffled_results)
    summary_shuffled.to_csv(RESULTS_DIR / "summary_shuffled.csv", index=False)

    # Report
    held_out_v2a_mean = summary_normal.loc[summary_normal["metric"] == "held_out_v2a_retrieval", "mean"].values[0]
    held_out_v2a_std = summary_normal.loc[summary_normal["metric"] == "held_out_v2a_retrieval", "std"].values[0]
    procrustes_mean = summary_normal.loc[summary_normal["metric"] == "procrustes_retrieval", "mean"].values[0]
    procrustes_std = summary_normal.loc[summary_normal["metric"] == "procrustes_retrieval", "std"].values[0]
    in_domain_mean = summary_normal.loc[summary_normal["metric"] == "in_domain_v2a_retrieval", "mean"].values[0]

    shuffled_v2a = summary_shuffled.loc[summary_shuffled["metric"] == "held_out_v2a_retrieval", "mean"].values[0]

    # Verdict
    # Threshold: 25% chance + 10% margin = 35%
    if held_out_v2a_mean > 0.35:
        verdict = "JOINT CO-TRAINING GENERALIZES"
    else:
        verdict = "STILL ENGINEERED / NO GENERALIZATION"

    report = f"""
======================================================================
JOINT SHARED ENCODER — REPORT
======================================================================
PRE-REGISTERED EXPECTATIONS:
- In-domain (0-5) alignment: HIGH by construction (SupCon + shared classifier).
- Held-out (6-9) retrieval > 25% => generalizable cross-modal structure.
- Held-out retrieval above frozen-separate baseline (~chance) => joint training
  recovers what post-hoc alignment could not.

CONFIG:
{cfg.to_json()}

RESULTS (normal, n={len(normal_results)} seeds):
  In-domain V2A retrieval:     {in_domain_mean:.4f}  (expected HIGH)
  Held-out V2A retrieval:      {held_out_v2a_mean:.4f} ± {held_out_v2a_std:.4f}
  Held-out gap:                {summary_normal.loc[summary_normal['metric']=='held_out_gap','mean'].values[0]:.4f} ± {summary_normal.loc[summary_normal['metric']=='held_out_gap','std'].values[0]:.4f}
  Procrustes held-out ret:     {procrustes_mean:.4f} ± {procrustes_std:.4f}

SHUFFLED CONTROL (n={len(shuffled_results)}):
  Held-out V2A retrieval:      {shuffled_v2a:.4f}  (expected ~chance)

HEAD-TO-HEAD vs FROZEN-SEPARATE (v2):
  Frozen-separate held-out V2A:  ~0.22 ± 0.20 (chance = 0.25)
  Frozen-separate Procrustes:     0.00
  Joint-shared held-out V2A:     {held_out_v2a_mean:.4f} ± {held_out_v2a_std:.4f}
  Joint-shared Procrustes:       {procrustes_mean:.4f} ± {procrustes_std:.4f}

VERDICT: {verdict}
======================================================================
"""
    with open(RESULTS_DIR / "report.txt", "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()

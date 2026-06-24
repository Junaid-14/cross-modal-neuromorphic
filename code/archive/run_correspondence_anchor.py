#!/usr/bin/env python3
"""
Cross-Modal Correspondence with Fixed Shared Anchors
======================================================

Tests whether per-modality projection heads over frozen backbones can map
each modality's digit representations to fixed orthonormal concept anchors —
creating cross-modal structure with architecturally-zero forgetting.

Design (locked):
- Frozen backbones + task heads from A_no_align checkpoint
- Trainable projection heads: visual_proj, audio_proj (512 -> 256 -> 64, GELU, L2 norm)
- Fixed anchors: 10 orthonormal unit vectors in R^64, never trained
- Loss: cosine classifier CE( (z @ A.T) / tau, y )
- ONLY projection heads receive gradients

Output: results/hypergraph_anchor/
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tonic
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT))

from m7_trimodal_run_provenance import (
    TASKS,
    ExperimentConfig,
    M7_ContinualAdaptiveModel,
    get_device,
    nmnist_collate_fn,
    shd_collate_fn,
)
from utils.cross_modal_transfer import evaluate_cross_modal_transfer

# Use full TASKS so checkpoint loads; we only use nmnist and shd
DIGIT_TASKS = TASKS

RESULTS_DIR = REPO_ROOT / "results" / "hypergraph_anchor"
CACHE_DIR = RESULTS_DIR / "cache"
CKPT_SOURCE = REPO_ROOT / "results" / "cl_confirmation" / "checkpoints" / "A_no_align_seed42" / "final_shd_shd.pt"


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_params(model: nn.Module, prefix: str) -> str:
    """MD5 hash of all parameters whose names start with prefix."""
    h = hashlib.md5()
    for name, p in sorted(model.named_parameters()):
        if name.startswith(prefix):
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _hash_anchors(anchors: torch.Tensor) -> str:
    return hashlib.md5(anchors.cpu().numpy().tobytes()).hexdigest()


def _build_digit_loaders(cfg: ExperimentConfig):
    """Build N-MNIST and SHD loaders (copied from run_digit_only_alignment.py)."""
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

    shd_train_data = [_events_to_dense(e, l) for e, l in tqdm(shd_train, desc="  SHD train", leave=False)]
    shd_test_data = [_events_to_dense(e, l) for e, l in tqdm(shd_test, desc="  SHD test", leave=False)]
    n_val_shd = int(0.1 * len(shd_train_data))
    shd_train_split, shd_val_split = random_split(
        shd_train_data, [len(shd_train_data) - n_val_shd, n_val_shd],
        generator=torch.Generator().manual_seed(42))
    shd_loaders = {
        "train": DataLoader(shd_train_split, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=shd_collate_fn),
        "val":   DataLoader(shd_val_split,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn),
        "test":  DataLoader(shd_test_data,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn),
    }

    return {
        "nmnist": nmnist_loaders,
        "shd": shd_loaders,
        "nmnist_train_split": train_split,
        "shd_train_split": shd_train_split,
    }


def _extract_backbone_features(model: nn.Module, loader: DataLoader, modality: str, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract 512-d backbone features and labels. Returns (features, labels) on CPU."""
    model.eval()
    all_feats, all_lbls = [], []
    with torch.no_grad():
        for data, labels in tqdm(loader, desc=f"  Extract {modality}", leave=False):
            data = data.to(device)
            bb = model.backbones[modality]
            if modality == "nmnist" or modality == "dvs":
                _, feats = bb(data)
            else:
                B, T = data.size(0), data.size(1)
                mem1 = bb["lif1"].init_leaky()
                mem2 = bb["lif2"].init_leaky()
                mem3 = bb["lif3"].init_leaky()
                feat_list = []
                for t in range(T):
                    xt = data[:, t].view(B, -1)
                    s1, mem1 = bb["lif1"](bb["fc1"](xt), mem1)
                    s2, mem2 = bb["lif2"](bb["fc2"](s1), mem2)
                    s3, mem3 = bb["lif3"](bb["fc3"](s2), mem3)
                    feat_list.append(s3)
                feats = torch.stack(feat_list, dim=1).sum(dim=1)
            all_feats.append(feats.cpu())
            all_lbls.append(labels)
    return torch.cat(all_feats), torch.cat(all_lbls)


def _cache_features(model: nn.Module, loaders: Dict, device) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Extract and cache backbone features. Returns dict of (features, labels)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = {}
    for modality, split in [("nmnist", "train"), ("nmnist", "test"),
                            ("shd", "train"), ("shd", "test")]:
        cache_path = CACHE_DIR / f"{modality}_{split}_features.pt"
        if cache_path.exists():
            print(f"   Loading cached {modality} {split} features...")
            data = torch.load(cache_path, map_location="cpu")
            cached[f"{modality}_{split}"] = (data["features"], data["labels"])
        else:
            print(f"   Extracting {modality} {split} features...")
            if modality == "shd" and split == "train":
                loader = loaders["shd"]["train"]
            elif modality == "shd" and split == "test":
                loader = loaders["shd"]["test"]
            elif modality == "nmnist" and split == "train":
                loader = loaders["nmnist"]["train"]
            else:
                loader = loaders["nmnist"]["test"]
            feats, lbls = _extract_backbone_features(model, loader, modality, device)
            torch.save({"features": feats, "labels": lbls}, cache_path)
            cached[f"{modality}_{split}"] = (feats, lbls)
    return cached


# ═══════════════════════════════════════════════════════════════════════════════
#  CORRESPONDENCE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class CorrespondenceModel(nn.Module):
    """
    Frozen base model + trainable projection heads + fixed anchors.
    Compatible with evaluate_cross_modal_transfer via forward_with_shared.
    """

    def __init__(self, base_model: M7_ContinualAdaptiveModel, anchor_dim: int = 64, temperature: float = 0.1):
        super().__init__()
        self.base_model = base_model
        # Freeze base
        for p in self.base_model.parameters():
            p.requires_grad = False

        shared_dim = base_model.cfg.shared_dim  # 512

        self.visual_proj = nn.Sequential(
            nn.Linear(shared_dim, 256),
            nn.GELU(),
            nn.Linear(256, anchor_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(shared_dim, 256),
            nn.GELU(),
            nn.Linear(256, anchor_dim),
        )

        # Fixed orthonormal anchors: 10 x anchor_dim
        rng = torch.Generator().manual_seed(42)
        rand_mat = torch.randn(anchor_dim, anchor_dim, generator=rng)
        q, _ = torch.linalg.qr(rand_mat)
        anchors = q[:10, :]  # (10, anchor_dim)
        self.register_buffer("anchors", anchors)
        self.temperature = temperature

    def _get_backbone_features(self, x, modality):
        """Extract 512-d features from frozen backbone."""
        bb = self.base_model.backbones[modality]
        if modality == "nmnist" or modality == "dvs":
            _, feats = bb(x)
        else:
            B, T = x.size(0), x.size(1)
            mem1 = bb["lif1"].init_leaky()
            mem2 = bb["lif2"].init_leaky()
            mem3 = bb["lif3"].init_leaky()
            feat_list = []
            for t in range(T):
                xt = x[:, t].view(B, -1)
                s1, mem1 = bb["lif1"](bb["fc1"](xt), mem1)
                s2, mem2 = bb["lif2"](bb["fc2"](s1), mem2)
                s3, mem3 = bb["lif3"](bb["fc3"](s2), mem3)
                feat_list.append(s3)
            feats = torch.stack(feat_list, dim=1).sum(dim=1)
        return feats

    def project(self, features, modality):
        """Project backbone features through trainable head and L2-normalize."""
        if modality == "nmnist":
            z = self.visual_proj(features)
        else:
            z = self.audio_proj(features)
        return F.normalize(z, dim=-1)

    def forward_with_shared(self, x, modality, task_name):
        """
        Compatible with evaluate_cross_modal_transfer.
        Returns: (logits, gate, full, shared) where shared is the projected embedding.
        """
        feats = self._get_backbone_features(x, modality)
        z = self.project(feats, modality)
        # Dummy values for unused outputs; transfer eval only uses 4th element
        dummy_logits = torch.zeros(x.size(0), 10, device=x.device)
        dummy_gate = torch.zeros(x.size(0), 2, device=x.device)
        dummy_full = torch.zeros(x.size(0), self.base_model.cfg.shared_dim, device=x.device)
        return dummy_logits, dummy_gate, dummy_full, z

    def forward(self, x, modality, task_name):
        """Standard forward: uses frozen backbone -> frozen head for task accuracy."""
        return self.base_model(x, modality, task_name)

    def task_accuracy(self, loader, modality, task_name, device):
        """Evaluate accuracy via frozen backbone + frozen head."""
        self.base_model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(device), labels.to(device)
                logits, _, _ = self.base_model(data, modality, task_name)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def build_paired_feature_dataset(
    cached: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    samples_per_class: int = 500,
    seed: int = 42,
    held_out_classes: Tuple[int, ...] = (),
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Build paired (visual_feature, audio_feature, shared_label) datasets from cached features.
    Returns (train_dataset, val_dataset).
    """
    vis_feats, vis_lbls = cached["nmnist_train"]
    aud_feats, aud_lbls = cached["shd_train"]

    # SHD labels are already 0-19 in raw data; map to 0-9 identity
    aud_lbls = aud_lbls % 10

    # Subsample to balanced class counts
    rng = np.random.default_rng(seed)
    vis_indices = []
    aud_indices = []
    for c in range(10):
        vis_idx = np.where(vis_lbls.numpy() == c)[0]
        aud_idx = np.where(aud_lbls.numpy() == c)[0]
        if len(vis_idx) == 0 or len(aud_idx) == 0:
            continue
        n = min(samples_per_class, len(vis_idx), len(aud_idx))
        vis_indices.extend(rng.choice(vis_idx, size=n, replace=False).tolist())
        aud_indices.extend(rng.choice(aud_idx, size=n, replace=False).tolist())

    # Actually we need PAIRED samples with the SAME label
    # Better: for each class c, take min(vis_count, aud_count) pairs
    paired_vis = []
    paired_aud = []
    paired_lbl = []
    for c in range(10):
        vis_idx = np.where(vis_lbls.numpy() == c)[0]
        aud_idx = np.where(aud_lbls.numpy() == c)[0]
        if len(vis_idx) == 0 or len(aud_idx) == 0:
            continue
        n = min(samples_per_class, len(vis_idx), len(aud_idx))
        vis_sel = rng.choice(vis_idx, size=n, replace=False)
        aud_sel = rng.choice(aud_idx, size=n, replace=False)
        paired_vis.append(vis_feats[vis_sel])
        paired_aud.append(aud_feats[aud_sel])
        paired_lbl.append(torch.full((n,), c, dtype=torch.long))

    paired_vis = torch.cat(paired_vis)
    paired_aud = torch.cat(paired_aud)
    paired_lbl = torch.cat(paired_lbl)

    # Shuffle
    perm = torch.randperm(len(paired_lbl), generator=torch.Generator().manual_seed(seed))
    paired_vis = paired_vis[perm]
    paired_aud = paired_aud[perm]
    paired_lbl = paired_lbl[perm]

    # Train/val split (90/10)
    n_val = int(0.1 * len(paired_lbl))
    n_train = len(paired_lbl) - n_val

    train_ds = TensorDataset(paired_vis[:n_train], paired_aud[:n_train], paired_lbl[:n_train])
    val_ds = TensorDataset(paired_vis[n_train:], paired_aud[n_train:], paired_lbl[n_train:])
    return train_ds, val_ds


def train_correspondence(
    model: CorrespondenceModel,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    device,
    num_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 128,
    patience: int = 5,
    held_out_classes: Tuple[int, ...] = (),
    shuffled_labels: bool = False,
) -> Dict:
    """Train projection heads. Returns history dict."""
    optimizer = torch.optim.AdamW(
        list(model.visual_proj.parameters()) + list(model.audio_proj.parameters()),
        lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    ce = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for vis_feats, aud_feats, labels in train_loader:
            vis_feats = vis_feats.to(device)
            aud_feats = aud_feats.to(device)
            labels = labels.to(device)

            if shuffled_labels:
                # Shuffle labels within batch for control experiment
                labels = labels[torch.randperm(len(labels))]

            # Project and normalize
            vis_z = F.normalize(model.visual_proj(vis_feats), dim=-1)
            aud_z = F.normalize(model.audio_proj(aud_feats), dim=-1)

            # Mask held-out classes (loss = 0 for them)
            mask = torch.ones_like(labels, dtype=torch.bool)
            for hc in held_out_classes:
                mask = mask & (labels != hc)

            # Cosine classifier
            logits_vis = (vis_z @ model.anchors.T) / model.temperature
            logits_aud = (aud_z @ model.anchors.T) / model.temperature

            loss_vis = ce(logits_vis, labels)
            loss_aud = ce(logits_aud, labels)
            loss = loss_vis + loss_aud

            # Apply mask: zero out loss for held-out classes
            if len(held_out_classes) > 0:
                loss = loss * mask.float()
                loss = loss.sum() / (mask.sum().clamp(min=1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.visual_proj.parameters()) + list(model.audio_proj.parameters()), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = train_loss_sum / n_batches

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for vis_feats, aud_feats, labels in val_loader:
                vis_feats = vis_feats.to(device)
                aud_feats = aud_feats.to(device)
                labels = labels.to(device)

                vis_z = F.normalize(model.visual_proj(vis_feats), dim=-1)
                aud_z = F.normalize(model.audio_proj(aud_feats), dim=-1)

                logits_vis = (vis_z @ model.anchors.T) / model.temperature
                logits_aud = (aud_z @ model.anchors.T) / model.temperature
                loss = ce(logits_vis, labels) + ce(logits_aud, labels)
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / val_batches
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"  Epoch {epoch+1:02d}/{num_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    return history


# ═══════════════════════════════════════════════════════════════════════════════
#  HELD-OUT-CLASS PROTOTYPE ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_held_out_alignment(
    model: CorrespondenceModel,
    cached: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    held_out_classes: Tuple[int, ...] = (8, 9),
    device=None,
) -> Dict[str, float]:
    """
    Compute cross-modal prototype alignment for held-out classes.
    Returns {same_class, off_diag, null_baseline}.
    """
    vis_feats, vis_lbls = cached["nmnist_test"]
    aud_feats, aud_lbls = cached["shd_test"]
    aud_lbls = aud_lbls % 10

    model.eval()
    with torch.no_grad():
        vis_z = F.normalize(model.visual_proj(vis_feats.to(device)), dim=-1).cpu().numpy()
        aud_z = F.normalize(model.audio_proj(aud_feats.to(device)), dim=-1).cpu().numpy()

    vis_lbls = vis_lbls.numpy()
    aud_lbls = aud_lbls.numpy()

    # Compute prototypes for held-out classes
    prototypes_v = {}
    prototypes_a = {}
    for c in held_out_classes:
        prototypes_v[c] = vis_z[vis_lbls == c].mean(axis=0)
        prototypes_a[c] = aud_z[aud_lbls == c].mean(axis=0)

    # Normalize
    for c in held_out_classes:
        prototypes_v[c] = prototypes_v[c] / (np.linalg.norm(prototypes_v[c]) + 1e-8)
        prototypes_a[c] = prototypes_a[c] / (np.linalg.norm(prototypes_a[c]) + 1e-8)

    # Same-class alignment
    same_class_sims = []
    for c in held_out_classes:
        sim = np.dot(prototypes_v[c], prototypes_a[c])
        same_class_sims.append(sim)
    same_class = float(np.mean(same_class_sims))

    # Off-diagonal (visual c vs audio d, c != d)
    off_diag_sims = []
    for c in held_out_classes:
        for d in held_out_classes:
            if c != d:
                sim = np.dot(prototypes_v[c], prototypes_a[d])
                off_diag_sims.append(sim)
    off_diag = float(np.mean(off_diag_sims)) if off_diag_sims else 0.0

    # Null baseline: shuffle audio prototypes
    shuffled_classes = list(held_out_classes)
    np.random.shuffle(shuffled_classes)
    null_sims = []
    for c, d in zip(held_out_classes, shuffled_classes):
        sim = np.dot(prototypes_v[c], prototypes_a[d])
        null_sims.append(sim)
    null_baseline = float(np.mean(null_sims))

    return {
        "held_out_same_class": same_class,
        "held_out_off_diag": off_diag,
        "held_out_null_baseline": null_baseline,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_seed(seed: int, cfg: ExperimentConfig, loaders: Dict, cached: Dict, device) -> Dict:
    """Run correspondence experiment for one seed."""
    print(f"\n{'='*70}")
    print(f"  SEED {seed}")
    print(f"{'='*70}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load frozen base model
    base_model = M7_ContinualAdaptiveModel(tasks=DIGIT_TASKS, cfg=cfg).to(device)
    ckpt = torch.load(CKPT_SOURCE, map_location=str(device))
    base_model.load_state_dict(ckpt["model_state_dict"])

    # Hash backbone + head params BEFORE
    hash_nmnist_before = _hash_params(base_model, "backbones.nmnist")
    hash_shd_before = _hash_params(base_model, "backbones.shd")
    hash_heads_before = _hash_params(base_model, "heads.")

    # Create correspondence model
    corr_model = CorrespondenceModel(base_model, anchor_dim=64, temperature=0.1).to(device)
    anchors_before = corr_model.anchors.clone()

    # ── 1. Task accuracy BEFORE alignment ──
    acc_nmnist_before = corr_model.task_accuracy(loaders["nmnist"]["test"], "nmnist", "nmnist", device)
    acc_shd_before = corr_model.task_accuracy(loaders["shd"]["test"], "shd", "shd", device)
    print(f"   N-MNIST before: {acc_nmnist_before:.2f}%")
    print(f"   SHD before:     {acc_shd_before:.2f}%")

    # ── 2. Untrained-head control ──
    print("\n   [Control] Untrained projection heads...")
    untrained_transfer = evaluate_cross_modal_transfer(
        corr_model, loaders["nmnist"]["test"], loaders["shd"]["test"], device,
        num_classes=10, samples_per_class=100, random_state=seed,
    )
    untrained_v2a = untrained_transfer["v2a"] * 100
    print(f"   Untrained V2A: {untrained_v2a:.2f}%")
    assert untrained_v2a < 15.0, f"Untrained V2A too high: {untrained_v2a:.2f}%"

    # ── 3. Train projection heads (full supervision, 0-9) ──
    print("\n   Training projection heads (digits 0-9)...")
    train_ds, val_ds = build_paired_feature_dataset(cached, samples_per_class=500, seed=seed)
    history = train_correspondence(
        corr_model, train_ds, val_ds, device,
        num_epochs=20, lr=1e-3, batch_size=128, patience=5,
    )

    # ── 4. Task accuracy AFTER alignment ──
    acc_nmnist_after = corr_model.task_accuracy(loaders["nmnist"]["test"], "nmnist", "nmnist", device)
    acc_shd_after = corr_model.task_accuracy(loaders["shd"]["test"], "shd", "shd", device)
    print(f"   N-MNIST after:  {acc_nmnist_after:.2f}%")
    print(f"   SHD after:      {acc_shd_after:.2f}%")

    # ── 5. Invariant asserts ──
    hash_nmnist_after = _hash_params(base_model, "backbones.nmnist")
    hash_shd_after = _hash_params(base_model, "backbones.shd")
    hash_heads_after = _hash_params(base_model, "heads.")
    assert hash_nmnist_before == hash_nmnist_after, "N-MNIST backbone drifted!"
    assert hash_shd_before == hash_shd_after, "SHD backbone drifted!"
    assert hash_heads_before == hash_heads_after, "Task heads drifted!"
    assert torch.allclose(anchors_before, corr_model.anchors), "Anchors changed!"
    assert abs(acc_nmnist_before - acc_nmnist_after) < 0.01, "N-MNIST accuracy changed!"
    assert abs(acc_shd_before - acc_shd_after) < 0.01, "SHD accuracy changed!"
    print("   ✅ All invariant asserts passed")

    # ── 6. Transfer eval on trained heads ──
    print("\n   [Eval] Transfer eval on trained heads...")
    trained_transfer = evaluate_cross_modal_transfer(
        corr_model, loaders["nmnist"]["test"], loaders["shd"]["test"], device,
        num_classes=10, samples_per_class=100, random_state=seed,
    )

    # ── 7. Held-out-class test (digits 8,9) ──
    print("\n   [Eval] Held-out-class test (digits 8,9)...")
    # Retrain from scratch with only 0-7
    base_model2 = M7_ContinualAdaptiveModel(tasks=DIGIT_TASKS, cfg=cfg).to(device)
    base_model2.load_state_dict(ckpt["model_state_dict"])
    corr_model2 = CorrespondenceModel(base_model2, anchor_dim=64, temperature=0.1).to(device)

    train_ds2, val_ds2 = build_paired_feature_dataset(cached, samples_per_class=500, seed=seed)
    train_correspondence(
        corr_model2, train_ds2, val_ds2, device,
        num_epochs=20, lr=1e-3, batch_size=128, patience=5,
        held_out_classes=(8, 9),
    )
    held_out = compute_held_out_alignment(corr_model2, cached, held_out_classes=(8, 9), device=device)
    print(f"   Held-out same-class: {held_out['held_out_same_class']:.4f}")
    print(f"   Held-out null:       {held_out['held_out_null_baseline']:.4f}")

    # ── 8. Label-shuffled control ──
    print("\n   [Control] Label-shuffled training...")
    base_model3 = M7_ContinualAdaptiveModel(tasks=DIGIT_TASKS, cfg=cfg).to(device)
    base_model3.load_state_dict(ckpt["model_state_dict"])
    corr_model3 = CorrespondenceModel(base_model3, anchor_dim=64, temperature=0.1).to(device)
    train_correspondence(
        corr_model3, train_ds, val_ds, device,
        num_epochs=20, lr=1e-3, batch_size=128, patience=5,
        shuffled_labels=True,
    )
    shuffled_transfer = evaluate_cross_modal_transfer(
        corr_model3, loaders["nmnist"]["test"], loaders["shd"]["test"], device,
        num_classes=10, samples_per_class=100, random_state=seed,
    )
    shuffled_v2a = shuffled_transfer["v2a"] * 100
    print(f"   Shuffled V2A: {shuffled_v2a:.2f}%")
    assert shuffled_v2a < 15.0, f"Shuffled V2A too high: {shuffled_v2a:.2f}%"

    # ── 9. Verdict ──
    verdict = "discovered shared structure" if held_out["held_out_same_class"] > held_out["held_out_null_baseline"] + 0.1 else "engineered bridge"
    print(f"\n   🏛 VERDICT: {verdict.upper()}")

    return {
        "seed": seed,
        "nmnist_before": acc_nmnist_before,
        "nmnist_after": acc_nmnist_after,
        "shd_before": acc_shd_before,
        "shd_after": acc_shd_after,
        "untrained_v2a": untrained_v2a,
        "trained_v2a": trained_transfer["v2a"] * 100,
        "trained_a2v": trained_transfer["a2v"] * 100,
        "trained_proto_align": trained_transfer["prototype_alignment"],
        "trained_sil_visual": trained_transfer["sil_visual"],
        "trained_sil_audio": trained_transfer["sil_audio"],
        "held_out_same_class": held_out["held_out_same_class"],
        "held_out_off_diag": held_out["held_out_off_diag"],
        "held_out_null_baseline": held_out["held_out_null_baseline"],
        "shuffled_v2a": shuffled_v2a,
        "verdict": verdict,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        seed=42, task_order=["nmnist", "shd", "dvs"],
        lr=1e-3, weight_decay=1e-4, batch_size=32,
        max_epochs=50, patience=12, gradient_clip=1.0,
        beta=0.9, dropout=0.2, hidden_dim=512, shared_dim=512,
        num_patterns=100, hopfield_temperature=1.0, num_gru_layers=2,
        num_workers=0, time_steps=25, pin_memory=True,
        use_contrastive=True, contrastive_temperature=0.07, contrastive_weight=0.1,
        use_replay=False, replay_batch=32, replay_weight=0.3, replay_samples_per_class=50,
        device="auto",
        align_cross_modal=False, align_shared_dim=512, align_weight=0.5,
        align_temperature=0.1, align_use_orthogonal=False, align_orthogonal_weight=0.01,
    )
    device = get_device(cfg)
    print(f"Device: {device}")

    print("\nBuilding digit loaders...")
    loaders = _build_digit_loaders(cfg)

    print("\nLoading frozen base model...")
    base_model = M7_ContinualAdaptiveModel(tasks=DIGIT_TASKS, cfg=cfg).to(device)
    ckpt = torch.load(CKPT_SOURCE, map_location=str(device))
    base_model.load_state_dict(ckpt["model_state_dict"])
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    print("\nCaching backbone features...")
    cached = _cache_features(base_model, loaders, device)

    seeds = [42, 123, 456]
    all_results = []
    for seed in seeds:
        try:
            res = run_one_seed(seed, cfg, loaders, cached, device)
            all_results.append(res)
            # Write per-seed CSV
            csv_path = RESULTS_DIR / f"seed{seed}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(res.keys()))
                writer.writeheader()
                writer.writerow(res)
        except Exception as e:
            print(f"\n❌ Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate summary
    if all_results:
        print("\n" + "=" * 70)
        print("AGGREGATE SUMMARY")
        print("=" * 70)
        df_data = {}
        for key in all_results[0].keys():
            if key == "seed" or key == "verdict":
                continue
            vals = [r[key] for r in all_results]
            df_data[key] = {"mean": np.mean(vals), "std": np.std(vals)}
            print(f"   {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        # Write summary CSV
        summary_path = RESULTS_DIR / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean", "std"])
            for key, stats in df_data.items():
                writer.writerow([key, f"{stats['mean']:.6f}", f"{stats['std']:.6f}"])

        # Verdicts
        verdicts = [r["verdict"] for r in all_results]
        majority = "discovered shared structure" if verdicts.count("discovered shared structure") > len(verdicts) / 2 else "engineered bridge"
        print(f"\n   Majority verdict: {majority.upper()}")

        # Baseline comparison
        print("\n" + "=" * 70)
        print("BASELINE COMPARISON")
        print("=" * 70)
        mean_v2a = np.mean([r["trained_v2a"] for r in all_results])
        mean_retention = np.mean([r["nmnist_after"] for r in all_results])
        print(f"   Condition                    | Retention | V2A")
        print(f"   A_no_align                   | ~97.0%    | ~13.3%")
        print(f"   Frozen-C                     | 13.3%     | 14.8%")
        print(f"   Unfrozen + rehearsal         | 95.7%     | 14.4%")
        print(f"   Correspondence (this run)    | {mean_retention:.1f}%    | {mean_v2a:.1f}%")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()

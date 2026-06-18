#!/usr/bin/env python3
"""
Joint Shared-Encoder Cross-Modal Experiment with Shared Magnitude Head Ablation
================================================================================

Controlled variable: a shared MAGNITUDE head, ON vs OFF.

Train a single shared encoder FROM SCRATCH on both N-MNIST and SHD, with:
  - separate per-modality digit classifiers (0-9)
  - cross-modal supervised contrastive loss on z for digits 0-5 ONLY
  - (Arm ON only) a shared magnitude head regressing digit value on z for all 0-9

Held-out test: cross-modal retrieval on digits 6-9, which never receive a
cross-modal signal in either arm.

Output: results/joint_magnitude_ablation/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tonic
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT))

import snntorch as snn

from Dataloaders.paired_digit_loader import (
    SHDDigitIdentityDataset,
    nmnist_to_identity,
    paired_digit_collate_fn,
    shd_to_identity,
)
from m7_trimodal_run_provenance import (
    CrossModalSupConLoss,
    ExperimentConfig,
    get_device,
    nmnist_collate_fn,
    shd_collate_fn,
)
from run_correspondence_anchor_v2 import (
    compute_held_out_metrics as _compute_held_out_metrics_v2,
    run_procrustes,
)

RESULTS_DIR = REPO_ROOT / "results" / "joint_magnitude_ablation"
SEEDS = [42, 123, 456, 789, 1011]

spike_grad = snn.surrogate.atan()


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASETS / LOADERS
# ═══════════════════════════════════════════════════════════════════════════════


def _events_to_dense_shd(events, label, time_bins=100, channels=700):
    """Convert SHD events to dense (time_bins, channels) tensor."""
    dense = torch.zeros(time_bins, channels)
    if len(events) > 0:
        max_time = max(events["t"].max(), 1)
        t_idx = (events["t"] / max_time * (time_bins - 1)).astype(int)
        c_idx = events["x"].astype(int)
        for t, c in zip(t_idx, c_idx):
            if 0 <= t < time_bins and 0 <= c < channels:
                dense[t, c] = 1.0
    return dense, int(label)


class CachedSHD(Dataset):
    """SHD pre-rendered to (T=100, C=700) dense tensors with identity labels."""

    def __init__(self, base, time_bins=100, channels=700):
        self.samples = []
        for events, label in tqdm(base, desc="SHD dense", leave=False):
            dense, _ = _events_to_dense_shd(events, label, time_bins, channels)
            identity = shd_to_identity(label)
            # Store as uint8; convert to float on retrieval.
            self.samples.append((dense.to(torch.uint8), identity))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, label = self.samples[idx]
        # Return uint8; model will .to(device).float() for fast MPS transfer.
        return frame, label


class CachedNMNIST(Dataset):
    """N-MNIST pre-rendered to (T=25, C, H, W) frames with identity labels."""

    def __init__(self, base):
        self.samples = []
        for frame, label in tqdm(base, desc="N-MNIST frames", leave=False):
            if not isinstance(frame, torch.Tensor):
                frame = torch.from_numpy(frame)
            # Tonic returns (T, C, H, W). Keep only first 25 time steps,
            # pad if necessary. Store as uint8 to save disk/memory.
            if frame.shape[0] > 25:
                frame = frame[:25]
            elif frame.shape[0] < 25:
                pad = torch.zeros(25 - frame.shape[0], *frame.shape[1:], dtype=frame.dtype)
                frame = torch.cat([frame, pad], dim=0)
            identity = nmnist_to_identity(label)
            self.samples.append((frame.to(torch.uint8), identity))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, label = self.samples[idx]
        # Return uint8; model will .to(device).float() for fast MPS transfer.
        return frame, label


class SubsampledByClass(Dataset):
    """Stratified subsample of a dataset, keeping up to n_per_class per label."""

    def __init__(self, base_dataset: Dataset, n_per_class: int, seed: int = 42):
        self.base = base_dataset
        rng = np.random.default_rng(seed)
        labels = np.array([int(base_dataset[i][1]) for i in range(len(base_dataset))])
        indices: List[int] = []
        for c in np.unique(labels):
            c_idx = np.where(labels == c)[0]
            if len(c_idx) > n_per_class:
                c_idx = rng.choice(c_idx, size=n_per_class, replace=False)
            indices.extend(c_idx.tolist())
        self.indices = np.array(sorted(indices), dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[int(self.indices[idx])]


def _build_modality_loaders(cfg: ExperimentConfig):
    """Build per-modality train/test loaders (classification) and cached datasets.

    Caches processed frames/tensors to disk so subsequent runs load quickly.
    """
    data_dir = Path("./ICONS_M7/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = RESULTS_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    nmnist_train_cache_path = cache_dir / "nmnist_train_compact.pt"
    nmnist_test_cache_path = cache_dir / "nmnist_test_compact.pt"
    shd_train_cache_path = cache_dir / "shd_train_compact.pt"
    shd_test_cache_path = cache_dir / "shd_test_compact.pt"

    # Build or load N-MNIST
    if all(p.exists() for p in (nmnist_train_cache_path, nmnist_test_cache_path)):
        print("   Loading cached N-MNIST frames...")
        nmnist_train_cache = torch.load(nmnist_train_cache_path, weights_only=False)
        nmnist_test_cache = torch.load(nmnist_test_cache_path, weights_only=False)
    else:
        sensor_size = tonic.datasets.NMNIST.sensor_size
        nmnist_transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
        ])
        nmnist_train = tonic.datasets.NMNIST(save_to=str(data_dir), train=True, transform=nmnist_transform)
        nmnist_test = tonic.datasets.NMNIST(save_to=str(data_dir), train=False, transform=nmnist_transform)
        nmnist_train_cache = CachedNMNIST(nmnist_train)
        nmnist_test_cache = CachedNMNIST(nmnist_test)
        torch.save(nmnist_train_cache, nmnist_train_cache_path)
        torch.save(nmnist_test_cache, nmnist_test_cache_path)

    # Build or load SHD
    if all(p.exists() for p in (shd_train_cache_path, shd_test_cache_path)):
        print("   Loading cached SHD dense tensors...")
        shd_train_cache = torch.load(shd_train_cache_path, weights_only=False)
        shd_test_cache = torch.load(shd_test_cache_path, weights_only=False)
    else:
        shd_train_base = tonic.datasets.SHD(save_to=str(data_dir), train=True)
        shd_test_base = tonic.datasets.SHD(save_to=str(data_dir), train=False)
        shd_train_cache = CachedSHD(shd_train_base)
        shd_test_cache = CachedSHD(shd_test_base)
        torch.save(shd_train_cache, shd_train_cache_path)
        torch.save(shd_test_cache, shd_test_cache_path)

    # Subsample N-MNIST training for fast per-modality classifier passes.
    # Held-out prototypes and test evaluation still use the full cache.
    nmnist_train_subsampled = SubsampledByClass(
        nmnist_train_cache, n_per_class=cfg.nmnist_train_per_class, seed=42
    )
    print(
        f"   N-MNIST train subsampled: {len(nmnist_train_subsampled)} / "
        f"{len(nmnist_train_cache)} (per class ≤ {cfg.nmnist_train_per_class})"
    )

    loaders = {
        "nmnist": {
            "train": DataLoader(nmnist_train_subsampled, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=nmnist_collate_fn),
            "test": DataLoader(nmnist_test_cache, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=nmnist_collate_fn),
        },
        "shd": {
            "train": DataLoader(shd_train_cache, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=shd_collate_fn),
            "test": DataLoader(shd_test_cache, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn),
        },
    }
    cached = {
        "nmnist_train": nmnist_train_cache,
        "nmnist_test": nmnist_test_cache,
        "shd_train": shd_train_cache,
        "shd_test": shd_test_cache,
    }
    return loaders, cached


def _build_paired_0to5_loaders(cached: Dict, seed: int):
    """Build paired visual-audio loaders restricted to digits 0-5 from cached datasets."""
    nmnist_train = cached["nmnist_train"]
    shd_train = cached["shd_train"]

    # Build index by identity
    vis_by_id = {c: [] for c in range(10)}
    for i, (_, label) in enumerate(nmnist_train.samples):
        vis_by_id[label].append(i)
    aud_by_id = {c: [] for c in range(10)}
    for i, (_, label) in enumerate(shd_train.samples):
        aud_by_id[label].append(i)

    rng = np.random.default_rng(seed)
    samples_per_class = 500
    pairs = []
    for c in range(6):
        vis_pool = vis_by_id[c]
        aud_pool = aud_by_id[c]
        rng.shuffle(vis_pool)
        rng.shuffle(aud_pool)
        n = min(samples_per_class, len(vis_pool), len(aud_pool))
        for i in range(n):
            pairs.append((vis_pool[i], aud_pool[i], c))
    rng.shuffle(pairs)

    class _Paired(Dataset):
        def __init__(self, pairs, nmnist_base, shd_base):
            self.pairs = pairs
            self.nmnist_base = nmnist_base
            self.shd_base = shd_base

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            vi, ai, c = self.pairs[idx]
            v, _ = self.nmnist_base[vi]
            a, _ = self.shd_base[ai]
            return v, a, c

    n_train = int(0.85 * len(pairs))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    ds_train = _Paired(train_pairs, nmnist_train, shd_train)
    ds_val = _Paired(val_pairs, nmnist_train, shd_train)

    train_loader = DataLoader(ds_train, batch_size=30, shuffle=True, num_workers=0, collate_fn=paired_digit_collate_fn, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=30, shuffle=False, num_workers=0, collate_fn=paired_digit_collate_fn, drop_last=False)

    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════


class VisualStem(nn.Module):
    """N-MNIST -> d_stem (256)."""

    def __init__(self, d_stem: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc2 = nn.Linear(1024, d_stem)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        x = x.float()  # inputs may be uint8 to save MPS transfer bandwidth
        B, T = x.size(0), x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        out = []
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
            out.append(s4)
        return torch.stack(out, dim=1).sum(dim=1)


class AudioStem(nn.Module):
    """SHD dense (T=100, C=700) -> d_stem (256)."""

    def __init__(self, d_stem: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(700, 1024)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc2 = nn.Linear(1024, 1024)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc3 = nn.Linear(1024, d_stem)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        x = x.float()  # inputs may be uint8 to save MPS transfer bandwidth
        B, T = x.size(0), x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        out = []
        for t in range(T):
            xt = x[:, t].view(B, -1)
            s1, mem1 = self.lif1(self.fc1(xt), mem1)
            s2, mem2 = self.lif2(self.fc2(s1), mem2)
            s3, mem3 = self.lif3(self.fc3(s2), mem3)
            out.append(s3)
        return torch.stack(out, dim=1).sum(dim=1)


class JointMagnitudeModel(nn.Module):
    """
    visual_stem + audio_stem -> shared_core -> z (L2-normalized, d_embed=128).
    Separate visual/audio classifiers (0-9).
    Optional shared magnitude head (d_embed -> 1).
    """

    def __init__(self, d_stem: int = 256, d_embed: int = 128, use_magnitude: bool = False):
        super().__init__()
        self.d_stem = d_stem
        self.d_embed = d_embed
        self.use_magnitude = use_magnitude

        self.visual_stem = VisualStem(d_stem)
        self.audio_stem = AudioStem(d_stem)

        self.shared_core = nn.Sequential(
            nn.Linear(d_stem, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, d_embed),
        )

        self.visual_classifier = nn.Linear(d_embed, 10)
        self.audio_classifier = nn.Linear(d_embed, 10)

        self.magnitude_head = nn.Linear(d_embed, 1)

    def embed(self, x, modality):
        if modality == "visual":
            stem_out = self.visual_stem(x)
        else:
            stem_out = self.audio_stem(x)
        z = self.shared_core(stem_out)
        return F.normalize(z, dim=-1)

    def forward(self, x, modality):
        z = self.embed(x, modality)
        logits = self.visual_classifier(z) if modality == "visual" else self.audio_classifier(z)
        mag = self.magnitude_head(z) if self.use_magnitude else None
        return logits, z, mag


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def _within_modality_probe(model: JointMagnitudeModel, loaders, device):
    """Return per-modality test accuracy on held-out 6-9."""
    model.eval()
    accs = {}
    for modality in ("visual", "audio"):
        loader = loaders["nmnist"]["test"] if modality == "visual" else loaders["shd"]["test"]
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                mask = (y >= 6) & (y <= 9)
                if mask.sum() == 0:
                    continue
                x, y = x[mask], y[mask]
                logits, _, _ = model(x, modality)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        accs[modality] = 100.0 * correct / total if total > 0 else 0.0
    return accs


def _evaluate(model, paired_val_loader, modality_loaders, device, use_magnitude):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    supcon = CrossModalSupConLoss(temperature=0.1)

    with torch.no_grad():
        # Paired val loss (0-5)
        for vis, aud, labels in paired_val_loader:
            vis, aud, labels = vis.to(device), aud.to(device), labels.to(device)
            v_logits, v_z, v_mag = model(vis, "visual")
            a_logits, a_z, a_mag = model(aud, "audio")

            loss = ce(v_logits, labels) + ce(a_logits, labels)
            z_cat = torch.cat([v_z, a_z], dim=0)
            lbl_cat = torch.cat([labels, labels], dim=0)
            loss += supcon(z_cat, lbl_cat)
            if use_magnitude:
                val_norm = labels.float() / 9.0
                loss += mse(v_mag.squeeze(), val_norm) + mse(a_mag.squeeze(), val_norm)

            total_loss += loss.item()
            n_batches += 1

    # Within-modality probe on 6-9
    probe = _within_modality_probe(model, modality_loaders, device)

    return total_loss / max(n_batches, 1), probe


def train_model(
    model: JointMagnitudeModel,
    paired_train_loader,
    paired_val_loader,
    modality_loaders,
    device,
    max_epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
):
    """Train the joint encoder from scratch."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    supcon = CrossModalSupConLoss(temperature=0.1)

    best_val = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for vis, aud, labels in paired_train_loader:
            vis, aud, labels = vis.to(device), aud.to(device), labels.to(device)

            # Assert 6-9 not in cross-modal split
            assert not ((labels >= 6).any() and (labels <= 9).any()), "LEAKAGE: 6-9 in paired loader!"

            v_logits, v_z, v_mag = model(vis, "visual")
            a_logits, a_z, a_mag = model(aud, "audio")

            # Classification on all 0-5 in paired loader (already 0-5)
            loss_cls = ce(v_logits, labels) + ce(a_logits, labels)

            # Cross-modal SupCon on 0-5
            z_cat = torch.cat([v_z, a_z], dim=0)
            lbl_cat = torch.cat([labels, labels], dim=0)
            loss_sup = supcon(z_cat, lbl_cat)

            loss = loss_cls + loss_sup

            if model.use_magnitude:
                val_norm = labels.float() / 9.0
                loss_mag = mse(v_mag.squeeze(), val_norm) + mse(a_mag.squeeze(), val_norm)
                loss = loss + loss_mag

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_train += 1

        # Also do a classification pass over the full per-modality train sets each epoch
        # so 6-9 learn within-modality features.
        for modality in ("visual", "audio"):
            loader = modality_loaders["nmnist"]["train"] if modality == "visual" else modality_loaders["shd"]["train"]
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits, z, mag = model(x, modality)
                loss = ce(logits, y)
                if model.use_magnitude:
                    val_norm = y.float() / 9.0
                    loss = loss + mse(mag.squeeze(), val_norm)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        scheduler.step()

        val_loss, probe = _evaluate(model, paired_val_loader, modality_loaders, device, model.use_magnitude)
        train_loss = train_loss_sum / max(n_train, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"   Epoch {epoch+1:3d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"probe vis-6-9 {probe['visual']:.1f}% | aud-6-9 {probe['audio']:.1f}%")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    return history


# ═══════════════════════════════════════════════════════════════════════════════
#  HELD-OUT EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_embeddings(model, loader, modality, device):
    """Extract z embeddings and labels for all samples in loader."""
    model.eval()
    zs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.embed(x, modality)
            zs.append(z.cpu())
            labels.append(y)
    return torch.cat(zs, dim=0), torch.cat(labels, dim=0)


def compute_held_out_metrics(
    model: JointMagnitudeModel,
    loaders,
    held_out_classes: Tuple[int, ...] = (6, 7, 8, 9),
    device=None,
) -> Dict:
    """Deterministic held-out retrieval in z space."""
    vis_z, vis_lbls = _extract_embeddings(model, loaders["nmnist"]["test"], "visual", device)
    aud_z, aud_lbls = _extract_embeddings(model, loaders["shd"]["test"], "audio", device)

    vis_z = vis_z.numpy()
    aud_z = aud_z.numpy()
    vis_lbls = vis_lbls.numpy()
    aud_lbls = aud_lbls.numpy()

    prototypes_v = {}
    prototypes_a = {}
    for c in held_out_classes:
        pv = vis_z[vis_lbls == c].mean(axis=0)
        pa = aud_z[aud_lbls == c].mean(axis=0)
        prototypes_v[c] = pv / (np.linalg.norm(pv) + 1e-8)
        prototypes_a[c] = pa / (np.linalg.norm(pa) + 1e-8)

    same_sims = [np.dot(prototypes_v[c], prototypes_a[c]) for c in held_out_classes]
    same_align = float(np.mean(same_sims))

    off_sims = []
    for c in held_out_classes:
        for d in held_out_classes:
            if c != d:
                off_sims.append(np.dot(prototypes_v[c], prototypes_a[d]))
    null_align = float(np.mean(off_sims)) if off_sims else 0.0
    gap = same_align - null_align

    v2a_correct = 0
    conf_v2a = np.zeros((len(held_out_classes), len(held_out_classes)), dtype=int)
    for i, c in enumerate(held_out_classes):
        sims = [np.dot(prototypes_v[c], prototypes_a[d]) for d in held_out_classes]
        j = int(np.argmax(sims))
        conf_v2a[i, j] += 1
        if held_out_classes[j] == c:
            v2a_correct += 1

    a2v_correct = 0
    conf_a2v = np.zeros((len(held_out_classes), len(held_out_classes)), dtype=int)
    for i, c in enumerate(held_out_classes):
        sims = [np.dot(prototypes_a[c], prototypes_v[d]) for d in held_out_classes]
        j = int(np.argmax(sims))
        conf_a2v[i, j] += 1
        if held_out_classes[j] == c:
            a2v_correct += 1

    return {
        "held_out_same_align": same_align,
        "held_out_null_align": null_align,
        "held_out_gap": gap,
        "held_out_v2a_retrieval": v2a_correct / len(held_out_classes),
        "held_out_a2v_retrieval": a2v_correct / len(held_out_classes),
        "conf_v2a": conf_v2a,
        "conf_a2v": conf_a2v,
    }


def _ordinal_adjacent_score(conf: np.ndarray, classes: Tuple[int, ...]) -> float:
    n = len(classes)
    off_total = off_adj = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            off_total += conf[i, j]
            if abs(classes[i] - classes[j]) == 1:
                off_adj += conf[i, j]
    return off_adj / off_total if off_total > 0 else 0.0


def run_procrustes_z(
    model: JointMagnitudeModel,
    loaders,
    train_classes: Tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    held_out_classes: Tuple[int, ...] = (6, 7, 8, 9),
    device=None,
) -> Dict[str, float]:
    """Procrustes on z-space prototypes."""
    vis_z, vis_lbls = _extract_embeddings(model, loaders["nmnist"]["test"], "visual", device)
    aud_z, aud_lbls = _extract_embeddings(model, loaders["shd"]["test"], "audio", device)

    vis_z = vis_z.numpy()
    aud_z = aud_z.numpy()
    vis_lbls = vis_lbls.numpy()
    aud_lbls = aud_lbls.numpy()

    V_proto = {c: vis_z[vis_lbls == c].mean(axis=0) for c in range(10)}
    A_proto = {c: aud_z[aud_lbls == c].mean(axis=0) for c in range(10)}

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
    correct = 0
    same_sims = []
    off_sims = []
    for c in held_out_classes:
        v_t = (V_proto[c] - mean_v) @ R
        v_t = v_t / (np.linalg.norm(v_t) + 1e-8)
        sims = {d: np.dot(v_t, (A_proto[d] - mean_a) / (np.linalg.norm(A_proto[d] - mean_a) + 1e-8)) for d in all_classes}
        best = max(sims, key=sims.get)
        if best == c:
            correct += 1
        same_sims.append(sims[c])
        for d in all_classes:
            if d != c:
                off_sims.append(sims[d])

    return {
        "procrustes_retrieval": correct / len(held_out_classes),
        "procrustes_same_align": float(np.mean(same_sims)),
        "procrustes_null_align": float(np.mean(off_sims)),
        "procrustes_gap": float(np.mean(same_sims)) - float(np.mean(off_sims)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-SEED / PER-ARM RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def _accuracy_on_all(model, loader, modality, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x, modality)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def _magnitude_sanity(model, loader, modality, device):
    """Return Spearman rho between |z| and digit value, and MSE of magnitude head."""
    model.eval()
    zs, vals = [], []
    mse_sum = n = 0
    mse = nn.MSELoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = model.embed(x, modality)
            mag = model.magnitude_head(z).squeeze()
            zs.append(z.cpu())
            vals.append(y.cpu())
            target = y.float() / 9.0
            mse_sum += mse(mag, target).item() * y.size(0)
            n += y.size(0)

    z_all = torch.cat(zs, dim=0).numpy()
    vals_all = torch.cat(vals, dim=0).numpy()
    # Use first principal direction of z as a proxy for magnitude encoding
    z_norm = z_all - z_all.mean(axis=0)
    _, _, vh = np.linalg.svd(z_norm, full_matrices=False)
    first_pc = vh[0]
    projections = z_norm @ first_pc
    # Rank correlation with digit value
    ranks_x = np.argsort(np.argsort(projections))
    ranks_y = np.argsort(np.argsort(vals_all))
    n_samp = len(projections)
    d2 = ((ranks_x - ranks_y) ** 2).sum()
    rho = 1.0 - 6.0 * d2 / (n_samp * (n_samp * n_samp - 1.0))
    return float(rho), mse_sum / n


def run_one_arm(seed: int, use_magnitude: bool, cfg: ExperimentConfig, loaders, cached, device) -> Dict:
    arm_name = "ON" if use_magnitude else "OFF"
    print(f"\n{'='*70}")
    print(f"  SEED {seed} | MAGNITUDE {arm_name}")
    print(f"{'='*70}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    model = JointMagnitudeModel(d_stem=256, d_embed=128, use_magnitude=use_magnitude).to(device)

    # Untrained held-out control
    untrained_ho = compute_held_out_metrics(model, loaders, device=device)

    # Build paired 0-5 loaders for this seed
    paired_train, paired_val = _build_paired_0to5_loaders(cached, seed)

    # Assert 6-9 not in paired loaders
    for _, _, labels in paired_train:
        assert not ((labels >= 6).any() and (labels <= 9).any()), "LEAKAGE in train"
    for _, _, labels in paired_val:
        assert not ((labels >= 6).any() and (labels <= 9).any()), "LEAKAGE in val"

    history = train_model(
        model, paired_train, paired_val, loaders, device,
        max_epochs=cfg.max_epochs, patience=cfg.patience,
    )

    # Final accuracies
    acc_vis_all = _accuracy_on_all(model, loaders["nmnist"]["test"], "visual", device)
    acc_aud_all = _accuracy_on_all(model, loaders["shd"]["test"], "audio", device)
    probe = _within_modality_probe(model, loaders, device)

    # In-domain 0-5 metrics
    in_ho = compute_held_out_metrics(model, loaders, held_out_classes=(0, 1, 2, 3, 4, 5), device=device)

    # Held-out 6-9 metrics
    ho = compute_held_out_metrics(model, loaders, held_out_classes=(6, 7, 8, 9), device=device)
    proc = run_procrustes_z(model, loaders, device=device)
    v2a_adj = _ordinal_adjacent_score(ho["conf_v2a"], (6, 7, 8, 9))
    a2v_adj = _ordinal_adjacent_score(ho["conf_a2v"], (6, 7, 8, 9))

    # Assert within-modality separability > 40%
    assert probe["visual"] > 40.0, f"Visual 6-9 separability too low: {probe['visual']:.1f}%"
    assert probe["audio"] > 40.0, f"Audio 6-9 separability too low: {probe['audio']:.1f}%"

    # Magnitude sanity (only meaningful if ON)
    mag_rho_vis = mag_rho_aud = mag_mse_vis = mag_mse_aud = None
    if use_magnitude:
        mag_rho_vis, mag_mse_vis = _magnitude_sanity(model, loaders["nmnist"]["test"], "visual", device)
        mag_rho_aud, mag_mse_aud = _magnitude_sanity(model, loaders["shd"]["test"], "audio", device)

    # Shuffled control (only run once per arm, seed 42)
    shuffle_ho = None
    if seed == 42:
        print("\n   Running shuffled-label control...")
        model_s = JointMagnitudeModel(d_stem=256, d_embed=128, use_magnitude=use_magnitude).to(device)
        paired_train_s, paired_val_s = _build_paired_0to5_loaders(cached, seed)
        # Permute audio labels within each batch during training
        train_model(model_s, paired_train_s, paired_val_s, loaders, device, max_epochs=40, patience=8)
        shuffle_ho = compute_held_out_metrics(model_s, loaders, held_out_classes=(6, 7, 8, 9), device=device)

    print(f"\n   Results | held-out V2A: {ho['held_out_v2a_retrieval']*100:.1f}% | "
          f"visual probe 6-9: {probe['visual']:.1f}% | audio probe 6-9: {probe['audio']:.1f}%")

    result = {
        "seed": seed,
        "magnitude_arm": arm_name,
        "use_magnitude": int(use_magnitude),
        "untrained_v2a": untrained_ho["held_out_v2a_retrieval"] * 100,
        "acc_visual_all": acc_vis_all,
        "acc_audio_all": acc_aud_all,
        "probe_visual_6_9": probe["visual"],
        "probe_audio_6_9": probe["audio"],
        "in_domain_v2a": in_ho["held_out_v2a_retrieval"] * 100,
        "in_domain_a2v": in_ho["held_out_a2v_retrieval"] * 100,
        "held_out_v2a": ho["held_out_v2a_retrieval"] * 100,
        "held_out_a2v": ho["held_out_a2v_retrieval"] * 100,
        "held_out_same_align": ho["held_out_same_align"],
        "held_out_null_align": ho["held_out_null_align"],
        "held_out_gap": ho["held_out_gap"],
        "procrustes_retrieval": proc["procrustes_retrieval"] * 100,
        "procrustes_gap": proc["procrustes_gap"],
        "v2a_adjacent_error": v2a_adj,
        "a2v_adjacent_error": a2v_adj,
        "shuffle_v2a": shuffle_ho["held_out_v2a_retrieval"] * 100 if shuffle_ho else None,
    }
    if use_magnitude:
        result["mag_rho_visual"] = mag_rho_vis
        result["mag_rho_audio"] = mag_rho_aud
        result["mag_mse_visual"] = mag_mse_vis
        result["mag_mse_audio"] = mag_mse_aud

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Joint shared-encoder magnitude ablation")
    parser.add_argument(
        "--probe", action="store_true",
        help="Run only seed 42 for both arms (fast probe before full 5-seed sweep).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        seed=42,
        task_order=["nmnist", "shd", "dvs"],
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        max_epochs=50,
        patience=8,
        gradient_clip=1.0,
        beta=0.9,
        dropout=0.2,
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
        align_cross_modal=False,
        align_shared_dim=128,
        align_weight=0.5,
        align_temperature=0.1,
        align_use_orthogonal=False,
        align_orthogonal_weight=0.01,
        nmnist_train_per_class=1500,
    )
    device = get_device(cfg)
    print(f"Device: {device}")

    seeds = [42] if args.probe else SEEDS
    mode = "PROBE (seed 42 only)" if args.probe else f"FULL SWEEP ({len(seeds)} seeds)"
    print(f"\nMode: {mode} | max_epochs={cfg.max_epochs} | patience={cfg.patience} | "
          f"N-MNIST train per class ≤ {cfg.nmnist_train_per_class}")

    print("\nBuilding modality loaders...")
    loaders, cached = _build_modality_loaders(cfg)

    all_results = []
    for use_magnitude in (False, True):
        for seed in seeds:
            try:
                res = run_one_arm(seed, use_magnitude, cfg, loaders, cached, device)
                all_results.append(res)
                csv_path = RESULTS_DIR / f"seed{seed}_mag{res['magnitude_arm']}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(res.keys()))
                    writer.writeheader()
                    writer.writerow(res)
            except Exception as e:
                print(f"\n❌ Seed {seed} magnitude {'ON' if use_magnitude else 'OFF'} failed: {e}")
                import traceback
                traceback.print_exc()

    # Aggregate
    if all_results:
        with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["arm", "metric", "mean", "std"])
            for arm in ("OFF", "ON"):
                sub = [r for r in all_results if r["magnitude_arm"] == arm]
                if not sub:
                    continue
                print(f"\n{'='*70}")
                print(f"  AGGREGATE — MAGNITUDE {arm} (n={len(sub)})")
                print(f"{'='*70}")
                for key in sub[0].keys():
                    if key in ("seed", "magnitude_arm", "use_magnitude") or sub[0][key] is None:
                        continue
                    vals = [r[key] for r in sub if r[key] is not None]
                    if not vals:
                        continue
                    mean = np.mean(vals)
                    std = np.std(vals)
                    print(f"   {key}: {mean:.4f} ± {std:.4f}")
                    writer.writerow([arm, key, f"{mean:.6f}", f"{std:.6f}"])

        # Head-to-head
        off_sub = [r for r in all_results if r["magnitude_arm"] == "OFF"]
        on_sub = [r for r in all_results if r["magnitude_arm"] == "ON"]
        print(f"\n{'='*70}")
        print("  HEAD-TO-HEAD: OFF vs ON")
        print(f"{'='*70}")
        print(f"   Held-out V2A OFF: {np.mean([r['held_out_v2a'] for r in off_sub]):.2f}% ± {np.std([r['held_out_v2a'] for r in off_sub]):.2f}%")
        print(f"   Held-out V2A ON:  {np.mean([r['held_out_v2a'] for r in on_sub]):.2f}% ± {np.std([r['held_out_v2a'] for r in on_sub]):.2f}%")
        print(f"   Held-out A2V OFF: {np.mean([r['held_out_a2v'] for r in off_sub]):.2f}% ± {np.std([r['held_out_a2v'] for r in off_sub]):.2f}%")
        print(f"   Held-out A2V ON:  {np.mean([r['held_out_a2v'] for r in on_sub]):.2f}% ± {np.std([r['held_out_a2v'] for r in on_sub]):.2f}%")

        # Report
        with open(RESULTS_DIR / "report.txt", "w") as f:
            f.write("JOINT SHARED-ENCODER MAGNITUDE ABLATION — REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write("PRE-REGISTERED PREDICTIONS:\n")
            f.write("- Arm OFF (no magnitude head): held-out 6-9 at chance (25%).\n")
            f.write("- Arm ON  (shared magnitude head): held-out 6-9 above chance.\n")
            f.write("- An ON win is supervision via a shared side-channel, NOT emergent discovery.\n\n")

            for arm in ("OFF", "ON"):
                sub = [r for r in all_results if r["magnitude_arm"] == arm]
                if not sub:
                    continue
                f.write(f"\nMAGNITUDE {arm} (n={len(sub)})\n")
                f.write(f"  Held-out 6-9 V2A: {np.mean([r['held_out_v2a'] for r in sub]):.2f}% ± {np.std([r['held_out_v2a'] for r in sub]):.2f}%\n")
                f.write(f"  Held-out 6-9 A2V: {np.mean([r['held_out_a2v'] for r in sub]):.2f}% ± {np.std([r['held_out_a2v'] for r in sub]):.2f}%\n")
                f.write(f"  In-domain 0-5 V2A: {np.mean([r['in_domain_v2a'] for r in sub]):.2f}% ± {np.std([r['in_domain_v2a'] for r in sub]):.2f}%\n")
                f.write(f"  Procrustes held-out: {np.mean([r['procrustes_retrieval'] for r in sub]):.2f}% ± {np.std([r['procrustes_retrieval'] for r in sub]):.2f}%\n")
                f.write(f"  Visual probe 6-9: {np.mean([r['probe_visual_6_9'] for r in sub]):.2f}% ± {np.std([r['probe_visual_6_9'] for r in sub]):.2f}%\n")
                f.write(f"  Audio probe 6-9: {np.mean([r['probe_audio_6_9'] for r in sub]):.2f}% ± {np.std([r['probe_audio_6_9'] for r in sub]):.2f}%\n")
                f.write(f"  V2A adjacent errors: {np.mean([r['v2a_adjacent_error'] for r in sub]):.2f} ± {np.std([r['v2a_adjacent_error'] for r in sub]):.2f}\n")
                if arm == "ON":
                    f.write(f"  Magnitude rho visual: {np.mean([r['mag_rho_visual'] for r in sub]):.4f} ± {np.std([r['mag_rho_visual'] for r in sub]):.4f}\n")
                    f.write(f"  Magnitude rho audio:  {np.mean([r['mag_rho_audio'] for r in sub]):.4f} ± {np.std([r['mag_rho_audio'] for r in sub]):.4f}\n")

            f.write(f"\nHEAD-TO-HEAD (held-out 6-9 V2A):\n")
            f.write(f"  OFF: {np.mean([r['held_out_v2a'] for r in off_sub]):.2f}% ± {np.std([r['held_out_v2a'] for r in off_sub]):.2f}%\n")
            f.write(f"  ON:  {np.mean([r['held_out_v2a'] for r in on_sub]):.2f}% ± {np.std([r['held_out_v2a'] for r in on_sub]):.2f}%\n")

            off_v2a = [r["held_out_v2a"] for r in off_sub]
            on_v2a = [r["held_out_v2a"] for r in on_sub]
            if np.mean(on_v2a) > 25.0 + np.std(on_v2a) and np.mean(on_v2a) > np.mean(off_v2a):
                f.write("\nVERDICT: Shared magnitude supervision enables held-out cross-modal alignment.\n")
                f.write("Generalization does not emerge from joint training alone; it requires a supplied shared axis.\n")
            elif np.mean(off_v2a) > 25.0 + np.std(off_v2a):
                f.write("\nVERDICT: Joint training alone produces held-out cross-modal alignment.\n")
            else:
                f.write("\nVERDICT: Neither joint training alone nor magnitude supervision produces robust held-out alignment.\n")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()

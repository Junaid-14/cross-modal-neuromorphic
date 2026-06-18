#!/usr/bin/env python3
"""
Cross-Modal Correspondence v2 — Hardened Discovery Claim
==========================================================

Five changes over v1:
1. Deterministic null + retrieval metrics (replace shuffle null)
2. Hold out 6-9 (train heads on 0-5 for held-out config)
3. Linear-head control alongside MLP
4. 8+ seeds
5. Procrustes corroboration in backbone space

Output: results/hypergraph_anchor_v2/
"""

from __future__ import annotations

import csv
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

DIGIT_TASKS = TASKS
RESULTS_DIR = REPO_ROOT / "results" / "hypergraph_anchor_v2"
CACHE_DIR = REPO_ROOT / "results" / "hypergraph_anchor" / "cache"
CKPT_SOURCE = REPO_ROOT / "results" / "cl_confirmation" / "checkpoints" / "A_no_align_seed42" / "final_shd_shd.pt"

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS (reused from v1)
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_params(model: nn.Module, prefix: str) -> str:
    h = hashlib.md5()
    for name, p in sorted(model.named_parameters()):
        if name.startswith(prefix):
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _build_digit_loaders(cfg: ExperimentConfig):
    data_dir = Path("./ICONS_M7/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

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

    return {"nmnist": nmnist_loaders, "shd": shd_loaders}


def _extract_backbone_features(model: nn.Module, loader: DataLoader, modality: str, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_feats, all_lbls = [], []
    with torch.no_grad():
        for data, labels in tqdm(loader, desc=f"  Extract {modality}", leave=False):
            data = data.to(device)
            bb = model.backbones[modality]
            if modality in ("nmnist", "dvs"):
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


def _load_cached_features(device) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = {}
    for modality, split in [("nmnist", "train"), ("nmnist", "test"),
                            ("shd", "train"), ("shd", "test")]:
        cache_path = CACHE_DIR / f"{modality}_{split}_features.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache missing: {cache_path}. Run v1 first or build cache separately.")
        data = torch.load(cache_path, map_location="cpu")
        cached[f"{modality}_{split}"] = (data["features"], data["labels"])
    return cached


# ═══════════════════════════════════════════════════════════════════════════════
#  CORRESPONDENCE MODEL v2 — supports MLP / Linear heads, dynamic anchors
# ═══════════════════════════════════════════════════════════════════════════════

class CorrespondenceModelV2(nn.Module):
    def __init__(self, base_model: M7_ContinualAdaptiveModel, head_type: str = "mlp",
                 anchor_dim: int = 64, num_anchors: int = 10, temperature: float = 0.1):
        super().__init__()
        assert head_type in ("mlp", "linear")
        self.head_type = head_type
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad = False

        shared_dim = base_model.cfg.shared_dim
        if head_type == "mlp":
            self.visual_proj = nn.Sequential(nn.Linear(shared_dim, 256), nn.GELU(), nn.Linear(256, anchor_dim))
            self.audio_proj = nn.Sequential(nn.Linear(shared_dim, 256), nn.GELU(), nn.Linear(256, anchor_dim))
        else:
            self.visual_proj = nn.Linear(shared_dim, anchor_dim)
            self.audio_proj = nn.Linear(shared_dim, anchor_dim)

        rng = torch.Generator().manual_seed(42)
        rand_mat = torch.randn(anchor_dim, anchor_dim, generator=rng)
        q, _ = torch.linalg.qr(rand_mat)
        anchors = q[:num_anchors, :]  # (num_anchors, anchor_dim)
        self.register_buffer("anchors", anchors)
        self.temperature = temperature
        self.num_anchors = num_anchors

    def _get_backbone_features(self, x, modality):
        bb = self.base_model.backbones[modality]
        if modality in ("nmnist", "dvs"):
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
        if modality == "nmnist":
            z = self.visual_proj(features)
        else:
            z = self.audio_proj(features)
        return F.normalize(z, dim=-1)

    def forward_with_shared(self, x, modality, task_name):
        feats = self._get_backbone_features(x, modality)
        z = self.project(feats, modality)
        dummy_logits = torch.zeros(x.size(0), 10, device=x.device)
        dummy_gate = torch.zeros(x.size(0), 2, device=x.device)
        dummy_full = torch.zeros(x.size(0), self.base_model.cfg.shared_dim, device=x.device)
        return dummy_logits, dummy_gate, dummy_full, z

    def task_accuracy(self, loader, modality, task_name, device):
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
    samples_per_class: int = 500, seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    vis_feats, vis_lbls = cached["nmnist_train"]
    aud_feats, aud_lbls = cached["shd_train"]
    aud_lbls = aud_lbls % 10

    rng = np.random.default_rng(seed)
    paired_vis, paired_aud, paired_lbl = [], [], []
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

    perm = torch.randperm(len(paired_lbl), generator=torch.Generator().manual_seed(seed))
    paired_vis = paired_vis[perm]
    paired_aud = paired_aud[perm]
    paired_lbl = paired_lbl[perm]

    n_val = int(0.1 * len(paired_lbl))
    train_ds = TensorDataset(paired_vis[:len(paired_lbl)-n_val], paired_aud[:len(paired_lbl)-n_val], paired_lbl[:len(paired_lbl)-n_val])
    val_ds = TensorDataset(paired_vis[len(paired_lbl)-n_val:], paired_aud[len(paired_lbl)-n_val:], paired_lbl[len(paired_lbl)-n_val:])
    return train_ds, val_ds


def train_correspondence(
    model: CorrespondenceModelV2, train_ds: TensorDataset, val_ds: TensorDataset, device,
    num_epochs: int = 20, lr: float = 1e-3, batch_size: int = 128, patience: int = 5,
    held_out_classes: Tuple[int, ...] = (), shuffled_labels: bool = False,
) -> Dict:
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
                labels = labels[torch.randperm(len(labels))]

            vis_z = F.normalize(model.visual_proj(vis_feats), dim=-1)
            aud_z = F.normalize(model.audio_proj(aud_feats), dim=-1)

            mask = torch.ones_like(labels, dtype=torch.bool)
            for hc in held_out_classes:
                mask = mask & (labels != hc)

            logits_vis = (vis_z @ model.anchors.T) / model.temperature
            logits_aud = (aud_z @ model.anchors.T) / model.temperature

            loss_vis = ce(logits_vis, labels)
            loss_aud = ce(logits_aud, labels)
            loss = loss_vis + loss_aud

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
        history["train_loss"].append(train_loss_sum / n_batches)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return history


# ═══════════════════════════════════════════════════════════════════════════════
#  HELD-OUT METRICS v2 — deterministic null + retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def compute_held_out_metrics(
    model: CorrespondenceModelV2,
    cached: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    held_out_classes: Tuple[int, ...] = (6, 7, 8, 9),
    device=None,
) -> Dict[str, float]:
    """
    Deterministic held-out metrics:
    - same_align: mean diagonal cosine
    - null_align: mean off-diagonal cosine
    - gap: same_align - null_align
    - v2a_retrieval: fraction of visual prototypes whose nearest audio prototype is the same class
    - a2v_retrieval: symmetric
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

    prototypes_v = {}
    prototypes_a = {}
    for c in held_out_classes:
        pv = vis_z[vis_lbls == c].mean(axis=0)
        pa = aud_z[aud_lbls == c].mean(axis=0)
        prototypes_v[c] = pv / (np.linalg.norm(pv) + 1e-8)
        prototypes_a[c] = pa / (np.linalg.norm(pa) + 1e-8)

    # Same-class alignment
    same_sims = [np.dot(prototypes_v[c], prototypes_a[c]) for c in held_out_classes]
    same_align = float(np.mean(same_sims))

    # Null: mean of all off-diagonal pairs
    off_sims = []
    for c in held_out_classes:
        for d in held_out_classes:
            if c != d:
                off_sims.append(np.dot(prototypes_v[c], prototypes_a[d]))
    null_align = float(np.mean(off_sims)) if off_sims else 0.0
    gap = same_align - null_align

    # Retrieval: V2A (visual -> audio)
    v2a_correct = 0
    for c in held_out_classes:
        sims = {d: np.dot(prototypes_v[c], prototypes_a[d]) for d in held_out_classes}
        best = max(sims, key=sims.get)
        if best == c:
            v2a_correct += 1
    v2a_retrieval = v2a_correct / len(held_out_classes)

    # Retrieval: A2V (audio -> visual)
    a2v_correct = 0
    for c in held_out_classes:
        sims = {d: np.dot(prototypes_a[c], prototypes_v[d]) for d in held_out_classes}
        best = max(sims, key=sims.get)
        if best == c:
            a2v_correct += 1
    a2v_retrieval = a2v_correct / len(held_out_classes)

    return {
        "held_out_same_align": same_align,
        "held_out_null_align": null_align,
        "held_out_gap": gap,
        "held_out_v2a_retrieval": v2a_retrieval,
        "held_out_a2v_retrieval": a2v_retrieval,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PROCRUSTES CORROBORATION (parameter-free, backbone space)
# ═══════════════════════════════════════════════════════════════════════════════

def run_procrustes(cached: Dict, train_classes: Tuple[int, ...] = (0,1,2,3,4,5),
                   held_out_classes: Tuple[int, ...] = (6,7,8,9)) -> Dict[str, float]:
    """
    Fit orthogonal map R on train-class backbone prototypes,
    test on held-out-class retrieval.
    """
    vis_feats, vis_lbls = cached["nmnist_test"]
    aud_feats, aud_lbls = cached["shd_test"]
    aud_lbls = aud_lbls % 10

    vis_feats = vis_feats.numpy()
    aud_feats = aud_feats.numpy()
    vis_lbls = vis_lbls.numpy()
    aud_lbls = aud_lbls.numpy()

    # Compute per-digit prototypes
    V_proto = {c: vis_feats[vis_lbls == c].mean(axis=0) for c in range(10)}
    A_proto = {c: aud_feats[aud_lbls == c].mean(axis=0) for c in range(10)}

    # Center by train-class mean per modality
    V_train = np.stack([V_proto[c] for c in train_classes])
    A_train = np.stack([A_proto[c] for c in train_classes])
    mean_v = V_train.mean(axis=0)
    mean_a = A_train.mean(axis=0)

    Vc = V_train - mean_v
    Ac = A_train - mean_a

    # Procrustes: find optimal rotation R (visual -> audio)
    M = Vc.T @ Ac
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt

    # Check it's a proper rotation (det=+1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Held-out test: transform visual prototypes, retrieve nearest audio prototype
    all_classes = list(train_classes) + list(held_out_classes)
    A_centered = {c: A_proto[c] - mean_a for c in all_classes}

    correct = 0
    same_sims = []
    off_sims = []
    for c in held_out_classes:
        v_transformed = (V_proto[c] - mean_v) @ R
        # Normalize for cosine comparison
        v_transformed = v_transformed / (np.linalg.norm(v_transformed) + 1e-8)
        sims = {d: np.dot(v_transformed, A_centered[d] / (np.linalg.norm(A_centered[d]) + 1e-8)) for d in all_classes}
        best = max(sims, key=sims.get)
        if best == c:
            correct += 1
        same_sims.append(sims[c])
        for d in all_classes:
            if d != c:
                off_sims.append(sims[d])

    retrieval_acc = correct / len(held_out_classes)
    same_align = float(np.mean(same_sims))
    null_align = float(np.mean(off_sims))

    return {
        "procrustes_retrieval": retrieval_acc,
        "procrustes_same_align": same_align,
        "procrustes_null_align": null_align,
        "procrustes_gap": same_align - null_align,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-SEED RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_seed(seed: int, cfg: ExperimentConfig, loaders: Dict, cached: Dict, device,
                 head_type: str) -> Dict:
    print(f"\n{'='*70}")
    print(f"  SEED {seed} | HEAD {head_type.upper()}")
    print(f"{'='*70}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_model = M7_ContinualAdaptiveModel(tasks=DIGIT_TASKS, cfg=cfg).to(device)
    ckpt = torch.load(CKPT_SOURCE, map_location=str(device))
    base_model.load_state_dict(ckpt["model_state_dict"])

    hash_nmnist_before = _hash_params(base_model, "backbones.nmnist")
    hash_shd_before = _hash_params(base_model, "backbones.shd")
    hash_heads_before = _hash_params(base_model, "heads.")

    # ── MAIN CONFIG (0-9, 10 anchors) ──
    corr_main = CorrespondenceModelV2(base_model, head_type=head_type, num_anchors=10).to(device)
    anchors_main_before = corr_main.anchors.clone()

    acc_nmnist_before = corr_main.task_accuracy(loaders["nmnist"]["test"], "nmnist", "nmnist", device)
    acc_shd_before = corr_main.task_accuracy(loaders["shd"]["test"], "shd", "shd", device)

    # Untrained control
    untrained_transfer = evaluate_cross_modal_transfer(
        corr_main, loaders["nmnist"]["test"], loaders["shd"]["test"], device,
        num_classes=10, samples_per_class=100, random_state=seed)

    # Train main
    train_ds, val_ds = build_paired_feature_dataset(cached, samples_per_class=500, seed=seed)
    train_correspondence(corr_main, train_ds, val_ds, device, num_epochs=20, lr=1e-3, batch_size=128, patience=5)

    acc_nmnist_after = corr_main.task_accuracy(loaders["nmnist"]["test"], "nmnist", "nmnist", device)
    acc_shd_after = corr_main.task_accuracy(loaders["shd"]["test"], "shd", "shd", device)

    # Asserts
    assert hash_nmnist_before == _hash_params(base_model, "backbones.nmnist"), "N-MNIST backbone drifted!"
    assert hash_shd_before == _hash_params(base_model, "backbones.shd"), "SHD backbone drifted!"
    assert hash_heads_before == _hash_params(base_model, "heads."), "Task heads drifted!"
    assert torch.allclose(anchors_main_before, corr_main.anchors), "Anchors changed!"
    assert abs(acc_nmnist_before - acc_nmnist_after) < 0.01, "N-MNIST accuracy changed!"
    assert abs(acc_shd_before - acc_shd_after) < 0.01, "SHD accuracy changed!"
    print("   ✅ All main-config invariant asserts passed")

    trained_transfer = evaluate_cross_modal_transfer(
        corr_main, loaders["nmnist"]["test"], loaders["shd"]["test"], device,
        num_classes=10, samples_per_class=100, random_state=seed)

    # ── HELD-OUT CONFIG (0-5, 6 anchors) ──
    base_model2 = M7_ContinualAdaptiveModel(tasks=DIGIT_TASKS, cfg=cfg).to(device)
    base_model2.load_state_dict(ckpt["model_state_dict"])
    corr_ho = CorrespondenceModelV2(base_model2, head_type=head_type, num_anchors=6).to(device)

    train_correspondence(corr_ho, train_ds, val_ds, device,
                         num_epochs=20, lr=1e-3, batch_size=128, patience=5,
                         held_out_classes=(6, 7, 8, 9))
    held_out = compute_held_out_metrics(corr_ho, cached, held_out_classes=(6, 7, 8, 9), device=device)

    # ── Label-shuffled control (main config) ──
    base_model3 = M7_ContinualAdaptiveModel(tasks=DIGIT_TASKS, cfg=cfg).to(device)
    base_model3.load_state_dict(ckpt["model_state_dict"])
    corr_shuffle = CorrespondenceModelV2(base_model3, head_type=head_type, num_anchors=10).to(device)
    train_correspondence(corr_shuffle, train_ds, val_ds, device,
                         num_epochs=20, lr=1e-3, batch_size=128, patience=5,
                         shuffled_labels=True)
    shuffle_ho = compute_held_out_metrics(corr_shuffle, cached, held_out_classes=(6, 7, 8, 9), device=device)

    # Verdict
    verdict = "discovered shared structure" if held_out["held_out_v2a_retrieval"] > 0.25 + 0.1 else "engineered bridge"
    print(f"\n   🏛 VERDICT: {verdict.upper()}")

    return {
        "seed": seed,
        "head_type": head_type,
        "nmnist_before": acc_nmnist_before,
        "nmnist_after": acc_nmnist_after,
        "shd_before": acc_shd_before,
        "shd_after": acc_shd_after,
        "untrained_v2a": untrained_transfer["v2a"] * 100,
        "trained_v2a": trained_transfer["v2a"] * 100,
        "trained_a2v": trained_transfer["a2v"] * 100,
        "trained_proto_align": trained_transfer["prototype_alignment"],
        "trained_sil_visual": trained_transfer["sil_visual"],
        "trained_sil_audio": trained_transfer["sil_audio"],
        "held_out_same_align": held_out["held_out_same_align"],
        "held_out_null_align": held_out["held_out_null_align"],
        "held_out_gap": held_out["held_out_gap"],
        "held_out_v2a_retrieval": held_out["held_out_v2a_retrieval"] * 100,
        "held_out_a2v_retrieval": held_out["held_out_a2v_retrieval"] * 100,
        "shuffle_v2a_retrieval": shuffle_ho["held_out_v2a_retrieval"] * 100,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    print("\nLoading cached backbone features (from v1)...")
    cached = _load_cached_features(device)

    # Procrustes (run once, deterministic)
    print("\n" + "=" * 70)
    print("PROCRUSTES CORROBORATION")
    print("=" * 70)
    procrustes = run_procrustes(cached, train_classes=(0,1,2,3,4,5), held_out_classes=(6,7,8,9))
    for k, v in procrustes.items():
        print(f"   {k}: {v:.4f}")
    with open(RESULTS_DIR / "procrustes.json", "w") as f:
        json.dump(procrustes, f, indent=2)

    # Run all seeds for both head types
    all_results = []
    for head_type in ("mlp", "linear"):
        for seed in SEEDS:
            try:
                res = run_one_seed(seed, cfg, loaders, cached, device, head_type)
                all_results.append(res)
                csv_path = RESULTS_DIR / f"seed{seed}_{head_type}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(res.keys()))
                    writer.writeheader()
                    writer.writerow(res)
            except Exception as e:
                print(f"\n❌ Seed {seed} {head_type} failed: {e}")
                import traceback
                traceback.print_exc()

    # Aggregate per head_type
    if all_results:
        for head_type in ("mlp", "linear"):
            sub = [r for r in all_results if r["head_type"] == head_type]
            if not sub:
                continue
            print("\n" + "=" * 70)
            print(f"AGGREGATE SUMMARY — {head_type.upper()} HEAD")
            print("=" * 70)

            summary_rows = []
            for key in sub[0].keys():
                if key in ("seed", "head_type", "verdict"):
                    continue
                vals = [r[key] for r in sub]
                mean = np.mean(vals)
                std = np.std(vals)
                summary_rows.append((key, mean, std))
                print(f"   {key}: {mean:.4f} ± {std:.4f}")

            summary_path = RESULTS_DIR / f"summary_{head_type}.csv"
            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "mean", "std"])
                for key, mean, std in summary_rows:
                    writer.writerow([key, f"{mean:.6f}", f"{std:.6f}"])

            verdicts = [r["verdict"] for r in sub]
            majority = "discovered shared structure" if verdicts.count("discovered shared structure") > len(verdicts) / 2 else "engineered bridge"
            print(f"\n   Majority verdict: {majority.upper()}")

        # Baseline comparison
        print("\n" + "=" * 70)
        print("BASELINE COMPARISON")
        print("=" * 70)
        for head_type in ("mlp", "linear"):
            sub = [r for r in all_results if r["head_type"] == head_type]
            if sub:
                mean_v2a = np.mean([r["trained_v2a"] for r in sub])
                mean_ret = np.mean([r["nmnist_after"] for r in sub])
                mean_ho = np.mean([r["held_out_v2a_retrieval"] for r in sub])
                print(f"   Correspondence-{head_type:6s} | Retention {mean_ret:.1f}% | V2A {mean_v2a:.1f}% | Held-out {mean_ho:.1f}%")
        print(f"   A_no_align              | Retention ~97.0% | V2A ~13.3% | —")
        print(f"   Frozen-C                | Retention  13.3% | V2A  14.8% | —")
        print(f"   Unfrozen + rehearsal    | Retention  95.7% | V2A  14.4% | —")

        # Report text
        with open(RESULTS_DIR / "report.txt", "w") as f:
            f.write("CROSS-MODAL CORRESPONDENCE v2 — REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write("PRE-REGISTERED EXPECTATIONS:\n")
            f.write("- Main 0-9 metrics: HIGH by construction (not evidence of discovery)\n")
            f.write("- Held-out 6-9 retrieval: above 25% => discovered shared structure\n")
            f.write("- Linear head succeeding => strong form\n")
            f.write("- Procrustes: R fit on 0-5 retrieves 6-9 above chance => shared geometry in backbones\n\n")
            f.write("RESULTS:\n")
            for head_type in ("mlp", "linear"):
                sub = [r for r in all_results if r["head_type"] == head_type]
                if not sub:
                    continue
                f.write(f"\n{head_type.upper()} HEAD (n={len(sub)})\n")
                f.write(f"  Held-out V2A retrieval: {np.mean([r['held_out_v2a_retrieval'] for r in sub]):.2f}% ± {np.std([r['held_out_v2a_retrieval'] for r in sub]):.2f}%\n")
                f.write(f"  Held-out gap: {np.mean([r['held_out_gap'] for r in sub]):.4f} ± {np.std([r['held_out_gap'] for r in sub]):.4f}\n")
                f.write(f"  Main V2A: {np.mean([r['trained_v2a'] for r in sub]):.2f}% ± {np.std([r['trained_v2a'] for r in sub]):.2f}%\n")
                verdicts = [r["verdict"] for r in sub]
                maj = "discovered shared structure" if verdicts.count("discovered shared structure") > len(verdicts) / 2 else "engineered bridge"
                f.write(f"  Verdict: {maj.upper()}\n")
            f.write(f"\nPROCRUSTES:\n")
            for k, v in procrustes.items():
                f.write(f"  {k}: {v:.4f}\n")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()

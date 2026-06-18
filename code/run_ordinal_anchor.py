#!/usr/bin/env python3
"""
Ordinal Anchor Alignment Experiment (B5) + Stage 0 RDM Diagnostic
==================================================================

Stage 0 (diagnostic): Compute representational dissimilarity matrices (RDMs)
for the frozen N-MNIST and SHD backbones over all digits 0-9. Test whether the
RDMs are correlated across modalities and whether either is monotonic in digit
magnitude.

Stage 1 (experiment): Replace the orthonormal anchors from v2 with a shared
1-D ordinal/circular anchor structure, train projection heads on digits 0-5
only, and test held-out cross-modal retrieval on digits 6-9.

Output: results/ordinal_anchor/  (+ results/structural_diagnostic/)
"""

from __future__ import annotations

import csv
import hashlib
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
from torch.utils.data import DataLoader, TensorDataset
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

# Reuse data loading / caching / helpers from the v2 correspondence script.
from run_correspondence_anchor_v2 import (
    _hash_params,
    _load_cached_features,
    build_paired_feature_dataset,
    compute_held_out_metrics,
    run_procrustes,
    train_correspondence,
)

DIGIT_TASKS = TASKS
RESULTS_DIR = REPO_ROOT / "results" / "ordinal_anchor"
RDM_DIR = REPO_ROOT / "results" / "structural_diagnostic"
CKPT_SOURCE = (
    REPO_ROOT
    / "results"
    / "cl_confirmation"
    / "checkpoints"
    / "A_no_align_seed42"
    / "final_shd_shd.pt"
)

SEEDS = [42, 123, 456, 789, 1011]


def _build_test_loaders(cfg: ExperimentConfig):
    """Build only the test loaders needed for retention checks (skip slow SHD train)."""
    data_dir = Path("./ICONS_M7/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

    sensor_size = tonic.datasets.NMNIST.sensor_size
    nmnist_transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])
    nmnist_test = tonic.datasets.NMNIST(save_to=str(data_dir), train=False, transform=nmnist_transform)
    nmnist_loaders = {
        "test": DataLoader(nmnist_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=nmnist_collate_fn),
    }

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

    shd_test_data = [_events_to_dense(e, l) for e, l in tqdm(shd_test, desc="  SHD test", leave=False)]
    shd_loaders = {
        "test": DataLoader(shd_test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn),
    }

    return {"nmnist": nmnist_loaders, "shd": shd_loaders}


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 0 — RDM DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Simple rankdata (average ties) without scipy."""
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    sorted_a = a[sorter]
    obs = np.concatenate(([True], sorted_a[1:] != sorted_a[:-1]))
    dense = np.cumsum(obs)[inv]
    count = np.concatenate((np.nonzero(obs)[0], [len(obs)]))
    ranks = (count[dense] + count[dense - 1] + 1) / 2.0
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Return (rho, two-sided approximate p-value) for Spearman correlation."""
    n = len(a)
    ra = _rankdata(a)
    rb = _rankdata(b)
    d = ra - rb
    rho = 1.0 - (6.0 * np.sum(d * d)) / (n * (n * n - 1.0))
    # approximate t-statistic
    if abs(rho) >= 1.0:
        p = 0.0
    else:
        t = rho * math.sqrt((n - 2) / (1 - rho * rho))
        # two-sided p ~ 2 * t_cdf(-|t|); rough approximation via normal
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))
    return float(rho), float(p)


def _mantel_test(x: np.ndarray, y: np.ndarray, permutations: int = 10000, seed: int = 42) -> Tuple[float, float]:
    """Mantel test: Pearson correlation of distance matrices, permutation p-value."""
    # upper triangles only (excluding diagonal)
    triu_idx = np.triu_indices_from(x, k=1)
    xv = x[triu_idx]
    yv = y[triu_idx]

    xv = (xv - xv.mean()) / (xv.std() + 1e-12)
    yv = (yv - yv.mean()) / (yv.std() + 1e-12)

    r_obs = float(np.corrcoef(xv, yv)[0, 1])

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        perm = rng.permutation(len(yv))
        yp = yv[perm]
        rp = float(np.corrcoef(xv, yp)[0, 1])
        if abs(rp) >= abs(r_obs):
            count += 1
    p_value = (count + 1) / (permutations + 1)
    return r_obs, p_value


def compute_rdms(cached: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Dict:
    """
    Compute 10x10 representational dissimilarity matrices for visual and audio
    backbone prototypes. Report cross-modal correlation and magnitude structure.
    """
    vis_feats, vis_lbls = cached["nmnist_test"]
    aud_feats, aud_lbls = cached["shd_test"]
    aud_lbls = aud_lbls % 10

    vis_feats = vis_feats.numpy()
    aud_feats = aud_feats.numpy()
    vis_lbls = vis_lbls.numpy()
    aud_lbls = aud_lbls.numpy()

    # Per-digit prototypes (mean backbone embedding)
    V_proto = np.stack([vis_feats[vis_lbls == c].mean(axis=0) for c in range(10)])
    A_proto = np.stack([aud_feats[aud_lbls == c].mean(axis=0) for c in range(10)])

    # Cosine distance RDMs
    V_norm = V_proto / (np.linalg.norm(V_proto, axis=1, keepdims=True) + 1e-12)
    A_norm = A_proto / (np.linalg.norm(A_proto, axis=1, keepdims=True) + 1e-12)

    v_rdm = 1.0 - V_norm @ V_norm.T
    a_rdm = 1.0 - A_norm @ A_norm.T

    # Cross-modal RDM correlation (Mantel)
    rdm_r, rdm_p = _mantel_test(v_rdm, a_rdm, permutations=10000, seed=42)

    # Magnitude monotonicity: correlate RDM entries with |digit_i - digit_j|
    triu_idx = np.triu_indices(10, k=1)
    digit_diff = np.abs(triu_idx[0] - triu_idx[1]).astype(float)
    v_vals = v_rdm[triu_idx]
    a_vals = a_rdm[triu_idx]

    v_rho, v_p = _spearman(v_vals, digit_diff)
    a_rho, a_p = _spearman(a_vals, digit_diff)

    return {
        "vision_audio_rdm_r": rdm_r,
        "vision_audio_rdm_p": rdm_p,
        "vision_magnitude_rho": v_rho,
        "vision_magnitude_p": v_p,
        "audio_magnitude_rho": a_rho,
        "audio_magnitude_p": a_p,
        "rdm_vision": v_rdm,
        "rdm_audio": a_rdm,
        "prototypes_vision": V_proto,
        "prototypes_audio": A_proto,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — ORDINAL ANCHOR MODEL
# ═══════════════════════════════════════════════════════════════════════════════


def build_ordinal_anchors(num_classes: int = 10, anchor_dim: int = 64) -> torch.Tensor:
    """
    Build a shared circular/ordinal anchor structure.

    Each digit d is placed at angle theta_d = pi * d / (num_classes - 1) on the
    unit circle, embedded in `anchor_dim` dimensions. The dot product between
    anchors for digits i and j is cos(pi * |i-j| / (num_classes - 1)), so
    similarity decreases smoothly with digit distance.
    """
    assert anchor_dim >= 2, "anchor_dim must be at least 2"
    anchors = torch.zeros(num_classes, anchor_dim)
    max_d = num_classes - 1
    for d in range(num_classes):
        theta = math.pi * d / max_d
        anchors[d, 0] = math.cos(theta)
        anchors[d, 1] = math.sin(theta)
    # Already unit-norm because cos^2 + sin^2 = 1.
    return anchors


class OrdinalCorrespondenceModel(nn.Module):
    """Same architecture as v2, but anchors are ordinal/circular instead of QR."""

    def __init__(
        self,
        base_model: M7_ContinualAdaptiveModel,
        head_type: str = "mlp",
        anchor_dim: int = 64,
        num_anchors: int = 10,
        temperature: float = 0.1,
    ):
        super().__init__()
        assert head_type in ("mlp", "linear")
        self.head_type = head_type
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad = False

        shared_dim = base_model.cfg.shared_dim
        if head_type == "mlp":
            self.visual_proj = nn.Sequential(
                nn.Linear(shared_dim, 256), nn.GELU(), nn.Linear(256, anchor_dim)
            )
            self.audio_proj = nn.Sequential(
                nn.Linear(shared_dim, 256), nn.GELU(), nn.Linear(256, anchor_dim)
            )
        else:
            self.visual_proj = nn.Linear(shared_dim, anchor_dim)
            self.audio_proj = nn.Linear(shared_dim, anchor_dim)

        anchors = build_ordinal_anchors(num_anchors, anchor_dim)
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
#  PER-SEED RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def compute_confusion_matrix(
    model: OrdinalCorrespondenceModel,
    cached: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    classes: Tuple[int, ...] = (6, 7, 8, 9),
    device=None,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute held-out confusion matrices for V2A and A2V retrieval.
    Returns (v2a_confusion, a2v_confusion, v2a_acc, a2v_acc).
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

    prototypes_v = {c: vis_z[vis_lbls == c].mean(axis=0) for c in classes}
    prototypes_a = {c: aud_z[aud_lbls == c].mean(axis=0) for c in classes}
    for c in classes:
        prototypes_v[c] = prototypes_v[c] / (np.linalg.norm(prototypes_v[c]) + 1e-12)
        prototypes_a[c] = prototypes_a[c] / (np.linalg.norm(prototypes_a[c]) + 1e-12)

    n = len(classes)
    v2a_conf = np.zeros((n, n), dtype=int)
    a2v_conf = np.zeros((n, n), dtype=int)

    for i, c in enumerate(classes):
        sims = np.array([np.dot(prototypes_v[c], prototypes_a[d]) for d in classes])
        v2a_conf[i, sims.argmax()] += 1

        sims = np.array([np.dot(prototypes_a[c], prototypes_v[d]) for d in classes])
        a2v_conf[i, sims.argmax()] += 1

    v2a_acc = np.trace(v2a_conf) / n
    a2v_acc = np.trace(a2v_conf) / n
    return v2a_conf, a2v_conf, v2a_acc, a2v_acc


def ordinal_error_score(conf: np.ndarray, classes: Tuple[int, ...]) -> float:
    """
    Fraction of off-diagonal confusions that are ordinal-adjacent
    (e.g. 6<->7 or 7<->8) rather than non-adjacent.
    """
    n = len(classes)
    off_total = off_adjacent = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            off_total += conf[i, j]
            if abs(classes[i] - classes[j]) == 1:
                off_adjacent += conf[i, j]
    return off_adjacent / off_total if off_total > 0 else 0.0


def run_one_seed(
    seed: int,
    cfg: ExperimentConfig,
    loaders: Dict,
    cached: Dict,
    device,
    head_type: str,
) -> Dict:
    print(f"\n{'='*70}")
    print(f"  SEED {seed} | HEAD {head_type.upper()} | ORDINAL ANCHORS")
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
    corr_main = OrdinalCorrespondenceModel(
        base_model, head_type=head_type, num_anchors=10
    ).to(device)
    anchors_main_before = corr_main.anchors.clone()

    acc_nmnist_before = corr_main.task_accuracy(
        loaders["nmnist"]["test"], "nmnist", "nmnist", device
    )
    acc_shd_before = corr_main.task_accuracy(
        loaders["shd"]["test"], "shd", "shd", device
    )

    untrained_transfer = evaluate_cross_modal_transfer(
        corr_main,
        loaders["nmnist"]["test"],
        loaders["shd"]["test"],
        device,
        num_classes=10,
        samples_per_class=100,
        random_state=seed,
    )

    train_ds, val_ds = build_paired_feature_dataset(
        cached, samples_per_class=500, seed=seed
    )
    train_correspondence(
        corr_main,
        train_ds,
        val_ds,
        device,
        num_epochs=20,
        lr=1e-3,
        batch_size=128,
        patience=5,
    )

    acc_nmnist_after = corr_main.task_accuracy(
        loaders["nmnist"]["test"], "nmnist", "nmnist", device
    )
    acc_shd_after = corr_main.task_accuracy(
        loaders["shd"]["test"], "shd", "shd", device
    )

    # Invariant asserts
    assert hash_nmnist_before == _hash_params(base_model, "backbones.nmnist"), "N-MNIST backbone drifted!"
    assert hash_shd_before == _hash_params(base_model, "backbones.shd"), "SHD backbone drifted!"
    assert hash_heads_before == _hash_params(base_model, "heads."), "Task heads drifted!"
    assert torch.allclose(anchors_main_before, corr_main.anchors), "Anchors changed!"
    assert abs(acc_nmnist_before - acc_nmnist_after) < 0.01, "N-MNIST accuracy changed!"
    assert abs(acc_shd_before - acc_shd_after) < 0.01, "SHD accuracy changed!"
    print("   ✅ All main-config invariant asserts passed")

    trained_transfer = evaluate_cross_modal_transfer(
        corr_main,
        loaders["nmnist"]["test"],
        loaders["shd"]["test"],
        device,
        num_classes=10,
        samples_per_class=100,
        random_state=seed,
    )

    # ── HELD-OUT CONFIG (0-5, 6 ordinal anchors) ──
    base_model2 = M7_ContinualAdaptiveModel(tasks=DIGIT_TASKS, cfg=cfg).to(device)
    base_model2.load_state_dict(ckpt["model_state_dict"])
    corr_ho = OrdinalCorrespondenceModel(
        base_model2, head_type=head_type, num_anchors=6
    ).to(device)

    train_correspondence(
        corr_ho,
        train_ds,
        val_ds,
        device,
        num_epochs=20,
        lr=1e-3,
        batch_size=128,
        patience=5,
        held_out_classes=(6, 7, 8, 9),
    )
    held_out = compute_held_out_metrics(
        corr_ho, cached, held_out_classes=(6, 7, 8, 9), device=device
    )

    # Confusion matrices / error structure
    v2a_conf, a2v_conf, v2a_acc, a2v_acc = compute_confusion_matrix(
        corr_ho, cached, classes=(6, 7, 8, 9), device=device
    )
    v2a_adj = ordinal_error_score(v2a_conf, (6, 7, 8, 9))
    a2v_adj = ordinal_error_score(a2v_conf, (6, 7, 8, 9))

    # ── Shuffled-label control ──
    base_model3 = M7_ContinualAdaptiveModel(tasks=DIGIT_TASKS, cfg=cfg).to(device)
    base_model3.load_state_dict(ckpt["model_state_dict"])
    corr_shuffle = OrdinalCorrespondenceModel(
        base_model3, head_type=head_type, num_anchors=10
    ).to(device)
    train_correspondence(
        corr_shuffle,
        train_ds,
        val_ds,
        device,
        num_epochs=20,
        lr=1e-3,
        batch_size=128,
        patience=5,
        shuffled_labels=True,
    )
    shuffle_ho = compute_held_out_metrics(
        corr_shuffle, cached, held_out_classes=(6, 7, 8, 9), device=device
    )
    s_v2a_conf, s_a2v_conf, _, _ = compute_confusion_matrix(
        corr_shuffle, cached, classes=(6, 7, 8, 9), device=device
    )
    s_v2a_adj = ordinal_error_score(s_v2a_conf, (6, 7, 8, 9))

    verdict = (
        "discovered ordinal structure"
        if held_out["held_out_v2a_retrieval"] > 0.25 + 0.1
        else "engineered bridge"
    )
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
        "held_out_same_align": held_out["held_out_same_align"],
        "held_out_null_align": held_out["held_out_null_align"],
        "held_out_gap": held_out["held_out_gap"],
        "held_out_v2a_retrieval": held_out["held_out_v2a_retrieval"] * 100,
        "held_out_a2v_retrieval": held_out["held_out_a2v_retrieval"] * 100,
        "held_out_v2a_adjacent_error": v2a_adj,
        "held_out_a2v_adjacent_error": a2v_adj,
        "shuffle_v2a_retrieval": shuffle_ho["held_out_v2a_retrieval"] * 100,
        "shuffle_v2a_adjacent_error": s_v2a_adj,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RDM_DIR.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        seed=42,
        task_order=["nmnist", "shd", "dvs"],
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        max_epochs=50,
        patience=12,
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
        align_shared_dim=512,
        align_weight=0.5,
        align_temperature=0.1,
        align_use_orthogonal=False,
        align_orthogonal_weight=0.01,
    )
    device = get_device(cfg)
    print(f"Device: {device}")

    print("\nBuilding digit loaders...")
    loaders = _build_test_loaders(cfg)

    print("\nLoading cached backbone features (from v2)...")
    cached = _load_cached_features(device)

    # ── STAGE 0: RDM diagnostic ──
    print("\n" + "=" * 70)
    print("STAGE 0 — RDM DIAGNOSTIC")
    print("=" * 70)
    rdm_results = compute_rdms(cached)
    print(f"   Vision↔Audio RDM correlation (Mantel): r = {rdm_results['vision_audio_rdm_r']:.4f}, p = {rdm_results['vision_audio_rdm_p']:.4f}")
    print(f"   Vision magnitude structure:  rho = {rdm_results['vision_magnitude_rho']:.4f}, p = {rdm_results['vision_magnitude_p']:.4f}")
    print(f"   Audio magnitude structure:   rho = {rdm_results['audio_magnitude_rho']:.4f}, p = {rdm_results['audio_magnitude_p']:.4f}")

    rdm_summary = {
        "vision_audio_rdm_r": float(rdm_results["vision_audio_rdm_r"]),
        "vision_audio_rdm_p": float(rdm_results["vision_audio_rdm_p"]),
        "vision_magnitude_rho": float(rdm_results["vision_magnitude_rho"]),
        "vision_magnitude_p": float(rdm_results["vision_magnitude_p"]),
        "audio_magnitude_rho": float(rdm_results["audio_magnitude_rho"]),
        "audio_magnitude_p": float(rdm_results["audio_magnitude_p"]),
    }
    with open(RDM_DIR / "rdm_analysis.txt", "w") as f:
        f.write("STAGE 0 — RDM DIAGNOSTIC\n")
        f.write("=" * 70 + "\n\n")
        for k, v in rdm_summary.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\nInterpretation:\n")
        if rdm_summary["vision_audio_rdm_r"] > 0.3 and rdm_summary["vision_audio_rdm_p"] < 0.05:
            f.write("- RDMs are significantly correlated: shared relational structure is possible.\n")
        else:
            f.write("- RDMs are NOT significantly correlated: no strong shared relational structure.\n")
        if abs(rdm_summary["vision_magnitude_rho"]) > 0.3 and rdm_summary["vision_magnitude_p"] < 0.05:
            f.write("- Vision RDM is magnitude-structured.\n")
        if abs(rdm_summary["audio_magnitude_rho"]) > 0.3 and rdm_summary["audio_magnitude_p"] < 0.05:
            f.write("- Audio RDM is magnitude-structured.\n")

    np.save(RDM_DIR / "rdm_vision.npy", rdm_results["rdm_vision"])
    np.save(RDM_DIR / "rdm_audio.npy", rdm_results["rdm_audio"])
    np.save(RDM_DIR / "prototypes_vision.npy", rdm_results["prototypes_vision"])
    np.save(RDM_DIR / "prototypes_audio.npy", rdm_results["prototypes_audio"])

    # ── STAGE 1: Ordinal anchor experiment ──
    print("\n" + "=" * 70)
    print("STAGE 1 — ORDINAL ANCHOR ALIGNMENT")
    print("=" * 70)

    # Procrustes on frozen backbones (same as v2 — baseline)
    print("\nPROCRUSTES BASELINE (frozen backbones)")
    procrustes = run_procrustes(
        cached, train_classes=(0, 1, 2, 3, 4, 5), held_out_classes=(6, 7, 8, 9)
    )
    for k, v in procrustes.items():
        print(f"   {k}: {v:.4f}")
    with open(RESULTS_DIR / "procrustes.json", "w") as f:
        json.dump({k: float(v) for k, v in procrustes.items()}, f, indent=2)

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

    # Aggregate
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
            majority = (
                "discovered ordinal structure"
                if verdicts.count("discovered ordinal structure") > len(verdicts) / 2
                else "engineered bridge"
            )
            print(f"\n   Majority verdict: {majority.upper()}")

        # Report
        with open(RESULTS_DIR / "report.txt", "w") as f:
            f.write("ORDINAL ANCHOR ALIGNMENT (B5) — REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write("PRE-REGISTERED EXPECTATIONS:\n")
            f.write("- If frozen backbones encode digit magnitude, ordinal anchors\n")
            f.write("  should enable held-out 6-9 cross-modal retrieval > 25%.\n")
            f.write("- Ordinal-adjacent confusions (6<->7, 7<->8) support magnitude structure.\n")
            f.write("- Non-ordinal confusions (6<->2) indicate no meaningful geometry.\n\n")

            f.write("STAGE 0 — RDM DIAGNOSTIC:\n")
            for k, v in rdm_summary.items():
                f.write(f"  {k}: {v:.6f}\n")
            f.write("\n")

            f.write("STAGE 1 — ORDINAL ANCHOR RESULTS:\n")
            for head_type in ("mlp", "linear"):
                sub = [r for r in all_results if r["head_type"] == head_type]
                if not sub:
                    continue
                f.write(f"\n{head_type.upper()} HEAD (n={len(sub)})\n")
                f.write(
                    f"  Held-out V2A retrieval: {np.mean([r['held_out_v2a_retrieval'] for r in sub]):.2f}% ± "
                    f"{np.std([r['held_out_v2a_retrieval'] for r in sub]):.2f}%\n"
                )
                f.write(
                    f"  Held-out gap: {np.mean([r['held_out_gap'] for r in sub]):.4f} ± "
                    f"{np.std([r['held_out_gap'] for r in sub]):.4f}\n"
                )
                f.write(
                    f"  V2A adjacent errors: {np.mean([r['held_out_v2a_adjacent_error'] for r in sub]):.2f} ± "
                    f"{np.std([r['held_out_v2a_adjacent_error'] for r in sub]):.2f}\n"
                )
                f.write(
                    f"  Main V2A: {np.mean([r['trained_v2a'] for r in sub]):.2f}% ± "
                    f"{np.std([r['trained_v2a'] for r in sub]):.2f}%\n"
                )
                verdicts = [r["verdict"] for r in sub]
                maj = (
                    "discovered ordinal structure"
                    if verdicts.count("discovered ordinal structure") > len(verdicts) / 2
                    else "engineered bridge"
                )
                f.write(f"  Verdict: {maj.upper()}\n")

            f.write(f"\nPROCRUSTES BASELINE:\n")
            for k, v in procrustes.items():
                f.write(f"  {k}: {v:.4f}\n")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()

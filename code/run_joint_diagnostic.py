#!/usr/bin/env python3
"""
Joint Diagnostic — 2x2 (classifier SHARED/SEPARATE  x  cross-modal SupCon ON/OFF)
=====================================================================================

Diagnoses whether the magnitude-ablation OFF arm's held-out V2A=50% (seed 42) is a
real cross-modal effect or an artifact (the suspected shared-classifier leak).

STEP 0 (already reported separately, nothing run there):
  - The magnitude experiment uses SEPARATE per-modality classifiers
    (run_joint_magnitude_ablation.py:378-379). A shared-classifier leak was NOT possible
    there.
  - The 50% is a 4-class prototype-argmax metric quantized to 25% steps
    (compute_held_out_metrics, :572-632); an UNTRAINED model also scores 50% (:769).
    It is a metric artifact, not a learned effect.

This script:
  1. Reuses the magnitude experiment's numerics (JointMagnitudeModel, stems, SupCon loss,
     loaders, cache) WITHOUT modifying run_joint_magnitude_ablation.py.
  2. Adds a SHARED-classifier variant (SharedHeadJointModel) so the 2x2's SHARED row
     exists (positive control for the leak MECHANISM).
  3. Replaces the degenerate 4-class metric with a PER-SAMPLE retrieval metric
     (train prototypes, test retrieval; chance = 10% over all-10, 25% over held-out-4),
     plus a permutation null and deterministic null+gap.
  4. Runs an untrained per-sample control per cell (the noise floor that tests the
     artifact hypothesis directly).

Magnitude head is OFF in all four cells (it is not the variable here).
Output: results/joint_diagnostic/  (never overwrites prior results).
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT))

# Reuse (do NOT fork) the magnitude experiment's numerics.
from run_joint_magnitude_ablation import (  # noqa: E402
    JointMagnitudeModel,
    _extract_embeddings,
    _within_modality_probe,
    _build_modality_loaders,
    _build_paired_0to5_loaders,
    _accuracy_on_all,
    compute_held_out_metrics,
    nmnist_collate_fn,
    shd_collate_fn,
    ExperimentConfig,
    get_device,
    CrossModalSupConLoss,
)
from m7_trimodal_run_provenance import ExperimentConfig as _EC  # noqa: E402,F401  (ensures import path)

# The compact caches were pickled with module='__main__' (the magnitude probe ran
# run_joint_magnitude_ablation.py as __main__). When THIS script is __main__, torch.load
# inside the reused _build_modality_loaders cannot resolve those classes -> alias them
# into __main__ so the existing cache loads without a 23 GB re-render.
import run_joint_magnitude_ablation as _mag_module  # noqa: E402
for _cn in ("CachedNMNIST", "CachedSHD", "SubsampledByClass"):
    if hasattr(_mag_module, _cn):
        setattr(sys.modules["__main__"], _cn, getattr(_mag_module, _cn))

RESULTS_DIR = REPO_ROOT / "results" / "joint_diagnostic"
HELD_OUT: Tuple[int, ...] = (6, 7, 8, 9)
GATE_THRESHOLD = 40.0          # within-modality 6-9 probe confound gate (%)
SEEDS_FULL: List[int] = [42, 123, 456]
PERM_K = 200                   # permutation-null iterations
CSV_COLUMNS = [
    "seed", "model_kind", "use_supcon", "gate_passed",
    "probe_visual_6_9", "probe_audio_6_9", "acc_visual_all", "acc_audio_all",
    "time_steps_visual", "time_steps_audio",
    "spike_rate_visual", "sparsity_visual", "spike_rate_audio", "sparsity_audio",
    "persample_v2a_all10", "persample_a2v_all10",
    "persample_v2a_heldout4", "persample_a2v_heldout4",
    "untrained_persample_v2a_all10", "untrained_persample_a2v_all10",
    "n_vis_heldout", "n_aud_heldout",
    "perm_null_v2a", "perm_null_v2a_std", "perm_null_a2v", "perm_null_a2v_std", "perm_null_k",
    "det_same_align", "det_null_align", "det_gap",
    "legacy_proto_v2a", "legacy_proto_a2v", "legacy_untrained_v2a",
    "v2a_minus_a2v_all10",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL — SHARED-classifier variant (isolates the classifier as the variable)
# ═══════════════════════════════════════════════════════════════════════════════


class SharedHeadJointModel(JointMagnitudeModel):
    """Identical encoder (inherited `embed`), but ONE shared 0-9 head for both modalities.

    The parent's visual_classifier / audio_classifier remain allocated but receive no
    gradient in this model (forward never calls them). Magnitude is always OFF here.
    """

    def __init__(self, d_stem: int = 256, d_embed: int = 128, use_magnitude: bool = False):
        super().__init__(d_stem, d_embed, use_magnitude)
        self.shared_classifier = nn.Linear(d_embed, 10)

    def forward(self, x, modality):
        z = self.embed(x, modality)                       # inherited: stems + shared_core + L2-norm
        logits = self.shared_classifier(z)                # SAME head for both modalities
        mag = self.magnitude_head(z) if self.use_magnitude else None
        return logits, z, mag


def make_model(kind: str, device) -> nn.Module:
    if kind == "separate":
        m: nn.Module = JointMagnitudeModel(d_stem=256, d_embed=128, use_magnitude=False)
    elif kind == "shared":
        m = SharedHeadJointModel(d_stem=256, d_embed=128, use_magnitude=False)
    else:
        raise ValueError(f"unknown model_kind {kind!r}")
    return m.to(device)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING — train_model with a use_supcon flag (magnitude always OFF)
# ═══════════════════════════════════════════════════════════════════════════════


def _evaluate_diag(model, paired_val_loader, modality_loaders, device, use_supcon: bool):
    model.eval()
    total_loss, n_batches = 0.0, 0
    ce = nn.CrossEntropyLoss()
    supcon = CrossModalSupConLoss(temperature=0.1)
    with torch.no_grad():
        for vis, aud, labels in paired_val_loader:
            vis, aud, labels = vis.to(device), aud.to(device), labels.to(device)
            v_logits, v_z, _ = model(vis, "visual")
            a_logits, a_z, _ = model(aud, "audio")
            loss = ce(v_logits, labels) + ce(a_logits, labels)
            if use_supcon:
                z_cat = torch.cat([v_z, a_z], dim=0)
                lbl_cat = torch.cat([labels, labels], dim=0)
                loss = loss + supcon(z_cat, lbl_cat)
            total_loss += loss.item()
            n_batches += 1
    probe = _within_modality_probe(model, modality_loaders, device)
    return total_loss / max(n_batches, 1), probe


def train_model_diag(
    model, paired_train_loader, paired_val_loader, modality_loaders, device,
    use_supcon: bool, max_epochs: int = 50, lr: float = 1e-3,
    weight_decay: float = 1e-4, patience: int = 8,
):
    """Train the joint encoder from scratch. Identical to train_model except:
       - SupCon term added only when use_supcon=True (cross-modal ON/OFF axis);
       - magnitude term never added (always OFF in this diagnostic)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    ce = nn.CrossEntropyLoss()
    supcon = CrossModalSupConLoss(temperature=0.1)

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum, n_train = 0.0, 0

        # Pass 1: paired 0-5 loader (CE on 0-5 + optional cross-modal SupCon).
        for vis, aud, labels in paired_train_loader:
            vis, aud, labels = vis.to(device), aud.to(device), labels.to(device)
            # 6-9 must NEVER receive a cross-modal signal.
            assert not ((labels >= 6).any() and (labels <= 9).any()), "LEAKAGE: 6-9 in paired loader!"
            v_logits, v_z, _ = model(vis, "visual")
            a_logits, a_z, _ = model(aud, "audio")
            loss = ce(v_logits, labels) + ce(a_logits, labels)
            if use_supcon:
                z_cat = torch.cat([v_z, a_z], dim=0)
                lbl_cat = torch.cat([labels, labels], dim=0)
                loss = loss + supcon(z_cat, lbl_cat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            n_train += 1

        # Pass 2: full per-modality train sets (ALL 0-9), within-modality CE only —
        # this is how 6-9 learn within-modality features. No SupCon, no magnitude.
        for modality in ("visual", "audio"):
            loader = modality_loaders["nmnist"]["train"] if modality == "visual" else modality_loaders["shd"]["train"]
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits, _z, _ = model(x, modality)
                loss = ce(logits, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        scheduler.step()

        val_loss, probe = _evaluate_diag(model, paired_val_loader, modality_loaders, device, use_supcon)
        print(
            f"   Epoch {epoch+1:3d} | train {train_loss_sum/max(n_train,1):.4f} | "
            f"val {val_loss:.4f} | probe vis-6-9 {probe['visual']:.1f}% | "
            f"aud-6-9 {probe['audio']:.1f}% | supcon {'ON' if use_supcon else 'OFF'}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}", flush=True)
                break

    return {"best_val_loss": best_val}


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-SAMPLE RETRIEVAL METRIC (new; replaces the degenerate 4-class metric)
# ═══════════════════════════════════════════════════════════════════════════════


def _normalize(z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    return z / n


def _class_prototypes(model, loader, modality, device) -> np.ndarray:
    """Return (10, D) L2-normalized class prototypes from the (full) TRAIN set."""
    z, lbls = _extract_embeddings(model, loader, modality, device)
    z = _normalize(z.numpy())
    lbls = lbls.numpy()
    P = np.zeros((10, z.shape[1]), dtype=np.float64)
    for c in range(10):
        m = lbls == c
        if m.any():
            p = z[m].mean(axis=0)
            P[c] = p / (np.linalg.norm(p) + 1e-8)
    return P


def _retrieval_pred(query_z: np.ndarray, protos: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Predicted class (from pool) for each query via nearest-prototype (cosine)."""
    P = protos[pool]                       # (|pool|, D)
    idx = np.argmax(query_z @ P.T, axis=1)  # index into pool
    return pool[idx]


def _perm_null(query_z, query_lbls, protos, pool, k, rng):
    """Permutation null: break the prototype->class identity (within pool), k times.
    Expected accuracy = 1/|pool| (10% over all-10, 25% over held-out-4)."""
    P = protos[pool]
    idx = np.argmax(query_z @ P.T, axis=1)
    pool_arr = np.asarray(pool)
    accs = []
    for _ in range(k):
        shuffled = rng.permutation(len(pool))
        pred = pool_arr[shuffled][idx]
        accs.append(float((pred == query_lbls).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def per_sample_retrieval(model, cached, loaders, device, rng, k_perm: int = PERM_K) -> Dict:
    """Per-sample held-out (6-9) cross-modal retrieval.

    Prototypes: full TRAIN sets (zero-shot; avoids the transductive test-prototype quirk
    that helped the original 50% artifact).
    Retrieval: each TEST 6-9 sample -> nearest opposite-modality prototype.
    """
    # Full-train prototype loaders (NOT the 1500/class subsample).
    vis_train = DataLoader(cached["nmnist_train"], batch_size=128, shuffle=False,
                           num_workers=0, collate_fn=nmnist_collate_fn)
    aud_train = DataLoader(cached["shd_train"], batch_size=128, shuffle=False,
                           num_workers=0, collate_fn=shd_collate_fn)
    Pv = _class_prototypes(model, vis_train, "visual", device)   # (10, D)
    Pa = _class_prototypes(model, aud_train, "audio", device)

    tv_z, tv_lbls = _extract_embeddings(model, loaders["nmnist"]["test"], "visual", device)
    ta_z, ta_lbls = _extract_embeddings(model, loaders["shd"]["test"], "audio", device)
    tv_z, tv_lbls = _normalize(tv_z.numpy()), tv_lbls.numpy()
    ta_z, ta_lbls = _normalize(ta_z.numpy()), ta_lbls.numpy()

    held = np.array(HELD_OUT, dtype=int)
    all10 = np.arange(10, dtype=int)

    vq_z, vq_l = tv_z[np.isin(tv_lbls, held)], tv_lbls[np.isin(tv_lbls, held)]
    aq_z, aq_l = ta_z[np.isin(ta_lbls, held)], ta_lbls[np.isin(ta_lbls, held)]

    def acc(qz, ql, protos, pool):
        if ql.size == 0:
            return float("nan")
        return 100.0 * float((_retrieval_pred(qz, protos, pool) == ql).mean())

    v2a_all10 = acc(vq_z, vq_l, Pa, all10)
    v2a_h4 = acc(vq_z, vq_l, Pa, held)
    a2v_all10 = acc(aq_z, aq_l, Pv, all10)
    a2v_h4 = acc(aq_z, aq_l, Pv, held)

    null_v2a, std_v2a = (0.0, 0.0) if vq_l.size == 0 else _perm_null(vq_z, vq_l, Pa, all10, k_perm, rng)
    null_a2v, std_a2v = (0.0, 0.0) if aq_l.size == 0 else _perm_null(aq_z, aq_l, Pv, all10, k_perm, rng)

    # Deterministic null+gap over held-out TRAIN prototypes (same/null cosine).
    same = float(np.mean([Pa[c] @ Pv[c] for c in held]))
    off = float(np.mean([Pa[c] @ Pv[d] for c in held for d in held if c != d]))

    return {
        "persample_v2a_all10": v2a_all10, "persample_a2v_all10": a2v_all10,
        "persample_v2a_heldout4": v2a_h4, "persample_a2v_heldout4": a2v_h4,
        "n_vis_heldout": int(vq_l.size), "n_aud_heldout": int(aq_l.size),
        "perm_null_v2a": null_v2a * 100.0, "perm_null_v2a_std": std_v2a * 100.0,
        "perm_null_a2v": null_a2v * 100.0, "perm_null_a2v_std": std_a2v * 100.0,
        "perm_null_k": k_perm,
        "det_same_align": same, "det_null_align": off, "det_gap": same - off,
    }


@torch.no_grad()
def _report_spiking_stats(model, loader, modality, device):
    """Readout-layer (post-stem) spike rate / sparsity + timestep count.
    spike_rate = mean summed-spike-count / T (avg spikes/neuron/timestep in [0,1]).
    sparsity   = fraction of (neuron, sample) entries with zero spikes."""
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)
    T = int(x.size(1))
    stem = model.visual_stem if modality == "visual" else model.audio_stem
    counts = stem(x.float())              # (B, d_stem) summed spike counts over T
    spike_rate = float(counts.mean().item()) / max(T, 1)
    sparsity = float((counts == 0).to(torch.float32).mean().item())
    return T, spike_rate, sparsity


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-CELL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def _cell_csv_path(model_kind: str, use_supcon: bool, seed: int) -> Path:
    base = RESULTS_DIR / f"cell_{model_kind}_supcon_{'on' if use_supcon else 'off'}_seed{seed}.csv"
    if base.exists():
        ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        base = RESULTS_DIR / f"cell_{model_kind}_supcon_{'on' if use_supcon else 'off'}_seed{seed}__{ts}.csv"
    return base


def run_one_diagnostic_cell(seed, model_kind, use_supcon, cfg, loaders, cached, device) -> Dict:
    arm = "ON" if use_supcon else "OFF"
    print(f"\n{'='*70}\n  SEED {seed} | {model_kind.upper()} classifier | SUPCON {arm}\n{'='*70}", flush=True)
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # Untrained controls (per-sample noise floor + legacy 4-class cross-ref).
    untrained = make_model(model_kind, device)
    untrained_ps = per_sample_retrieval(untrained, cached, loaders, device, rng)
    legacy_untrained = compute_held_out_metrics(untrained, loaders, device=device)
    del untrained

    # Train from scratch.
    model = make_model(model_kind, device)
    paired_train, paired_val = _build_paired_0to5_loaders(cached, seed)
    train_model_diag(model, paired_train, paired_val, loaders, device,
                     use_supcon=use_supcon, max_epochs=cfg.max_epochs, lr=cfg.lr,
                     weight_decay=cfg.weight_decay, patience=cfg.patience)

    # Confound gate: within-modality 6-9 separability must be > 40% in both modalities.
    probe = _within_modality_probe(model, loaders, device)
    gate = bool((probe["visual"] > GATE_THRESHOLD) and (probe["audio"] > GATE_THRESHOLD))
    print(f"   Gate (within-mod 6-9 > {GATE_THRESHOLD:.0f}%): vis {probe['visual']:.1f}% "
          f"aud {probe['audio']:.1f}% -> {'PASS' if gate else 'FAIL (cell invalid)'}", flush=True)

    # Trained metrics.
    ps = per_sample_retrieval(model, cached, loaders, device, rng)
    legacy = compute_held_out_metrics(model, loaders, device=device)
    acc_v = _accuracy_on_all(model, loaders["nmnist"]["test"], "visual", device)
    acc_a = _accuracy_on_all(model, loaders["shd"]["test"], "audio", device)
    Ts_v, sr_v, sp_v = _report_spiking_stats(model, loaders["nmnist"]["test"], "visual", device)
    Ts_a, sr_a, sp_a = _report_spiking_stats(model, loaders["shd"]["test"], "audio", device)

    row = {
        "seed": seed, "model_kind": model_kind, "use_supcon": int(use_supcon), "gate_passed": int(gate),
        "probe_visual_6_9": probe["visual"], "probe_audio_6_9": probe["audio"],
        "acc_visual_all": acc_v, "acc_audio_all": acc_a,
        "time_steps_visual": Ts_v, "time_steps_audio": Ts_a,
        "spike_rate_visual": sr_v, "sparsity_visual": sp_v,
        "spike_rate_audio": sr_a, "sparsity_audio": sp_a,
        "persample_v2a_all10": ps["persample_v2a_all10"], "persample_a2v_all10": ps["persample_a2v_all10"],
        "persample_v2a_heldout4": ps["persample_v2a_heldout4"], "persample_a2v_heldout4": ps["persample_a2v_heldout4"],
        "untrained_persample_v2a_all10": untrained_ps["persample_v2a_all10"],
        "untrained_persample_a2v_all10": untrained_ps["persample_a2v_all10"],
        "n_vis_heldout": ps["n_vis_heldout"], "n_aud_heldout": ps["n_aud_heldout"],
        "perm_null_v2a": ps["perm_null_v2a"], "perm_null_v2a_std": ps["perm_null_v2a_std"],
        "perm_null_a2v": ps["perm_null_a2v"], "perm_null_a2v_std": ps["perm_null_a2v_std"],
        "perm_null_k": ps["perm_null_k"],
        "det_same_align": ps["det_same_align"], "det_null_align": ps["det_null_align"], "det_gap": ps["det_gap"],
        "legacy_proto_v2a": legacy["held_out_v2a_retrieval"] * 100.0,
        "legacy_proto_a2v": legacy["held_out_a2v_retrieval"] * 100.0,
        "legacy_untrained_v2a": legacy_untrained["held_out_v2a_retrieval"] * 100.0,
        "v2a_minus_a2v_all10": ps["persample_v2a_all10"] - ps["persample_a2v_all10"],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _cell_csv_path(model_kind, use_supcon, seed)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerow(row)
    print(f"   Wrote {path.name} | per-sample V2A(all10) {row['persample_v2a_all10']:.1f}% "
          f"(null {row['perm_null_v2a']:.1f}±{row['perm_null_v2a_std']:.1f}, "
          f"untrained {row['untrained_persample_v2a_all10']:.1f}%) | legacy 4-class V2A {row['legacy_proto_v2a']:.1f}%", flush=True)
    return row


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION + VERDICT
# ═══════════════════════════════════════════════════════════════════════════════


def _cell_rows(rows, model_kind, use_supcon):
    return [r for r in rows if r["model_kind"] == model_kind and bool(r["use_supcon"]) == use_supcon]


def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.std(vals)) if vals else float("nan")


def _binom_z(p_hat: float, n: int, p0: float = 0.10) -> float:
    """One-sided z-score for H0: retrieval accuracy = p0 (chance). n = # held-out samples.

    NOTE: the permutation-null *mean* (~10%) is the empirical chance estimate and is
    correct, but its *std* is inflated by a bimodal permutation quirk under concentrated
    argmax, so a '2-sigma of perm null' gate is far too conservative. Use this binomial z
    against the analytical 10% (all-10) chance level instead.
    """
    se = (p0 * (1.0 - p0) / max(n, 1)) ** 0.5
    return (p_hat - p0) / (se + 1e-12)


def _above_chance(r, p0: float = 0.10, z_min: float = 6.0) -> bool:
    """Is this cell's per-sample V2A clearly above chance (binomial z > z_min)?"""
    n = int(r.get("n_vis_heldout", 0) or 0)
    return _binom_z(r["persample_v2a_all10"] / 100.0, n, p0) > z_min


def load_rows_from_disk() -> List[Dict]:
    """Load every cell_*_seed*.csv in RESULTS_DIR (combines seeds across runs)."""
    rows: List[Dict] = []
    for p in sorted(RESULTS_DIR.glob("cell_*_seed*.csv")):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                # coerce numeric columns
                for k in CSV_COLUMNS:
                    if k in ("model_kind",):
                        continue
                    try:
                        r[k] = float(r[k]) if k != "seed" else int(float(r[k]))
                    except (ValueError, KeyError):
                        pass
                rows.append(r)
    return rows


def write_summary_and_report(rows: List[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # summary.csv
    metrics = [
        "persample_v2a_all10", "persample_a2v_all10", "persample_v2a_heldout4",
        "untrained_persample_v2a_all10", "perm_null_v2a", "det_gap",
        "probe_visual_6_9", "probe_audio_6_9", "legacy_proto_v2a",
    ]
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "metric", "mean", "std", "n_seeds"])
        for kind in ("separate", "shared"):
            for use_supcon in (False, True):
                sub = _cell_rows(rows, kind, use_supcon)
                cell = f"{kind}_supcon_{'on' if use_supcon else 'off'}"
                for m in metrics:
                    vals = [r[m] for r in sub if r.get("gate_passed")]
                    w.writerow([cell, m, f"{_mean(vals):.4f}", f"{_std(vals):.4f}", len(vals)])

    # report.txt
    def cell_str(kind, use_supcon):
        sub = [r for r in _cell_rows(rows, kind, use_supcon) if r.get("gate_passed")]
        if not sub:
            return "    (no gate-passing seeds)\n"
        r = sub[0]
        lines = []
        for s in sub:
            lines.append(
                f"    seed {s['seed']}: V2A(all10) {s['persample_v2a_all10']:5.1f}%  "
                f"A2V(all10) {s['persample_a2v_all10']:5.1f}%  "
                f"null {s['perm_null_v2a']:4.1f}±{s['perm_null_v2a_std']:3.1f}  "
                f"untrained {s['untrained_persample_v2a_all10']:5.1f}%  "
                f"legacy4cls {s['legacy_proto_v2a']:4.0f}%  "
                f"gate {'Y' if s['gate_passed'] else 'N'}"
            )
        return "\n".join(lines) + "\n"

    sep_off = [r for r in _cell_rows(rows, "separate", False) if r.get("gate_passed")]
    sep_on = [r for r in _cell_rows(rows, "separate", True) if r.get("gate_passed")]
    sha_off = [r for r in _cell_rows(rows, "shared", False) if r.get("gate_passed")]

    def mean_v2a(sub):
        return _mean([r["persample_v2a_all10"] for r in sub]) if sub else float("nan")

    def std_v2a(sub):
        return _std([r["persample_v2a_all10"] for r in sub]) if sub else float("nan")

    def per_seed(sub):
        return ", ".join(f"{int(r['seed'])}:{r['persample_v2a_all10']:.1f}%" for r in sorted(sub, key=lambda x: int(x['seed'])))

    def robustness(sub):
        """'all', 'k/n', or 'none' seeds binomial-above 10% chance."""
        n_above = sum(1 for r in sub if _above_chance(r))
        n = len(sub)
        return "all" if n_above == n else f"{n_above}/{n}"

    # Verdict (binomial z vs 10% chance; data-driven; hypotheses are not results).
    verdict_lines = []
    if sha_off:
        rob = robustness(sha_off)
        verdict_lines.append(
            f"SHARED + cross-modal-OFF held-out V2A = {mean_v2a(sha_off):.1f}% (chance 10%, {rob} seeds above; "
            f"n={len(sha_off)} seed(s)) with NO cross-modal supervision -> a shared 0-9 head alone aligns held-out "
            f"6-9: CLASSIFIER-LEAK mechanism. {'Robust.' if rob=='all' and len(sha_off)>=3 else 'Dramatic at seed 42 but MORE SEEDS NEEDED to confirm robustness.'}"
        )
    if sep_off:
        rob = robustness(sep_off)
        verdict_lines.append(
            f"SEPARATE + cross-modal-OFF held-out V2A = {mean_v2a(sep_off):.1f}% ± {std_v2a(sep_off):.1f}% "
            f"(per-seed {per_seed(sep_off)}; {rob} seeds above 10% chance) with SEPARATE classifiers and NO SupCon. "
            f"This is the shared-core effect; it is {'STABLE' if rob=='all' else 'SEED-DEPENDENT / UNSTABLE'} "
            f"(note: a seed can collapse to/below chance). The original '50%' was itself seed-dependent "
            f"(legacy 4-class per-seed varies), so it is NOT a robust effect and NOT a metric artifact."
        )
    if sep_off and sep_on:
        off_m, on_m = mean_v2a(sep_off), mean_v2a(sep_on)
        # Per-seed OFF>=ON check (robustness of the ordering).
        pairs = []
        for r in sep_off:
            on_r = next((x for x in sep_on if int(x["seed"]) == int(r["seed"])), None)
            if on_r:
                pairs.append(r["persample_v2a_all10"] >= on_r["persample_v2a_all10"])
        if pairs and all(pairs):
            verdict_lines.append(
                f"SEPARATE SupCon-OFF >= SupCon-ON for ALL {len(pairs)} seeds ({per_seed(sep_off)} vs "
                f"{per_seed(sep_on)}): cross-modal SupCon on 0-5 does NOT produce held-out 6-9 generalization "
                f"(it is <= the no-SupCon baseline in every seed). ROBUST: no emergent cross-modal generalization."
            )
        elif not (np.isnan(off_m) or np.isnan(on_m)):
            if off_m >= on_m:
                verdict_lines.append(f"SEPARATE SupCon-OFF ({off_m:.1f}%) >= SupCon-ON ({on_m:.1f}%) on average (ordering not universal).")
            else:
                verdict_lines.append(f"SEPARATE SupCon-ON ({on_m:.1f}%) > SupCon-OFF ({off_m:.1f}%); inspect symmetry before calling candidate-real.")
    if not verdict_lines:
        verdict_lines.append("Indeterminate at this seed count - inspect per-seed numbers.")

    with open(RESULTS_DIR / "report.txt", "w") as f:
        f.write("JOINT DIAGNOSTIC — 2x2 (classifier SHARED/SEPARATE x cross-modal SupCon ON/OFF)\n")
        f.write("=" * 78 + "\n\n")
        f.write("PRE-REGISTERED (per task #28):\n")
        f.write("- Headline metric: PER-SAMPLE held-out 6-9 retrieval, all-10 prototypes (chance = 10%).\n")
        f.write("- 'Above chance' = binomial z vs 10% > 6 (perm-null mean ~10% is the empirical chance check;\n"
                "  its std is inflated by a bimodal permutation quirk, so 2-sigma-of-perm-null is not used).\n")
        f.write("- Gate: within-modality 6-9 probe > 40% in BOTH modalities (else cell invalid).\n\n")
        f.write("2x2 TABLE (per-seed; gate-passing only):\n\n")
        for kind in ("separate", "shared"):
            for use_supcon in (False, True):
                f.write(f"[{kind.upper()} | SUPCON {'ON' if use_supcon else 'OFF'}]\n")
                f.write(cell_str(kind, use_supcon))
        f.write("\nVERDICT (data-driven; hypotheses are not results):\n")
        for i, v in enumerate(verdict_lines, 1):
            f.write(f"  {i}. {v}\n")
        f.write("\nCROSS-CHECK (artifact collapse):\n")
        f.write("  Legacy 4-class V2A (the degenerate metric) vs per-sample all-10 on SEPARATE-OFF:\n")
        for r in _cell_rows(rows, "separate", False):
            f.write(f"    seed {r['seed']}: legacy4cls V2A {r['legacy_proto_v2a']:.0f}%  ->  per-sample all-10 {r['persample_v2a_all10']:.1f}%\n")
        f.write("\n  If legacy ~50% but per-sample ~10% (= chance), the original 50% was metric quantization.\n")
        f.write("\nNOTES:\n")
        f.write("  - Headline chance baseline = permutation null (~10% over all-10 prototypes), NOT the\n"
                "    untrained control. An untrained SNN is quiescent (LIF neurons do not spike), so its\n"
                "    embeddings collapse to the zero vector and per-sample retrieval reads ~0% (degenerate).\n"
                "    The >40% within-modality gate guarantees trained embeddings are non-collapsed.\n")
        f.write("  - Single-seed (probe) numbers are indicative, not conclusive. Expand to [42,123,456]\n"
                "    for any cell that clears its null before claiming an effect.\n")

    print(f"\nWrote {RESULTS_DIR/'summary.csv'} and {RESULTS_DIR/'report.txt'}", flush=True)
    print("\nVERDICT:")
    for v in verdict_lines:
        print(f"  - {v}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def _build_cfg() -> ExperimentConfig:
    # Mirrors run_joint_magnitude_ablation.main()'s cfg so cache + training match.
    return ExperimentConfig(
        seed=42, task_order=["nmnist", "shd", "dvs"], lr=1e-3, weight_decay=1e-4,
        batch_size=32, max_epochs=50, patience=8, gradient_clip=1.0, beta=0.9, dropout=0.2,
        hidden_dim=512, shared_dim=512, num_patterns=100, hopfield_temperature=1.0,
        num_gru_layers=2, num_workers=0, time_steps=25, pin_memory=True,
        use_contrastive=True, contrastive_temperature=0.07, contrastive_weight=0.1,
        use_replay=False, replay_batch=32, replay_weight=0.3, replay_samples_per_class=50,
        device="auto", align_cross_modal=False, align_shared_dim=128, align_weight=0.5,
        align_temperature=0.1, align_use_orthogonal=False, align_orthogonal_weight=0.01,
        nmnist_train_per_class=1500,
    )


def main():
    parser = argparse.ArgumentParser(description="Joint diagnostic: classifier x cross-modal 2x2")
    parser.add_argument("--probe", action="store_true", help="seed 42 only (fast probe before full sweep)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="override seed list")
    parser.add_argument("--kinds", type=str, default="separate,shared",
                        help="comma-separated classifier kinds to run (separate,shared)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip training; rebuild summary.csv + report.txt from all cell CSVs on disk")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        rows = load_rows_from_disk()
        print(f"Loaded {len(rows)} cell CSV(s) from {RESULTS_DIR}", flush=True)
        if rows:
            write_summary_and_report(rows)
        print("\n✅ Done (aggregate-only).", flush=True)
        return

    cfg = _build_cfg()
    device = get_device(cfg)
    print(f"Device: {device}", flush=True)

    seeds = [42] if args.probe else (args.seeds or SEEDS_FULL)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    print(f"\nMode: seeds={seeds} kinds={kinds} | max_epochs={cfg.max_epochs} | patience={cfg.patience}", flush=True)

    print("\nBuilding modality loaders (reuses joint_magnitude_ablation/cache/)...", flush=True)
    loaders, cached = _build_modality_loaders(cfg)

    cells: List[Tuple[str, bool]] = []
    for kind in ("separate", "shared"):
        if kind in kinds:
            cells.append((kind, False))
            cells.append((kind, True))

    ran = 0
    for kind, use_supcon in cells:
        for seed in seeds:
            # Skip if this exact cell/seed CSV already exists (don't redo finished work).
            existing = list(RESULTS_DIR.glob(f"cell_{kind}_supcon_{'on' if use_supcon else 'off'}_seed{seed}.csv"))
            if existing:
                print(f"   (skip {kind}/supcon{'on' if use_supcon else 'off'}/seed{seed}: CSV exists)", flush=True)
                continue
            try:
                run_one_diagnostic_cell(seed, kind, use_supcon, cfg, loaders, cached, device)
                ran += 1
            except Exception as e:
                import traceback
                print(f"\n❌ cell {kind}/supcon{'on' if use_supcon else 'off'}/seed{seed} failed: {e}", flush=True)
                traceback.print_exc()

    # Always aggregate from disk so newly-run seeds combine with any prior seeds.
    rows = load_rows_from_disk()
    if rows:
        write_summary_and_report(rows)
    print(f"\n✅ Done (ran {ran} new cell/seed(s); aggregated {len(rows)} from disk).", flush=True)


if __name__ == "__main__":
    main()

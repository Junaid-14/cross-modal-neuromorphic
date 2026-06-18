#!/usr/bin/env python3
"""
Digit-Only Cross-Modal Alignment Ablation
===========================================

Conditions A / B / C on digit tasks only (N-MNIST + SHD).
DVS is NEVER instantiated or trained.

Protocol per cell:
  T1: N-MNIST
  [Alignment phase for B/C only]
  [post_align transfer eval for B/C only]
  T2: SHD
  [post_t2 transfer eval for all]

Resumable: re-running skips completed cells.
Atomic writes: temp file + rename so incomplete writes are never seen.

Outputs:
  results/cross_modal_alignment_digit/
      ├── {condition}_seed{seed}.csv
      ├── {condition}_seed{seed}.config.json
      ├── summary.csv          (mean±std, 95% CI, Wilcoxon)
      ├── wilcoxon.csv
      └── master_config.json
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import tonic
from scipy import stats
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add repo root so we can import from code/ and utils/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT))

from m7_trimodal_run_provenance import (
    ExperimentConfig,
    M7_ContinualAdaptiveModel,
    ContinualLearner,
    EmbeddingReplayBuffer,
    get_device,
    nmnist_collate_fn,
    shd_collate_fn,
    random_init_baseline,
    print_task_sanity,
)
from utils.cross_modal_transfer import evaluate_cross_modal_transfer

# Only digit tasks — DVS backbone/head is NEVER created
DIGIT_TASKS = {
    "nmnist": {"num_classes": 10, "type": "visual"},
    "shd":    {"num_classes": 20, "type": "audio"},
}

# Expected CSV columns for a complete result (used by skip logic)
EXPECTED_COLUMNS = [
    "condition", "seed",
    "T1_val_acc", "SHD_zero_shot_acc", "SHD_val_acc",
    "post_align_V2A", "post_align_A2V",
    "post_align_prototype_alignment", "post_align_prototype_alignment_off_diag",
    "post_align_sil_visual", "post_align_sil_audio", "post_align_chance",
    "post_t2_V2A", "post_t2_A2V",
    "post_t2_prototype_alignment", "post_t2_prototype_alignment_off_diag",
    "post_t2_sil_visual", "post_t2_sil_audio", "post_t2_chance",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS (digit-only — DVS is NEVER touched)
# ═══════════════════════════════════════════════════════════════════════════════

def build_digit_loaders(cfg: ExperimentConfig):
    """Build N-MNIST and SHD loaders. DVS is NEVER loaded."""
    data_dir = Path("./ICONS_M7/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading N-MNIST...")
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

    print("  Loading SHD...")
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
        "train_val_test": (
            DataLoader(shd_train_split, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=shd_collate_fn),
            DataLoader(shd_val_split,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn),
            DataLoader(shd_test_data,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn),
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
        visual_dataset=train_split,
        audio_dataset=shd_train_split,
        batch_size=30,
        samples_per_class=min(500, len(train_split) // 10, len(shd_train_split) // 10),
        num_workers=0,
        seed=42,
        visual_transform=_visual_transform,
        audio_transform=None,
    )

    print("  ✅ DVS was NEVER loaded (digit-only ablation).")
    return {
        "nmnist": nmnist_loaders,
        "shd": shd_loaders,
        "paired": paired_loader,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ATOMIC / RESUMABLE IO
# ═══════════════════════════════════════════════════════════════════════════════

def _is_complete_result(csv_path: Path) -> bool:
    """Return True if csv_path exists and contains all EXPECTED_COLUMNS."""
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
        return all(col in df.columns for col in EXPECTED_COLUMNS)
    except Exception:
        return False


def _atomic_write_csv(df: pd.DataFrame, path: Path):
    """Write to a temp file and atomically rename."""
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(path)


def _save_cell_result(result: dict, cfg: ExperimentConfig, results_dir: Path):
    """Atomically write per-cell CSV + config JSON."""
    csv_path = results_dir / f"{result['condition']}_seed{result['seed']}.csv"
    cfg_path = results_dir / f"{result['condition']}_seed{result['seed']}.config.json"

    df = pd.DataFrame([result])
    _atomic_write_csv(df, csv_path)
    cfg.to_json(cfg_path)
    print(f"   💾 Saved → {csv_path.name} | {cfg_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSFER EVAL HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_transfer_eval(model, vis_loader, aud_loader, device, prefix: str, result: dict, seed: int):
    """Run evaluate_cross_modal_transfer with frozen model and populate result dict."""
    model.eval()
    with torch.no_grad():
        xfer = evaluate_cross_modal_transfer(
            model, vis_loader, aud_loader,
            device, num_classes=10, samples_per_class=100, random_state=seed,
        )
    result[f"{prefix}V2A"] = xfer["v2a"] * 100
    result[f"{prefix}A2V"] = xfer["a2v"] * 100
    result[f"{prefix}prototype_alignment"] = xfer["prototype_alignment"]
    result[f"{prefix}prototype_alignment_off_diag"] = xfer["prototype_alignment_off_diag"]
    result[f"{prefix}sil_visual"] = xfer["sil_visual"]
    result[f"{prefix}sil_audio"] = xfer["sil_audio"]
    result[f"{prefix}chance"] = xfer["chance"] * 100
    print(f"   {prefix}V2A: {xfer['v2a']*100:.2f}%  proto_align: {xfer['prototype_alignment']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ONE CELL — digit-only alignment ablation
# ═══════════════════════════════════════════════════════════════════════════════

def run_alignment_cell(condition: str, seed: int, cfg: ExperimentConfig,
                       data_loaders: dict, results_dir: Path, ckpt_dir: Path) -> dict:
    """
    Run one (condition, seed) cell of the digit-only ablation.
    Returns result dict; empty dict if skipped.
    """
    csv_path = results_dir / f"{condition}_seed{seed}.csv"
    if _is_complete_result(csv_path):
        print(f"SKIP {condition} seed {seed}")
        return {}

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*80}")
    print(f"  SEED {seed} | {condition}")
    print(f"{'='*80}")

    device = get_device(cfg)
    model = M7_ContinualAdaptiveModel(tasks=DIGIT_TASKS, cfg=cfg).to(device)
    buffer = EmbeddingReplayBuffer(
        feature_dim=cfg.shared_dim,
        samples_per_class=cfg.replay_samples_per_class,
        device=str(device),
    )
    trainer = ContinualLearner(model=model, buffer=buffer, cfg=cfg, device=device)

    # ── T1: N-MNIST ──────────────────────────────────────────────────────────
    task1 = "nmnist"
    print(f"\n📌 STEP 1: {task1.upper()}")
    t1_result = trainer.train_task(
        task1, task1, *data_loaders[task1]["train_val_test"],
        num_epochs=5, patience=3, lr=cfg.lr,
        use_replay=False, use_scl=cfg.use_contrastive,
        scl_weight=cfg.contrastive_weight, step_id=1,
    )
    t1_val_acc = t1_result["best_val"]
    print(f"   T1 Val: {t1_val_acc:.2f}%")

    # ── ALIGNMENT (B/C only) ─────────────────────────────────────────────────
    post_align_xfer = None
    if cfg.align_cross_modal and "paired" in data_loaders:
        print(f"\n🔗 ALIGNMENT: {condition}")
        align_history = trainer.align_cross_modal(
            paired_loader=data_loaders["paired"],
            num_epochs=5,
            lr=cfg.lr,
            use_scl=cfg.use_contrastive,
            scl_weight=cfg.contrastive_weight,
            align_weight=cfg.align_weight,
            step_id=1,
        )
        print(f"   Final losses → ce={align_history['ce_loss'][-1]:.3f} "
              f"xmscl={align_history['xmscl_loss'][-1]:.3f} "
              f"orth={align_history['orth_loss'][-1]:.4f}")

        # Post-alignment transfer eval
        print("\n📊 Transfer eval — post-alignment...")
        post_align_xfer = {}
        _run_transfer_eval(model, data_loaders["nmnist"]["test"], data_loaders["shd"]["test"],
                           device, "post_align_", post_align_xfer, seed)
    else:
        print("\n   (No alignment phase for condition A)")

    # ── T2: SHD ──────────────────────────────────────────────────────────────
    task2 = "shd"
    print(f"\n📌 STEP 2: {task2.upper()}")
    shd_zero_shot = trainer.evaluate(data_loaders[task2]["test"], task2, task2)
    chance_t2 = 100.0 / DIGIT_TASKS[task2]["num_classes"]
    print(f"   SHD zero-shot: {shd_zero_shot:.2f}% (chance={chance_t2:.2f}%)")

    t2_result = trainer.train_task(
        task2, task2, *data_loaders[task2]["train_val_test"],
        num_epochs=5, patience=3, lr=cfg.lr,
        use_replay=False, use_scl=cfg.use_contrastive,
        scl_weight=cfg.contrastive_weight, step_id=2,
    )
    shd_val_acc = t2_result["best_val"]
    print(f"   SHD Val: {shd_val_acc:.2f}%")

    # Post-T2 transfer eval
    print("\n📊 Transfer eval — post-T2...")
    post_t2_xfer = {}
    _run_transfer_eval(model, data_loaders["nmnist"]["test"], data_loaders["shd"]["test"],
                       device, "post_t2_", post_t2_xfer, seed)

    # ── Assemble result ──────────────────────────────────────────────────────
    result = {
        "condition": condition,
        "seed": seed,
        "T1_val_acc": t1_val_acc,
        "SHD_zero_shot_acc": shd_zero_shot,
        "SHD_val_acc": shd_val_acc,
    }

    if post_align_xfer is not None:
        result.update(post_align_xfer)
    else:
        for key in ["post_align_V2A", "post_align_A2V",
                    "post_align_prototype_alignment", "post_align_prototype_alignment_off_diag",
                    "post_align_sil_visual", "post_align_sil_audio", "post_align_chance"]:
            result[key] = np.nan

    result.update(post_t2_xfer)

    _save_cell_result(result, cfg, results_dir)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_digit_results(results_dir: Path) -> pd.DataFrame:
    """
    Ingest per-cell CSVs, compute mean ± std, 95% CI per metric per condition,
    plus paired Wilcoxon signed-rank tests (A-vs-B, A-vs-C).
    """
    csv_files = sorted(results_dir.glob("*_seed*.csv"))
    if not csv_files:
        print("No results to aggregate.")
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if all(col in df.columns for col in EXPECTED_COLUMNS):
                dfs.append(df)
            else:
                print(f"  Skipping {f.name}: incomplete columns")
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")
            continue

    if not dfs:
        print("No complete results to aggregate.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # ── Summary stats ────────────────────────────────────────────────────────
    rows = []
    metrics = [
        "T1_val_acc", "SHD_zero_shot_acc", "SHD_val_acc",
        "post_align_V2A", "post_align_A2V", "post_align_prototype_alignment",
        "post_t2_V2A", "post_t2_A2V", "post_t2_prototype_alignment",
        "post_t2_sil_visual", "post_t2_sil_audio",
    ]
    for metric in metrics:
        for cond in ["A_no_align", "B_full_512", "C_split_128"]:
            sub = df[df["condition"] == cond][metric].dropna()
            if len(sub) == 0:
                continue
            mean = sub.mean()
            std = sub.std(ddof=1)
            n = len(sub)
            ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            rows.append({
                "metric": metric,
                "condition": cond,
                "n": n,
                "mean": mean,
                "std": std,
                "ci95_lower": mean - ci95,
                "ci95_upper": mean + ci95,
            })

    summary = pd.DataFrame(rows)
    summary_path = results_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n💾 Summary saved to {summary_path}")

    # ── Wilcoxon: A-vs-B and A-vs-C ──────────────────────────────────────────
    wilcoxon_rows = []
    test_metrics = ["post_t2_V2A", "post_t2_A2V", "post_t2_prototype_alignment", "SHD_val_acc"]

    for metric in test_metrics:
        pivot = df.pivot_table(index="seed", columns="condition", values=metric, aggfunc="first")
        for test_cond in ["B_full_512", "C_split_128"]:
            if "A_no_align" not in pivot.columns or test_cond not in pivot.columns:
                continue
            paired = pivot[["A_no_align", test_cond]].dropna()
            if len(paired) <= 1:
                continue
            diff = paired[test_cond] - paired["A_no_align"]
            if (diff != 0).any():
                stat, pval = stats.wilcoxon(diff)
                wilcoxon_rows.append({
                    "metric": metric,
                    "comparison": f"{test_cond}_vs_A_no_align",
                    "wilcoxon_statistic": stat,
                    "wilcoxon_pvalue": pval,
                    "n_paired": len(paired),
                    "mean_diff": diff.mean(),
                })
            else:
                wilcoxon_rows.append({
                    "metric": metric,
                    "comparison": f"{test_cond}_vs_A_no_align",
                    "wilcoxon_statistic": np.nan,
                    "wilcoxon_pvalue": np.nan,
                    "n_paired": len(paired),
                    "mean_diff": 0.0,
                    "note": "all differences zero",
                })

    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    if not wilcoxon_df.empty:
        wilcoxon_path = results_dir / "wilcoxon.csv"
        wilcoxon_df.to_csv(wilcoxon_path, index=False)
        print(f"💾 Wilcoxon results saved to {wilcoxon_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL CL CONFIRMATION (set up, NOT run in main)
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_cl_confirmation(base_cfg: ExperimentConfig):
    """
    Run the full T1→T2→T3 sequence with forgetting/retention metrics
    for conditions A and C only, 3 seeds.
    Writes to results/cl_confirmation/.
    NOT called by main() — invoke explicitly if needed.
    """
    from m7_trimodal_run_provenance import build_data_loaders, TASKS, run_one_seed

    confirmation_dir = REPO_ROOT / "results" / "cl_confirmation"
    ckpt_dir = confirmation_dir / "checkpoints"
    confirmation_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456]
    conditions = [
        {"label": "A_no_align",   "align_cross_modal": False, "align_shared_dim": 512, "align_use_orthogonal": False},
        {"label": "C_shared_128", "align_cross_modal": True,  "align_shared_dim": 128, "align_use_orthogonal": True},
    ]

    print("=" * 80)
    print("🔬 FULL CL CONFIRMATION (T1→T2→T3) — Conditions A & C")
    print("=" * 80)

    print("Loading full datasets (includes DVS)...")
    data_loaders = build_data_loaders(base_cfg)
    print("Done.\n")

    all_results = []
    for seed in seeds:
        for cond in conditions:
            cfg = dataclasses.replace(
                base_cfg,
                seed=seed,
                align_cross_modal=cond["align_cross_modal"],
                align_shared_dim=cond["align_shared_dim"],
                align_use_orthogonal=cond["align_use_orthogonal"],
            )
            cfg.condition_label = cond["label"]
            result = run_one_seed(cfg, data_loaders, confirmation_dir, ckpt_dir)
            all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv(confirmation_dir / "cl_confirmation_results.csv", index=False)
    print(f"\n💾 Results saved to {confirmation_dir / 'cl_confirmation_results.csv'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    results_dir = REPO_ROOT / "results" / "cross_modal_alignment_digit"
    ckpt_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456, 789, 1011]

    base_cfg = ExperimentConfig(
        seed=42,
        task_order=["nmnist", "shd"],
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
        conditions=["A_no_align", "B_full_512", "C_split_128"],
        seeds=seeds,
        align_cross_modal=False,
        align_shared_dim=128,
        align_weight=0.5,
        align_temperature=0.1,
        align_use_orthogonal=False,
        align_orthogonal_weight=0.01,
    )

    print("=" * 80)
    print("🔗 DIGIT-ONLY CROSS-MODAL ALIGNMENT ABLATION")
    print("=" * 80)
    print(f"Device : {get_device(base_cfg)}")
    print(f"Seeds  : {seeds}")
    print(f"Conditions: A=no_align, B=full_512, C=shared_128")
    print(f"Results: {results_dir}")
    print()

    print("Loading digit datasets (NO DVS)...")
    data_loaders = build_digit_loaders(base_cfg)
    print("Done.\n")

    all_results = []
    conditions = [
        {"label": "A_no_align",   "align_cross_modal": False, "align_shared_dim": 512, "align_use_orthogonal": False},
        {"label": "B_full_512",   "align_cross_modal": True,  "align_shared_dim": 512, "align_use_orthogonal": False},
        {"label": "C_split_128",  "align_cross_modal": True,  "align_shared_dim": 128, "align_use_orthogonal": True},
    ]

    for seed in seeds:
        for cond in conditions:
            cfg = dataclasses.replace(
                base_cfg,
                seed=seed,
                align_cross_modal=cond["align_cross_modal"],
                align_shared_dim=cond["align_shared_dim"],
                align_use_orthogonal=cond["align_use_orthogonal"],
            )
            cfg.condition_label = cond["label"]

            res = run_alignment_cell(cond["label"], seed, cfg, data_loaders, results_dir, ckpt_dir)
            if res:
                all_results.append(res)

    if all_results:
        print("\n" + "=" * 80)
        print("📊 ALL RESULTS")
        print("=" * 80)
        df = pd.DataFrame(all_results)
        show_cols = [c for c in df.columns if "chance" not in c]
        print(df[show_cols].to_string(index=False))

        aggregate_digit_results(results_dir)

    base_cfg.to_json(results_dir / "master_config.json")
    print(f"\n💾 Master config saved to {results_dir / 'master_config.json'}")
    print("\n✅ Digit-only alignment ablation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digit-only cross-modal alignment ablation")
    parser.add_argument("--cell", nargs=2, metavar=("CONDITION", "SEED"),
                        help="Run a single cell, e.g. --cell A_no_align 42")
    parser.add_argument("--confirm", action="store_true",
                        help="Run full CL confirmation (A&C, 3 seeds)")
    args = parser.parse_args()

    if args.confirm:
        import run_cl_confirmation
        run_cl_confirmation.main()
    elif args.cell:
        condition, seed_str = args.cell
        seed = int(seed_str)

        conditions_map = {
            "A_no_align":  {"align_cross_modal": False, "align_shared_dim": 512, "align_use_orthogonal": False},
            "B_full_512":  {"align_cross_modal": True,  "align_shared_dim": 512, "align_use_orthogonal": False},
            "C_split_128": {"align_cross_modal": True,  "align_shared_dim": 128, "align_use_orthogonal": True},
        }
        if condition not in conditions_map:
            raise ValueError(f"Unknown condition: {condition}. Expected one of {list(conditions_map.keys())}")

        cond = conditions_map[condition]
        base_cfg = ExperimentConfig(
            seed=seed,
            task_order=["nmnist", "shd"],
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
            align_cross_modal=cond["align_cross_modal"],
            align_shared_dim=cond["align_shared_dim"],
            align_use_orthogonal=cond["align_use_orthogonal"],
            align_weight=0.5,
            align_temperature=0.1,
            align_orthogonal_weight=0.01,
        )
        base_cfg.condition_label = condition

        results_dir = REPO_ROOT / "results" / "cross_modal_alignment_digit"
        ckpt_dir = results_dir / "checkpoints"
        results_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        print("Loading digit datasets (NO DVS)...")
        data_loaders = build_digit_loaders(base_cfg)
        print("Done.\n")

        run_alignment_cell(condition, seed, base_cfg, data_loaders, results_dir, ckpt_dir)
    else:
        main()

#!/usr/bin/env python3
"""
CL Confirmation: Full T1→T2→T3 sequence for conditions A & C.
==============================================================

Verifies that cross-modal alignment (condition C) does not worsen
catastrophic forgetting compared to no alignment (condition A).

Protocol per cell:
  T1: N-MNIST
  [Alignment phase for C only]
  T2: SHD
  T3: DVS-Gesture
  Retention eval: T1/T2 after T3

Resumable: re-running skips completed cells.
Atomic writes: temp file + rename so incomplete writes are never seen.

Outputs:
  results/cl_confirmation/
      ├── {condition}_seed{seed}.csv
      ├── {condition}_seed{seed}.config.json
      ├── checkpoints/{condition}_seed{seed}/   (task checkpoints)
      ├── summary.csv          (mean±std per condition)
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
    TASKS,
    ExperimentConfig,
    M7_ContinualAdaptiveModel,
    ContinualLearner,
    EmbeddingReplayBuffer,
    get_device,
    nmnist_collate_fn,
    shd_collate_fn,
    dvs_collate_fn,
    random_init_baseline,
    print_task_sanity,
)

# Expected CSV columns for a complete CL confirmation result
EXPECTED_COLUMNS = [
    "condition", "seed",
    "T1_after_T1", "T2_after_T2", "T3_after_T3",
    "T1_after_T2", "T1_after_T3", "T2_after_T3",
    "T2_zero_shot", "T3_zero_shot",
    "T2_chance", "T3_chance",
    "T2_learning_gain", "T3_learning_gain",
    "T2_random_init_baseline", "T3_random_init_baseline",
    "T2_fwt", "T3_fwt",
    "forgetting_T1_T2", "forgetting_T1_T3", "forgetting_T2_T3",
    "total_forgetting_T1_T3", "total_forgetting_T2_T3",
]


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
#  DATA LOADERS (full tri-modal + paired)
# ═══════════════════════════════════════════════════════════════════════════════

def build_full_loaders(cfg: ExperimentConfig):
    """Build N-MNIST, SHD, DVS, and paired loaders."""
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
        visual_dataset=train_split,
        audio_dataset=shd_train_split,
        batch_size=30,
        samples_per_class=min(500, len(train_split) // 10, len(shd_train_split) // 10),
        num_workers=0,
        seed=42,
        visual_transform=_visual_transform,
        audio_transform=None,
    )

    print("  ✅ Full tri-modal datasets loaded (N-MNIST + SHD + DVS).")
    return {
        "nmnist": nmnist_loaders,
        "shd": shd_loaders,
        "dvs": dvs_loaders,
        "paired": paired_loader,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ONE CELL — full CL confirmation
# ═══════════════════════════════════════════════════════════════════════════════

def run_confirmation_cell(condition: str, seed: int, cfg: ExperimentConfig,
                          data_loaders: dict, results_dir: Path, ckpt_dir: Path) -> dict:
    """
    Run one (condition, seed) cell of the full CL confirmation.
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
    model = M7_ContinualAdaptiveModel(tasks=TASKS, cfg=cfg).to(device)
    buffer = EmbeddingReplayBuffer(
        feature_dim=cfg.shared_dim,
        samples_per_class=cfg.replay_samples_per_class,
        device=str(device),
    )
    trainer = ContinualLearner(model=model, buffer=buffer, cfg=cfg, device=device)

    # Condition-specific checkpoint subdir
    cell_ckpt_dir = ckpt_dir / f"{condition}_seed{seed}"
    cell_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute random-init baselines for FWT
    print("\n  🔧 Computing random-init baselines...")
    random_init_baselines = {}
    for task_name in cfg.task_order:
        ri = random_init_baseline(TASKS, task_name, data_loaders[task_name]["test"], cfg, device, n_seeds=3)
        random_init_baselines[task_name] = ri
        print(f"     {task_name}: {ri:.2f}%")

    # ═══════════════════════════════════════════════════════════════════════
    #  T1: N-MNIST
    # ═══════════════════════════════════════════════════════════════════════
    task1 = cfg.task_order[0]
    print(f"\n📌 STEP 1: {task1.upper()}")
    ckpt_t1 = cell_ckpt_dir / f"final_{task1}_{task1}.pt"
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
        cfg.to_json(cell_ckpt_dir / f"config_{task1}.json")

    acc_t1_after = trainer.evaluate(data_loaders[task1]["test"], task1, task1)
    print(f"   T1 Test: {acc_t1_after:.2f}%")

    # ═══════════════════════════════════════════════════════════════════════
    #  ALIGNMENT (C only)
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
        print(f"   Final losses → ce={align_history['ce_loss'][-1]:.3f} "
              f"xmscl={align_history['xmscl_loss'][-1]:.3f} "
              f"orth={align_history['orth_loss'][-1]:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    #  T2: SHD
    # ═══════════════════════════════════════════════════════════════════════
    task2 = cfg.task_order[1]
    print(f"\n📌 STEP 2: {task2.upper()}")

    acc_t2_zero_shot = trainer.evaluate(data_loaders[task2]["test"], task2, task2)
    chance_t2 = 100.0 / TASKS[task2]["num_classes"]
    print(f"   T2 zero-shot: {acc_t2_zero_shot:.2f}% (chance={chance_t2:.2f}%)")

    ckpt_t2 = cell_ckpt_dir / f"final_{task2}_{task2}.pt"
    if ckpt_t2.exists():
        print(f"   ⏩ Skipping T2 training (checkpoint found)")
        model.load_state_dict(torch.load(ckpt_t2, map_location=str(device))["model_state_dict"])
    else:
        trainer.train_task(
            task2, task2, *data_loaders[task2]["train_val_test"],
            num_epochs=15, patience=5, lr=cfg.lr,
            use_replay=False, use_scl=cfg.use_contrastive,
            scl_weight=cfg.contrastive_weight, step_id=2,
        )
        torch.save({"model_state_dict": model.state_dict()}, ckpt_t2)
        cfg.to_json(cell_ckpt_dir / f"config_{task2}.json")

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

    # ═══════════════════════════════════════════════════════════════════════
    #  T3: DVS
    # ═══════════════════════════════════════════════════════════════════════
    task3 = cfg.task_order[2]
    print(f"\n📌 STEP 3: {task3.upper()}")

    acc_t3_zero_shot = trainer.evaluate(data_loaders[task3]["test"], task3, task3)
    chance_t3 = 100.0 / TASKS[task3]["num_classes"]
    print(f"   T3 zero-shot: {acc_t3_zero_shot:.2f}% (chance={chance_t3:.2f}%)")

    ckpt_t3 = cell_ckpt_dir / f"final_{task3}_{task3}.pt"
    if ckpt_t3.exists():
        print(f"   ⏩ Skipping T3 training (checkpoint found)")
        model.load_state_dict(torch.load(ckpt_t3, map_location=str(device))["model_state_dict"])
    else:
        trainer.train_task(
            task3, task3, *data_loaders[task3]["train_val_test"],
            num_epochs=15, patience=5, lr=cfg.lr,
            use_replay=False, use_scl=cfg.use_contrastive,
            scl_weight=cfg.contrastive_weight, step_id=3,
        )
        torch.save({"model_state_dict": model.state_dict()}, ckpt_t3)
        cfg.to_json(cell_ckpt_dir / f"config_{task3}.json")

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
    forgetting_t1_t3 = acc_t1_after_t2 - acc_t1_after_t3  # incremental T2→T3
    forgetting_t2_t3 = acc_t2_after - acc_t2_after_t3
    total_forgetting_t1_t3 = acc_t1_after - acc_t1_after_t3  # total from peak
    total_forgetting_t2_t3 = acc_t2_after - acc_t2_after_t3  # total from peak

    result = {
        "condition": condition,
        "seed": seed,
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
        "total_forgetting_T1_T3": total_forgetting_t1_t3,
        "total_forgetting_T2_T3": total_forgetting_t2_t3,
    }

    _save_cell_result(result, cfg, results_dir)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_confirmation_results(results_dir: Path) -> pd.DataFrame:
    """
    Ingest per-cell CSVs, compute mean ± std per metric per condition.
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

    # Summary stats
    rows = []
    metrics = [
        "T1_after_T1", "T2_after_T2", "T3_after_T3",
        "T1_after_T2", "T1_after_T3", "T2_after_T3",
        "T2_zero_shot", "T3_zero_shot",
        "T2_learning_gain", "T3_learning_gain",
        "T2_fwt", "T3_fwt",
        "forgetting_T1_T2", "forgetting_T1_T3", "forgetting_T2_T3",
        "total_forgetting_T1_T3", "total_forgetting_T2_T3",
    ]
    for metric in metrics:
        for cond in ["A_no_align", "C_split_128"]:
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

    # Key retention/forgetting comparison table (for easy reading)
    print("\n" + "="*80)
    print("📊 RETENTION & FORGETTING COMPARISON")
    print("="*80)
    key_metrics = ["T1_after_T3", "T2_after_T3", "T3_after_T3",
                   "total_forgetting_T1_T3", "total_forgetting_T2_T3"]
    for metric in key_metrics:
        print(f"\n{metric}:")
        for cond in ["A_no_align", "C_split_128"]:
            sub = df[df["condition"] == cond][metric].dropna()
            if len(sub) > 0:
                print(f"  {cond:15s}: {sub.mean():.2f}% ± {sub.std(ddof=1):.2f}% (n={len(sub)})")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    results_dir = REPO_ROOT / "results" / "cl_confirmation"
    ckpt_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456]
    conditions = [
        {"label": "A_no_align",   "align_cross_modal": False, "align_shared_dim": 512, "align_use_orthogonal": False},
        {"label": "C_split_128",  "align_cross_modal": True,  "align_shared_dim": 128, "align_use_orthogonal": True},
    ]

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
        conditions=["A_no_align", "C_split_128"],
        seeds=seeds,
        align_cross_modal=False,
        align_shared_dim=128,
        align_weight=0.5,
        align_temperature=0.1,
        align_use_orthogonal=False,
        align_orthogonal_weight=0.01,
    )

    print("=" * 80)
    print("🔬 FULL CL CONFIRMATION (T1→T2→T3) — Conditions A & C")
    print("=" * 80)
    print(f"Device : {get_device(base_cfg)}")
    print(f"Seeds  : {seeds}")
    print(f"Conditions: A=no_align, C=shared_128+orthogonal")
    print(f"Results: {results_dir}")
    print()

    print("Loading full datasets (includes DVS)...")
    data_loaders = build_full_loaders(base_cfg)
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

            res = run_confirmation_cell(cond["label"], seed, cfg, data_loaders, results_dir, ckpt_dir)
            if res:
                all_results.append(res)

    if all_results:
        print("\n" + "=" * 80)
        print("📊 ALL RESULTS")
        print("=" * 80)
        df = pd.DataFrame(all_results)
        show_cols = [c for c in df.columns if "chance" not in c and "baseline" not in c]
        print(df[show_cols].to_string(index=False))

        aggregate_confirmation_results(results_dir)

    base_cfg.to_json(results_dir / "master_config.json")
    print(f"\n💾 Master config saved to {results_dir / 'master_config.json'}")
    print("\n✅ CL Confirmation complete!")


if __name__ == "__main__":
    main()

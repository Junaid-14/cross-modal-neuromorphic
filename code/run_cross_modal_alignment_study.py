#!/usr/bin/env python3
"""
Cross-Modal Alignment Study — Conditions A / B / C
==================================================

Condition A (baseline) : No cross-modal alignment
Condition B (full)     : Full 512-d shared space, SupCon on paired digits
Condition C (split)    : Shared=128-d + Private=384-d, with orthogonal penalty

Protocol:
  T1: N-MNIST → evaluate transfer (baseline)
  [Alignment phase for B/C only]
  T2: SHD       → evaluate transfer (post-alignment / post-T2)
  T3: DVS-Gesture

Transfer evaluation (Paper 1 engram protocol):
  - Linear-probe V2A / A2V on shared subspace
  - Prototype cosine alignment matrix
  - Silhouette scores per modality

Outputs:
  results/cross_modal_alignment/
      ├── config_seed{N}_{condition}.json   (per-seed provenance)
      ├── results_seed{N}_{condition}.csv    (per-seed metrics)
      ├── alignment_summary.csv              (mean ± std, 95% CI)
      ├── alignment_wilcoxon.csv             (paired Wilcoxon B-vs-A, C-vs-A)
      └── master_results.csv                 (all raw rows)
"""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats

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
    build_data_loaders,
    TASKS,
    random_init_baseline,
    print_task_sanity,
)
from utils.cross_modal_transfer import evaluate_cross_modal_transfer


# ═══════════════════════════════════════════════════════════════════════════════
#  PROVENANCE IO (isolated to results/cross_modal_alignment/)
# ═══════════════════════════════════════════════════════════════════════════════

def _save_seed_result(result: dict, cfg: ExperimentConfig, results_dir: Path):
    """Save one seed's result to per-seed CSV + config JSON; append to master CSV."""
    master_csv = results_dir / "master_results.csv"
    df = pd.DataFrame([result])
    if master_csv.exists():
        existing = pd.read_csv(master_csv)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(master_csv, index=False)

    seed_csv = results_dir / f"results_seed{result['seed']}_{result['condition']}.csv"
    df.to_csv(seed_csv, index=False)

    cfg_path = results_dir / f"config_seed{result['seed']}_{result['condition']}.json"
    cfg.to_json(cfg_path)

    print(f"   💾 Saved → {seed_csv.name} | {cfg_path.name}")


def _aggregate_alignment_results(results_dir: Path, metrics: List[str]) -> pd.DataFrame:
    """
    Compute mean ± std, 95% CI per condition, plus paired Wilcoxon
    signed-rank tests (B vs A, C vs A) using matched seeds.
    """
    master_csv = results_dir / "master_results.csv"
    if not master_csv.exists():
        print("No results to aggregate.")
        return pd.DataFrame()

    df = pd.read_csv(master_csv)
    df = df[df["T1_after_T1"].notna()].copy()

    # ── Summary stats ──────────────────────────────────────────────────────
    rows = []
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
    summary_path = results_dir / "alignment_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n💾 Alignment summary saved to {summary_path}")

    # ── Wilcoxon: paired per seed ──────────────────────────────────────────
    wilcoxon_rows = []
    seeds = sorted(df["seed"].unique())

    for metric in metrics:
        # Pivot so each seed is a row, conditions are columns
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
        wilcoxon_path = results_dir / "alignment_wilcoxon.csv"
        wilcoxon_df.to_csv(wilcoxon_path, index=False)
        print(f"💾 Wilcoxon results saved to {wilcoxon_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  ONE SEED — full A/B/C protocol
# ═══════════════════════════════════════════════════════════════════════════════

def _run_alignment_seed(
    cfg: ExperimentConfig,
    data_loaders: dict,
    results_dir: Path,
    ckpt_dir: Path,
) -> dict:
    """
    Run one seed for one alignment condition.
    Returns flat dict with all metrics for CSV serialization.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    condition = cfg.condition_label  # injected below

    print(f"\n{'='*80}")
    print(f"  SEED {cfg.seed} | {condition}")
    print(f"{'='*80}")

    device = get_device(cfg)
    model = M7_ContinualAdaptiveModel(tasks=TASKS, cfg=cfg).to(device)
    buffer = EmbeddingReplayBuffer(
        feature_dim=cfg.shared_dim,
        samples_per_class=cfg.replay_samples_per_class,
        device=str(device),
    )
    trainer = ContinualLearner(model=model, buffer=buffer, cfg=cfg, device=device)

    # Random-init baselines
    print("\n  🔧 Computing random-init baselines...")
    ri_baselines = {}
    for task_name in cfg.task_order:
        ri = random_init_baseline(TASKS, task_name, data_loaders[task_name]["test"], cfg, device, n_seeds=3)
        ri_baselines[task_name] = ri
        print(f"     {task_name}: {ri:.2f}%")

    # ── T1: N-MNIST ────────────────────────────────────────────────────────
    task1 = cfg.task_order[0]
    print(f"\n📌 STEP 1: {task1.upper()}")
    ckpt_t1 = ckpt_dir / f"{condition}_seed{cfg.seed}_T1.pt"
    if ckpt_t1.exists():
        print("   ⏩ Loading checkpoint")
        model.load_state_dict(torch.load(ckpt_t1, map_location=str(device))["model_state_dict"])
    else:
        trainer.train_task(
            task1, task1, *data_loaders[task1]["train_val_test"],
            num_epochs=5, patience=3, lr=cfg.lr,
            use_replay=False, use_scl=cfg.use_contrastive,
            scl_weight=cfg.contrastive_weight, step_id=1,
        )
        torch.save({"model_state_dict": model.state_dict()}, ckpt_t1)

    acc_t1_after = trainer.evaluate(data_loaders[task1]["test"], task1, task1)
    print(f"   T1 Test: {acc_t1_after:.2f}%")

    # Transfer evaluation AFTER T1 (baseline, before alignment)
    transfer_after_t1 = evaluate_cross_modal_transfer(
        model, data_loaders["nmnist"]["test"], data_loaders["shd"]["test"],
        device, num_classes=10, samples_per_class=100, random_state=cfg.seed,
    )
    print(f"   Transfer after T1 — V2A: {transfer_after_t1['v2a']*100:.2f}%")

    # Populate buffer for potential replay
    buffer.populate(model, data_loaders[task1]["train"], task1, task_id=1, label_offset=0)

    # ── ALIGNMENT (B/C only) ───────────────────────────────────────────────
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

        # Transfer evaluation AFTER alignment
        transfer_after_align = evaluate_cross_modal_transfer(
            model, data_loaders["nmnist"]["test"], data_loaders["shd"]["test"],
            device, num_classes=10, samples_per_class=100, random_state=cfg.seed,
        )
        print(f"   Transfer after ALIGN — V2A: {transfer_after_align['v2a']*100:.2f}%")
    else:
        # Condition A: no alignment — copy baseline metrics
        transfer_after_align = {
            k: (0.0 if k.startswith("sil_") else transfer_after_t1.get(k, 0.0))
            for k in ["v2a", "a2v", "prototype_alignment", "prototype_alignment_off_diag",
                      "sil_visual", "sil_audio", "chance"]
        }

    # ── T2: SHD ────────────────────────────────────────────────────────────
    task2 = cfg.task_order[1]
    print(f"\n📌 STEP 2: {task2.upper()}")

    acc_t2_zero_shot = trainer.evaluate(data_loaders[task2]["test"], task2, task2)
    chance_t2 = 100.0 / TASKS[task2]["num_classes"]
    print(f"   T2 zero-shot: {acc_t2_zero_shot:.2f}% (chance={chance_t2:.2f}%)")

    ckpt_t2 = ckpt_dir / f"{condition}_seed{cfg.seed}_T2.pt"
    if ckpt_t2.exists():
        print("   ⏩ Loading checkpoint")
        model.load_state_dict(torch.load(ckpt_t2, map_location=str(device))["model_state_dict"])
    else:
        trainer.train_task(
            task2, task2, *data_loaders[task2]["train_val_test"],
            num_epochs=5, patience=3, lr=cfg.lr,
            use_replay=cfg.use_replay,
            replay_task_ids=[1] if cfg.use_replay else None,
            use_scl=cfg.use_contrastive,
            scl_weight=cfg.contrastive_weight, step_id=2,
        )
        torch.save({"model_state_dict": model.state_dict()}, ckpt_t2)

    acc_t2_after = trainer.evaluate(data_loaders[task2]["test"], task2, task2)
    print(f"   T2 Test: {acc_t2_after:.2f}%")

    acc_t1_after_t2 = trainer.evaluate(data_loaders[task1]["test"], task1, task1)
    print(f"   T1 after T2: {acc_t1_after_t2:.2f}%")

    learning_gain_t2 = acc_t2_after - acc_t2_zero_shot
    fwt_t2 = acc_t2_zero_shot - ri_baselines[task2]
    print_task_sanity(
        task_name=task2, num_classes=TASKS[task2]["num_classes"], chance=chance_t2,
        zero_shot_acc=acc_t2_zero_shot, after_training_acc=acc_t2_after,
        learning_gain=learning_gain_t2, fwt=fwt_t2,
        random_init_baseline_acc=ri_baselines[task2],
    )

    buffer.populate(model, data_loaders[task2]["train"], task2, task_id=2, label_offset=0)

    # Transfer evaluation AFTER T2
    transfer_after_t2 = evaluate_cross_modal_transfer(
        model, data_loaders["nmnist"]["test"], data_loaders["shd"]["test"],
        device, num_classes=10, samples_per_class=100, random_state=cfg.seed,
    )
    print(f"   Transfer after T2 — V2A: {transfer_after_t2['v2a']*100:.2f}%")

    # ── T3: DVS-Gesture ────────────────────────────────────────────────────
    task3 = cfg.task_order[2]
    print(f"\n📌 STEP 3: {task3.upper()}")

    acc_t3_zero_shot = trainer.evaluate(data_loaders[task3]["test"], task3, task3)
    chance_t3 = 100.0 / TASKS[task3]["num_classes"]
    print(f"   T3 zero-shot: {acc_t3_zero_shot:.2f}% (chance={chance_t3:.2f}%)")

    ckpt_t3 = ckpt_dir / f"{condition}_seed{cfg.seed}_T3.pt"
    if ckpt_t3.exists():
        print("   ⏩ Loading checkpoint")
        model.load_state_dict(torch.load(ckpt_t3, map_location=str(device))["model_state_dict"])
    else:
        trainer.train_task(
            task3, task3, *data_loaders[task3]["train_val_test"],
            num_epochs=5, patience=3, lr=cfg.lr,
            use_replay=cfg.use_replay,
            replay_task_ids=[1, 2] if cfg.use_replay else None,
            use_scl=cfg.use_contrastive,
            scl_weight=cfg.contrastive_weight, step_id=3,
        )
        torch.save({"model_state_dict": model.state_dict()}, ckpt_t3)

    acc_t3_after = trainer.evaluate(data_loaders[task3]["test"], task3, task3)
    print(f"   T3 Test: {acc_t3_after:.2f}%")

    acc_t1_after_t3 = trainer.evaluate(data_loaders[task1]["test"], task1, task1)
    acc_t2_after_t3 = trainer.evaluate(data_loaders[task2]["test"], task2, task2)
    print(f"   T1 after T3: {acc_t1_after_t3:.2f}%")
    print(f"   T2 after T3: {acc_t2_after_t3:.2f}%")

    learning_gain_t3 = acc_t3_after - acc_t3_zero_shot
    fwt_t3 = acc_t3_zero_shot - ri_baselines[task3]
    print_task_sanity(
        task_name=task3, num_classes=TASKS[task3]["num_classes"], chance=chance_t3,
        zero_shot_acc=acc_t3_zero_shot, after_training_acc=acc_t3_after,
        learning_gain=learning_gain_t3, fwt=fwt_t3,
        random_init_baseline_acc=ri_baselines[task3],
    )

    # ── Flatten result ─────────────────────────────────────────────────────
    result = {
        "seed": cfg.seed,
        "condition": condition,
        # Standard CL
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
        "T2_fwt": fwt_t2,
        "T3_fwt": fwt_t3,
        "forgetting_T1_T2": acc_t1_after - acc_t1_after_t2,
        "forgetting_T1_T3": acc_t1_after_t2 - acc_t1_after_t3,
        "forgetting_T2_T3": acc_t2_after - acc_t2_after_t3,
        # Transfer after T1 (baseline)
        "T1_v2a": transfer_after_t1["v2a"] * 100,
        "T1_a2v": transfer_after_t1["a2v"] * 100,
        "T1_proto_align": transfer_after_t1["prototype_alignment"],
        "T1_proto_offdiag": transfer_after_t1["prototype_alignment_off_diag"],
        "T1_sil_vis": transfer_after_t1["sil_visual"],
        "T1_sil_aud": transfer_after_t1["sil_audio"],
        # Transfer after alignment
        "ALIGN_v2a": transfer_after_align["v2a"] * 100,
        "ALIGN_a2v": transfer_after_align["a2v"] * 100,
        "ALIGN_proto_align": transfer_after_align["prototype_alignment"],
        "ALIGN_proto_offdiag": transfer_after_align["prototype_alignment_off_diag"],
        "ALIGN_sil_vis": transfer_after_align["sil_visual"],
        "ALIGN_sil_aud": transfer_after_align["sil_audio"],
        # Transfer after T2
        "T2_v2a": transfer_after_t2["v2a"] * 100,
        "T2_a2v": transfer_after_t2["a2v"] * 100,
        "T2_proto_align": transfer_after_t2["prototype_alignment"],
        "T2_proto_offdiag": transfer_after_t2["prototype_alignment_off_diag"],
        "T2_sil_vis": transfer_after_t2["sil_visual"],
        "T2_sil_aud": transfer_after_t2["sil_audio"],
    }

    _save_seed_result(result, cfg, results_dir)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    results_dir = REPO_ROOT / "results" / "cross_modal_alignment"
    ckpt_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456, 789, 101112]

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
        conditions=["no_replay", "replay"],
        seeds=seeds,
        align_cross_modal=False,
        align_shared_dim=128,
        align_weight=0.5,
        align_temperature=0.1,
        align_use_orthogonal=False,
        align_orthogonal_weight=0.01,
    )

    print("=" * 80)
    print("🔗 CROSS-MODAL ALIGNMENT STUDY — Conditions A / B / C")
    print("=" * 80)
    print(f"Device : {get_device(base_cfg)}")
    print(f"Seeds  : {seeds}")
    print(f"Results: {results_dir}")
    print()

    print("Loading datasets...")
    data_loaders = build_data_loaders(base_cfg)
    print("Done.\n")

    all_results = []

    for seed in seeds:
        # ── Condition A: No alignment ──────────────────────────────────────
        cfg_a = dataclasses.replace(
            base_cfg,
            seed=seed,
            align_cross_modal=False,
            align_shared_dim=512,
            align_use_orthogonal=False,
        )
        cfg_a.condition_label = "A_no_align"
        res_a = _run_alignment_seed(cfg_a, data_loaders, results_dir, ckpt_dir)
        all_results.append(res_a)

        # ── Condition B: Full 512-d SupCon ─────────────────────────────────
        cfg_b = dataclasses.replace(
            base_cfg,
            seed=seed,
            align_cross_modal=True,
            align_shared_dim=512,
            align_use_orthogonal=False,
        )
        cfg_b.condition_label = "B_full_512"
        res_b = _run_alignment_seed(cfg_b, data_loaders, results_dir, ckpt_dir)
        all_results.append(res_b)

        # ── Condition C: Shared/private split ──────────────────────────────
        cfg_c = dataclasses.replace(
            base_cfg,
            seed=seed,
            align_cross_modal=True,
            align_shared_dim=128,
            align_use_orthogonal=True,
        )
        cfg_c.condition_label = "C_split_128"
        res_c = _run_alignment_seed(cfg_c, data_loaders, results_dir, ckpt_dir)
        all_results.append(res_c)

    # ── Summary print ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 ALL RESULTS")
    print("=" * 80)
    df = pd.DataFrame(all_results)
    # Hide chance/baseline columns for brevity
    show_cols = [c for c in df.columns if c not in ["T2_chance", "T3_chance"]]
    print(df[show_cols].to_string(index=False))

    # ── Aggregation ────────────────────────────────────────────────────────
    metrics = [
        "T1_after_T1", "T2_after_T2", "T3_after_T3",
        "T1_after_T2", "T1_after_T3", "T2_after_T3",
        "T2_zero_shot", "T3_zero_shot",
        "T2_learning_gain", "T3_learning_gain",
        "T2_fwt", "T3_fwt",
        "forgetting_T1_T2", "forgetting_T1_T3", "forgetting_T2_T3",
        # Transfer metrics
        "T1_v2a", "T1_a2v", "T1_proto_align", "T1_sil_vis", "T1_sil_aud",
        "ALIGN_v2a", "ALIGN_a2v", "ALIGN_proto_align", "ALIGN_sil_vis", "ALIGN_sil_aud",
        "T2_v2a", "T2_a2v", "T2_proto_align", "T2_sil_vis", "T2_sil_aud",
    ]
    _aggregate_alignment_results(results_dir, metrics)

    # Save master config
    base_cfg.to_json(results_dir / "master_config.json")
    print(f"\n💾 Master config saved to {results_dir / 'master_config.json'}")
    print("\n✅ Cross-modal alignment study complete!")


if __name__ == "__main__":
    main()

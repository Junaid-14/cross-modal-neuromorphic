#!/usr/bin/env python3
"""
CL Rehearsal Run: Unfrozen backbone + raw-input rehearsal.
============================================================

Tests whether rehearsing old tasks through the full (unfrozen) pipeline
can recover cross-modal transfer WITHOUT catastrophic forgetting.

Config: C_split_128, seed 42 only.
Sequence: T1 (N-MNIST) -> align -> T2 (SHD) -> T3 (DVS).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tonic
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT))

from m7_trimodal_run_provenance import (
    TASKS,
    ExperimentConfig,
    M7_ContinualAdaptiveModel,
    ContinualLearner,
    EmbeddingReplayBuffer,
    RawInputReplayBuffer,
    get_device,
    nmnist_collate_fn,
    shd_collate_fn,
    dvs_collate_fn,
    random_init_baseline,
    print_task_sanity,
)
from utils.cross_modal_transfer import evaluate_cross_modal_transfer


def build_full_loaders(cfg: ExperimentConfig):
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

    # Paired cross-modal loader
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

    print("  ✅ Full tri-modal datasets loaded.")
    return {
        "nmnist": nmnist_loaders,
        "shd": shd_loaders,
        "dvs": dvs_loaders,
        "paired": paired_loader,
    }


def run_rehearsal_cell(condition: str, seed: int, cfg: ExperimentConfig,
                       data_loaders: dict, results_dir: Path, ckpt_dir: Path,
                       raw_replay_weight: float = 1.0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*80}")
    print(f"  SEED {seed} | {condition} | raw_replay_weight={raw_replay_weight}")
    print(f"{'='*80}")

    device = get_device(cfg)
    model = M7_ContinualAdaptiveModel(tasks=TASKS, cfg=cfg).to(device)
    buffer = EmbeddingReplayBuffer(
        feature_dim=cfg.shared_dim,
        samples_per_class=cfg.replay_samples_per_class,
        device=str(device),
    )
    trainer = ContinualLearner(model=model, buffer=buffer, cfg=cfg, device=device)
    raw_buffer = RawInputReplayBuffer(samples_per_class=200)

    cell_ckpt_dir = ckpt_dir / f"{condition}_seed{seed}"
    cell_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Random-init baselines
    print("\n  🔧 Computing random-init baselines...")
    random_init_baselines = {}
    for task_name in cfg.task_order:
        ri = random_init_baseline(TASKS, task_name, data_loaders[task_name]["test"], cfg, device, n_seeds=3)
        random_init_baselines[task_name] = ri
        print(f"     {task_name}: {ri:.2f}%")

    # T1
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
            freeze_old=False,
        )
        torch.save({"model_state_dict": model.state_dict()}, ckpt_t1)
        cfg.to_json(cell_ckpt_dir / f"config_{task1}.json")

    acc_t1_after = trainer.evaluate(data_loaders[task1]["test"], task1, task1)
    print(f"   T1 Test: {acc_t1_after:.2f}%")

    # Populate raw buffer with T1 samples
    print("\n📦 Populating raw-input rehearsal buffer...")
    raw_buffer.populate(data_loaders[task1]["train"], task1, max_batches=100)

    # Alignment (C only) with rehearsal
    if cfg.align_cross_modal and "paired" in data_loaders:
        print("\n🔗 ALIGNMENT: cross-modal SupCon + T1 rehearsal")
        align_history = trainer.align_cross_modal(
            paired_loader=data_loaders["paired"],
            num_epochs=5,
            lr=cfg.lr,
            use_scl=cfg.use_contrastive,
            scl_weight=cfg.contrastive_weight,
            align_weight=cfg.align_weight,
            step_id=1,
            freeze_old=False,
            raw_replay_buffer=raw_buffer,
            raw_replay_weight=raw_replay_weight,
        )
        print(f"   Final losses → ce={align_history['ce_loss'][-1]:.3f} "
              f"xmscl={align_history['xmscl_loss'][-1]:.3f} "
              f"orth={align_history['orth_loss'][-1]:.4f}")

    # T2 with rehearsal
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
            freeze_old=False,
            raw_replay_buffer=raw_buffer,
            raw_replay_weight=raw_replay_weight,
        )
        torch.save({"model_state_dict": model.state_dict()}, ckpt_t2)
        cfg.to_json(cell_ckpt_dir / f"config_{task2}.json")

    acc_t2_after = trainer.evaluate(data_loaders[task2]["test"], task2, task2)
    print(f"   T2 Test: {acc_t2_after:.2f}%")

    acc_t1_after_t2 = trainer.evaluate(data_loaders[task1]["test"], task1, task1)
    print(f"   T1 after T2: {acc_t1_after_t2:.2f}%")

    learning_gain_t2 = acc_t2_after - acc_t2_zero_shot
    fwt_t2 = acc_t2_zero_shot - random_init_baselines[task2]
    print_task_sanity(
        task_name=task2, num_classes=TASKS[task2]["num_classes"],
        chance=chance_t2, zero_shot_acc=acc_t2_zero_shot,
        after_training_acc=acc_t2_after, learning_gain=learning_gain_t2,
        fwt=fwt_t2, random_init_baseline_acc=random_init_baselines[task2],
    )

    # Add T2 to raw buffer
    raw_buffer.populate(data_loaders[task2]["train"], task2, max_batches=100)

    # T3 with rehearsal
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
            freeze_old=False,
            raw_replay_buffer=raw_buffer,
            raw_replay_weight=raw_replay_weight,
        )
        torch.save({"model_state_dict": model.state_dict()}, ckpt_t3)
        cfg.to_json(cell_ckpt_dir / f"config_{task3}.json")

    acc_t3_after = trainer.evaluate(data_loaders[task3]["test"], task3, task3)
    print(f"   T3 Test: {acc_t3_after:.2f}%")

    acc_t1_after_t3 = trainer.evaluate(data_loaders[task1]["test"], task1, task1)
    acc_t2_after_t3 = trainer.evaluate(data_loaders[task2]["test"], task2, task2)
    print(f"   T1 after T3: {acc_t1_after_t3:.2f}%")
    print(f"   T2 after T3: {acc_t2_after_t3:.2f}%")

    learning_gain_t3 = acc_t3_after - acc_t3_zero_shot
    fwt_t3 = acc_t3_zero_shot - random_init_baselines[task3]
    print_task_sanity(
        task_name=task3, num_classes=TASKS[task3]["num_classes"],
        chance=chance_t3, zero_shot_acc=acc_t3_zero_shot,
        after_training_acc=acc_t3_after, learning_gain=learning_gain_t3,
        fwt=fwt_t3, random_init_baseline_acc=random_init_baselines[task3],
    )

    # Sanity: confirm nmnist backbone DID drift
    print("\n🔍 Sanity check: nmnist backbone drift (post-T1 vs post-T3)")
    sd_t1 = torch.load(ckpt_t1, map_location="cpu")["model_state_dict"]
    sd_t3 = torch.load(ckpt_t3, map_location="cpu")["model_state_dict"]
    total_l2 = 0.0
    n_params = 0
    for k in sd_t1:
        if k.startswith("backbones.nmnist") and sd_t1[k].is_floating_point():
            delta = (sd_t1[k] - sd_t3[k]).norm().item()
            total_l2 += delta
            n_params += 1
    print(f"   Total L2 drift: {total_l2:.4f} across {n_params} param tensors")
    if total_l2 < 1e-6:
        print("   ⚠️  WARNING: backbone did NOT drift — freezing may still be active!")
    else:
        print("   ✅ Backbone drifted as expected (unfrozen).")

    forgetting_t1_t2 = acc_t1_after - acc_t1_after_t2
    forgetting_t1_t3 = acc_t1_after_t2 - acc_t1_after_t3
    forgetting_t2_t3 = acc_t2_after - acc_t2_after_t3
    total_forgetting_t1_t3 = acc_t1_after - acc_t1_after_t3
    total_forgetting_t2_t3 = acc_t2_after - acc_t2_after_t3

    result = {
        "condition": condition, "seed": seed,
        "T1_after_T1": acc_t1_after, "T2_after_T2": acc_t2_after, "T3_after_T3": acc_t3_after,
        "T1_after_T2": acc_t1_after_t2, "T1_after_T3": acc_t1_after_t3, "T2_after_T3": acc_t2_after_t3,
        "T2_zero_shot": acc_t2_zero_shot, "T3_zero_shot": acc_t3_zero_shot,
        "T2_chance": chance_t2, "T3_chance": chance_t3,
        "T2_learning_gain": learning_gain_t2, "T3_learning_gain": learning_gain_t3,
        "T2_random_init_baseline": random_init_baselines[task2],
        "T3_random_init_baseline": random_init_baselines[task3],
        "T2_fwt": fwt_t2, "T3_fwt": fwt_t3,
        "forgetting_T1_T2": forgetting_t1_t2,
        "forgetting_T1_T3": forgetting_t1_t3,
        "forgetting_T2_T3": forgetting_t2_t3,
        "total_forgetting_T1_T3": total_forgetting_t1_t3,
        "total_forgetting_T2_T3": total_forgetting_t2_t3,
    }

    csv_path = results_dir / f"{condition}_seed{seed}.csv"
    cfg_path = results_dir / f"{condition}_seed{seed}.config.json"
    df = pd.DataFrame([result])
    tmp = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(csv_path)
    cfg.to_json(cfg_path)
    print(f"   💾 Saved → {csv_path.name} | {cfg_path.name}")
    return result


def main():
    results_dir = REPO_ROOT / "results" / "cl_rehearsal"
    ckpt_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw_replay_weight = 1.0

    base_cfg = ExperimentConfig(
        seed=42,
        task_order=["nmnist", "shd", "dvs"],
        lr=1e-3, weight_decay=1e-4, batch_size=32,
        max_epochs=50, patience=12, gradient_clip=1.0,
        beta=0.9, dropout=0.2, hidden_dim=512, shared_dim=512,
        num_patterns=100, hopfield_temperature=1.0, num_gru_layers=2,
        num_workers=0, time_steps=25, pin_memory=True,
        use_contrastive=True, contrastive_temperature=0.07, contrastive_weight=0.1,
        use_replay=False, replay_batch=32, replay_weight=0.3, replay_samples_per_class=50,
        device="auto",
        align_cross_modal=True, align_shared_dim=128, align_weight=0.5,
        align_temperature=0.1, align_use_orthogonal=True, align_orthogonal_weight=0.01,
    )

    print("=" * 80)
    print("🔬 CL REHEARSAL — Unfrozen backbone + raw-input rehearsal")
    print("=" * 80)
    print(f"Device : {get_device(base_cfg)}")
    print(f"Replay weight: {raw_replay_weight}")
    print(f"Results: {results_dir}")
    print()

    print("Loading full datasets...")
    data_loaders = build_full_loaders(base_cfg)
    print("Done.\n")

    res = run_rehearsal_cell("C_rehearsal", 42, base_cfg, data_loaders, results_dir, ckpt_dir,
                             raw_replay_weight=raw_replay_weight)

    # Transfer eval on final checkpoint
    print("\n" + "=" * 80)
    print("📊 TRANSFER EVAL (final checkpoint)")
    print("=" * 80)
    device = get_device(base_cfg)
    model = M7_ContinualAdaptiveModel(tasks=TASKS, cfg=base_cfg).to(device)
    ckpt_t3 = ckpt_dir / "C_rehearsal_seed42" / "final_dvs_dvs.pt"
    model.load_state_dict(torch.load(ckpt_t3, map_location=str(device))["model_state_dict"])
    model.eval()

    xfer = evaluate_cross_modal_transfer(
        model, data_loaders["nmnist"]["test"], data_loaders["shd"]["test"],
        device, num_classes=10, samples_per_class=100, random_state=42,
    )
    print(f"  V2A:                         {xfer['v2a']*100:.2f}%")
    print(f"  A2V:                         {xfer['a2v']*100:.2f}%")
    print(f"  Prototype alignment (diag):  {xfer['prototype_alignment']:.4f}")
    print(f"  Prototype alignment (off):   {xfer['prototype_alignment_off_diag']:.4f}")
    print(f"  Silhouette (visual):         {xfer['sil_visual']:.4f}")
    print(f"  Silhouette (audio):          {xfer['sil_audio']:.4f}")
    print(f"  Chance:                      {xfer['chance']*100:.2f}%")

    # Summary
    print("\n" + "=" * 80)
    print("📋 COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Metric':<30s} {'C_rehearsal':<15s} {'A_no_align':<15s} {'C_unfrozen':<15s} {'C_frozen':<15s}")
    print("-" * 90)
    print(f"{'T1 after T1':<30s} {res['T1_after_T1']:>14.2f}% {97.21:>14.2f}% {97.50:>14.2f}% {97.50:>14.2f}%")
    print(f"{'T1 after T2':<30s} {res['T1_after_T2']:>14.2f}% {97.28:>14.2f}% {54.85:>14.2f}% {15.33:>14.2f}%")
    print(f"{'T1 after T3':<30s} {res['T1_after_T3']:>14.2f}% {97.29:>14.2f}% {60.09:>14.2f}% {13.30:>14.2f}%")
    print(f"{'T2 after T2':<30s} {res['T2_after_T2']:>14.2f}% {79.64:>14.2f}% {75.09:>14.2f}% {81.67:>14.2f}%")
    print(f"{'T2 after T3':<30s} {res['T2_after_T3']:>14.2f}% {77.78:>14.2f}% {76.28:>14.2f}% {80.57:>14.2f}%")
    print(f"{'T3 after T3':<30s} {res['T3_after_T3']:>14.2f}% {78.41:>14.2f}% {75.38:>14.2f}% {82.20:>14.2f}%")
    print(f"{'Forgetting T1->T3':<30s} {res['total_forgetting_T1_T3']:>14.2f}% {-0.08:>14.2f}% {37.41:>14.2f}% {84.20:>14.2f}%")
    print(f"{'Forgetting T2->T3':<30s} {res['total_forgetting_T2_T3']:>14.2f}% {1.86:>14.2f}% {-1.19:>14.2f}% {1.10:>14.2f}%")
    print(f"{'V2A transfer':<30s} {xfer['v2a']*100:>14.2f}% {'—':>14s} {36.92:>14.2f}% {14.80:>14.2f}%")
    print(f"{'Prototype alignment':<30s} {xfer['prototype_alignment']:>14.4f} {'—':>14s} {0.59:>14.4f} {0.21:>14.4f}")

    base_cfg.to_json(results_dir / "master_config.json")
    print(f"\n💾 Master config saved to {results_dir / 'master_config.json'}")
    print("\n✅ Rehearsal run complete!")


if __name__ == "__main__":
    main()

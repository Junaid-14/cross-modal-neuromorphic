#!/usr/bin/env python3
"""
Phase 1 Sanity Check: Pinpoint when T1 collapses for C_split_128 seed 42.

Checks:
1. T1 accuracy at post-T1, post-T2, post-T3 checkpoints.
2. nmnist backbone + head weights byte-identical across checkpoints (frozen check).
3. Gate outputs sanity (no NaN, not all-zero, not all-identical).
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT))

from m7_trimodal_run_provenance import (
    ExperimentConfig,
    M7_ContinualAdaptiveModel,
    TASKS,
    get_device,
    nmnist_collate_fn,
)
from torch.utils.data import DataLoader, random_split
import tonic

CKPT_DIR = REPO_ROOT / "results" / "cl_confirmation" / "checkpoints" / "C_split_128_seed42"

def build_nmnist_test_loader(cfg):
    data_dir = Path("./ICONS_M7/datasets")
    sensor_size = tonic.datasets.NMNIST.sensor_size
    nmnist_transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])
    nmnist_test = tonic.datasets.NMNIST(save_to=str(data_dir), train=False, transform=nmnist_transform)
    return DataLoader(nmnist_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=nmnist_collate_fn)


def eval_t1(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            logits, _, _ = model(data, "nmnist", "nmnist")
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def compare_state_dicts(sd1, sd2, prefix):
    """Compare two state dicts for keys matching prefix. Returns list of changed keys."""
    changed = []
    for k in sd1:
        if not k.startswith(prefix):
            continue
        if k not in sd2:
            changed.append((k, "missing in sd2"))
            continue
        v1 = sd1[k]
        v2 = sd2[k]
        if not torch.equal(v1, v2):
            diff = (v1 - v2).abs().max().item()
            changed.append((k, f"max_diff={diff:.6e}"))
    return changed


def check_gate_sanity(model, loader, device):
    """Check gate outputs for T1 inputs: no NaN, not all-zero, not all-identical."""
    model.eval()
    all_gates = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            _, gate, _ = model(data, "nmnist", "nmnist")
            all_gates.append(gate.cpu())
    gates = torch.cat(all_gates, dim=0)
    results = {
        "has_nan": bool(torch.isnan(gates).any()),
        "all_zero": bool((gates == 0).all()),
        "all_identical": bool((gates.std(dim=0) < 1e-6).all()),
        "mean_gate0": float(gates[:, 0].mean()),
        "mean_gate1": float(gates[:, 1].mean()),
        "std_gate0": float(gates[:, 0].std()),
        "std_gate1": float(gates[:, 1].std()),
    }
    return results


def main():
    # Load config from post-T1 checkpoint dir
    config_path = CKPT_DIR / "config_nmnist.json"
    with open(config_path) as f:
        cfg_dict = json.load(f)
    cfg = ExperimentConfig(**cfg_dict)

    device = get_device(cfg)
    print(f"Device: {device}")
    print(f"Checkpoints: {CKPT_DIR}")
    print()

    # Build test loader once
    print("Building N-MNIST test loader...")
    test_loader = build_nmnist_test_loader(cfg)
    print(f"  Test samples: {len(test_loader.dataset)}\n")

    checkpoints = {
        "post_T1": CKPT_DIR / "final_nmnist_nmnist.pt",
        "post_T2": CKPT_DIR / "final_shd_shd.pt",
        "post_T3": CKPT_DIR / "final_dvs_dvs.pt",
    }

    # Evaluate T1 from each checkpoint
    print("=" * 70)
    print("1. T1 ACCURACY PER CHECKPOINT")
    print("=" * 70)
    t1_accs = {}
    for stage, path in checkpoints.items():
        model = M7_ContinualAdaptiveModel(tasks=TASKS, cfg=cfg).to(device)
        sd = torch.load(path, map_location=str(device))["model_state_dict"]
        model.load_state_dict(sd)
        acc = eval_t1(model, test_loader, device)
        t1_accs[stage] = acc
        print(f"  {stage:12s}: {acc:.2f}%")
    print()

    # Compare nmnist backbone + head weights
    print("=" * 70)
    print("2. FROZEN PARAM CHECK: nmnist backbone + head")
    print("=" * 70)
    sd_t1 = torch.load(checkpoints["post_T1"], map_location="cpu")["model_state_dict"]
    sd_t3 = torch.load(checkpoints["post_T3"], map_location="cpu")["model_state_dict"]

    changed_backbone = compare_state_dicts(sd_t1, sd_t3, "backbones.nmnist")
    changed_head = compare_state_dicts(sd_t1, sd_t3, "heads.nmnist")

    if not changed_backbone and not changed_head:
        print("  ✅ nmnist backbone and head are BYTE-IDENTICAL between post-T1 and post-T3.")
        print("     → T1 collapse is NOT from nmnist params changing.")
        print("     → GENUINE interference through shared components.")
    else:
        print("  ❌ nmnist params CHANGED between post-T1 and post-T3!")
        for k, reason in changed_backbone + changed_head:
            print(f"     {k}: {reason}")
        print("     → This is a PLUMBING BUG: gradients leaked into frozen params.")
    print()

    # Check gate sanity on the collapsed model (post-T3)
    print("=" * 70)
    print("3. GATE OUTPUT SANITY CHECK (post-T3 checkpoint)")
    print("=" * 70)
    model = M7_ContinualAdaptiveModel(tasks=TASKS, cfg=cfg).to(device)
    model.load_state_dict(torch.load(checkpoints["post_T3"], map_location=str(device))["model_state_dict"])
    gate_info = check_gate_sanity(model, test_loader, device)
    for k, v in gate_info.items():
        print(f"  {k:20s}: {v}")
    if gate_info["has_nan"]:
        print("  ⚠️  Gate outputs contain NaN!")
    elif gate_info["all_zero"]:
        print("  ⚠️  Gate outputs are all zero!")
    elif gate_info["all_identical"]:
        print("  ⚠️  Gate outputs are all identical!")
    else:
        print("  ✅ Gate outputs look healthy.")
    print()

    # Summary
    print("=" * 70)
    print("PHASE 1 VERDICT")
    print("=" * 70)
    t1_post_t1 = t1_accs["post_T1"]
    t1_post_t2 = t1_accs["post_T2"]
    t1_post_t3 = t1_accs["post_T3"]
    print(f"  T1 accuracy trajectory:")
    print(f"    post-T1:  {t1_post_t1:.2f}%")
    print(f"    post-T2:  {t1_post_t2:.2f}%  (Δ = {t1_post_t2 - t1_post_t1:+.2f}%)")
    print(f"    post-T3:  {t1_post_t3:.2f}%  (Δ = {t1_post_t3 - t1_post_t1:+.2f}%)")
    print()

    if not changed_backbone and not changed_head:
        print("  VERDICT: GENUINE INTERFERENCE")
        print("  The nmnist backbone and head are frozen correctly.")
        print("  T1 collapse happens because shared components (proj_shared, proj_private,")
        print("  adaptive_gate, hopfield, hgrn) are overwritten during T2/T3 training.")
    else:
        print("  VERDICT: PLUMBING BUG")
        print("  The nmnist backbone or head changed — gradients leaked into 'frozen' params.")


if __name__ == "__main__":
    main()

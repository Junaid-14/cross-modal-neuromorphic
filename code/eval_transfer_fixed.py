#!/usr/bin/env python3
"""
Transfer eval on fixed C_split_128 seed 42 checkpoints.
No training — eval only.
"""

import sys
from pathlib import Path
import json
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
    get_device,
    nmnist_collate_fn,
    shd_collate_fn,
)
from utils.cross_modal_transfer import evaluate_cross_modal_transfer

CKPT_DIR = REPO_ROOT / "results" / "cl_confirmation_fixed" / "checkpoints" / "C_split_128_seed42"

def build_nmnist_test_loader(cfg):
    data_dir = Path("./ICONS_M7/datasets")
    sensor_size = tonic.datasets.NMNIST.sensor_size
    nmnist_transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])
    nmnist_test = tonic.datasets.NMNIST(save_to=str(data_dir), train=False, transform=nmnist_transform)
    return DataLoader(nmnist_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=nmnist_collate_fn)

def build_shd_test_loader(cfg):
    data_dir = Path("./ICONS_M7/datasets")
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
    return DataLoader(shd_test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn)

def main():
    config_path = CKPT_DIR / "config_nmnist.json"
    with open(config_path) as f:
        cfg = ExperimentConfig(**json.load(f))
    device = get_device(cfg)

    print("Building test loaders...")
    nmnist_test = build_nmnist_test_loader(cfg)
    shd_test = build_shd_test_loader(cfg)
    print(f"  N-MNIST test: {len(nmnist_test.dataset)} samples")
    print(f"  SHD test: {len(shd_test.dataset)} samples")
    print()

    checkpoints = {
        "post_T2": CKPT_DIR / "final_shd_shd.pt",
        "post_T3": CKPT_DIR / "final_dvs_dvs.pt",
    }

    results = []
    for stage, path in checkpoints.items():
        if not path.exists():
            print(f"Skipping {stage}: checkpoint not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {stage} ({path.name})")
        print(f"{'='*60}")

        model = M7_ContinualAdaptiveModel(tasks=TASKS, cfg=cfg).to(device)
        sd = torch.load(path, map_location=str(device))["model_state_dict"]
        model.load_state_dict(sd)
        model.eval()

        xfer = evaluate_cross_modal_transfer(
            model, nmnist_test, shd_test,
            device, num_classes=10, samples_per_class=100, random_state=42,
        )

        print(f"  V2A:                         {xfer['v2a']*100:.2f}%")
        print(f"  A2V:                         {xfer['a2v']*100:.2f}%")
        print(f"  Prototype alignment (diag):  {xfer['prototype_alignment']:.4f}")
        print(f"  Prototype alignment (off):   {xfer['prototype_alignment_off_diag']:.4f}")
        print(f"  Silhouette (visual):         {xfer['sil_visual']:.4f}")
        print(f"  Silhouette (audio):          {xfer['sil_audio']:.4f}")
        print(f"  Chance:                      {xfer['chance']*100:.2f}%")

        results.append({
            "stage": stage,
            "v2a": xfer["v2a"] * 100,
            "a2v": xfer["a2v"] * 100,
            "prototype_alignment": xfer["prototype_alignment"],
            "prototype_alignment_off_diag": xfer["prototype_alignment_off_diag"],
            "sil_visual": xfer["sil_visual"],
            "sil_audio": xfer["sil_audio"],
            "chance": xfer["chance"] * 100,
        })

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    post_t2_v2a = df[df["stage"]=="post_T2"]["v2a"].values[0] if "post_T2" in df["stage"].values else 0
    if post_t2_v2a > 20:
        print(f"  ✅ Transfer SURVIVED the freeze. V2A = {post_t2_v2a:.1f}% (well above chance).")
        print("  The alignment design itself is sound.")
        print("  Retention is the only open problem — needs replay/EWC on shared components.")
    elif post_t2_v2a > 12:
        print(f"  ⚠️ Transfer PARTIALLY survived. V2A = {post_t2_v2a:.1f}% (modestly above chance).")
        print("  The alignment produces some transfer, but it's weaker than the unfrozen case.")
    else:
        print(f"  ❌ Transfer DID NOT survive. V2A = {post_t2_v2a:.1f}% (near chance).")
        print("  The earlier transfer was largely an artifact of the unfrozen backbone drifting.")
        print("  The alignment design itself needs rethinking.")


if __name__ == "__main__":
    main()

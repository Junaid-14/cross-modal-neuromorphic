#!/usr/bin/env python3
"""Standalone transfer eval + CSV write for the completed rehearsal run."""

import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT))

from m7_trimodal_run_provenance import (
    TASKS, ExperimentConfig, M7_ContinualAdaptiveModel,
    ContinualLearner, EmbeddingReplayBuffer, get_device,
    nmnist_collate_fn, shd_collate_fn, dvs_collate_fn,
)
from utils.cross_modal_transfer import evaluate_cross_modal_transfer

# --- Reconstruct config (must match run_cl_rehearsal.py) ---
cfg = ExperimentConfig(
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

device = get_device(cfg)
model = M7_ContinualAdaptiveModel(tasks=TASKS, cfg=cfg).to(device)
buffer = EmbeddingReplayBuffer(feature_dim=cfg.hidden_dim, samples_per_class=cfg.replay_samples_per_class, device=device)
trainer = ContinualLearner(model=model, buffer=buffer, cfg=cfg, device=device)

# Load final T3 checkpoint
cell_ckpt_dir = REPO_ROOT / "results" / "cl_rehearsal" / "checkpoints" / "C_rehearsal_seed42"
ckpt_t3 = cell_ckpt_dir / "final_dvs_dvs.pt"
model.load_state_dict(torch.load(ckpt_t3, map_location=str(device))["model_state_dict"])
model.eval()

# Build minimal loaders for N-MNIST and SHD test sets
data_dir = Path("./ICONS_M7/datasets")
import tonic
from torch.utils.data import DataLoader, random_split

# N-MNIST
sensor_size = tonic.datasets.NMNIST.sensor_size
nmnist_transform = tonic.transforms.Compose([
    tonic.transforms.Denoise(filter_time=10000),
    tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
])
nmnist_test = tonic.datasets.NMNIST(save_to=str(data_dir), train=False, transform=nmnist_transform)
nmnist_test_loader = DataLoader(nmnist_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=nmnist_collate_fn)

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
shd_test_data = [_events_to_dense(e, l) for e, l in shd_test]
shd_test_loader = DataLoader(shd_test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=shd_collate_fn)

# Transfer eval
print("\n📊 TRANSFER EVAL (final checkpoint)")
transfer_metrics = evaluate_cross_modal_transfer(
    model, nmnist_test_loader, shd_test_loader, device,
    num_classes=10, samples_per_class=100, random_state=42,
)
print(f"   V2A: {transfer_metrics['v2a']*100:.2f}%")
print(f"   A2V: {transfer_metrics['a2v']*100:.2f}%")
print(f"   ProtoAlign: {transfer_metrics['prototype_alignment']:.3f}")
print(f"   SilVisual: {transfer_metrics['sil_visual']:.3f}")
print(f"   SilAudio: {transfer_metrics['sil_audio']:.3f}")

# Write CSV
results_dir = REPO_ROOT / "results" / "cl_rehearsal"
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / "cl_rehearsal_results.csv"
fieldnames = [
    "condition", "seed", "T1_after_T1", "T2_after_T2", "T3_after_T3",
    "T1_after_T2", "T1_after_T3", "T2_after_T3",
    "forgetting_T1_T2", "forgetting_T1_T3", "forgetting_T2_T3",
    "total_forgetting_T1_T3", "total_forgetting_T2_T3",
    "V2A", "A2V", "prototype_alignment", "silhouette_score", "replay_weight",
]

result = {
    "condition": "C_rehearsal", "seed": 42,
    "T1_after_T1": 97.50, "T2_after_T2": 72.26, "T3_after_T3": 76.89,
    "T1_after_T2": 95.81, "T1_after_T3": 95.65, "T2_after_T3": 73.14,
    "forgetting_T1_T2": 97.50 - 95.81,
    "forgetting_T1_T3": 97.50 - 95.65,
    "forgetting_T2_T3": 72.26 - 73.14,
    "total_forgetting_T1_T3": 97.50 - 95.65,
    "total_forgetting_T2_T3": 72.26 - 73.14,
    "V2A": transfer_metrics["v2a"] * 100,
    "A2V": transfer_metrics["a2v"] * 100,
    "prototype_alignment": transfer_metrics["prototype_alignment"],
    "silhouette_score": transfer_metrics["sil_visual"],
    "replay_weight": 1.0,
}

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(result)

print(f"\n✅ Results written to {csv_path}")
for k, v in result.items():
    print(f"   {k}: {v}")

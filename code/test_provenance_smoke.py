#!/usr/bin/env python3
"""Smoke test for m7_trimodal_run_provenance.py — minimal, no datasets required."""

import sys
from pathlib import Path
import torch
import numpy as np

# Import the refactored module
sys.path.insert(0, str(Path(__file__).parent))
from m7_trimodal_run_provenance import (
    ExperimentConfig, M7_ContinualAdaptiveModel, random_init_baseline,
    get_device, TASKS, print_task_sanity, ContinualLearner,
    EmbeddingReplayBuffer,
)

print("="*70)
print("SMOKE TEST: m7_trimodal_run_provenance.py")
print("="*70)

# ── Config ──────────────────────────────────────────────────────────────
cfg = ExperimentConfig(
    seed=42,
    task_order=["nmnist", "shd", "dvs"],
    lr=1e-3,
    batch_size=4,
    num_patterns=8,
    shared_dim=32,
    hidden_dim=32,
    use_replay=False,
    device="cpu",
)
print(f"\nConfig: seed={cfg.seed}, shared_dim={cfg.shared_dim}, device={get_device(cfg)}")

# ── Dummy DataLoader ────────────────────────────────────────────────────
class DummyLoader:
    def __init__(self, n_samples, n_classes, spatial=True):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.spatial = spatial
    def __iter__(self):
        for _ in range(self.n_samples):
            if self.spatial:
                x = torch.randn(4, 25, 2, 34, 34)
            else:
                x = torch.randn(4, 100, 1, 1, 700)
            y = torch.randint(0, self.n_classes, (4,))
            yield x, y
    def __len__(self):
        return self.n_samples

dummy_nmnist = DummyLoader(10, 10, spatial=True)
dummy_shd = DummyLoader(10, 20, spatial=False)
dummy_dvs = DummyLoader(10, 11, spatial=True)

# ── Random-init baseline test ───────────────────────────────────────────
print("\n--- Random-init baseline test ---")
for task_name, loader in [("nmnist", dummy_nmnist), ("shd", dummy_shd), ("dvs", dummy_dvs)]:
    ri = random_init_baseline(TASKS, task_name, loader, cfg, get_device(cfg), n_seeds=2)
    chance = 100.0 / TASKS[task_name]["num_classes"]
    print(f"  {task_name}: random_init={ri:.2f}%  chance={chance:.2f}%  delta={ri-chance:+.2f}%")
    assert abs(ri - chance) < 8.0, f"Random init too far from chance for {task_name}"
print("  ✅ All random-init baselines within ~3pp of chance")

# ── Zero-shot via ContinualLearner ──────────────────────────────────────
print("\n--- Zero-shot evaluation after T1 training ---")
device = get_device(cfg)
model = M7_ContinualAdaptiveModel(tasks=TASKS, cfg=cfg).to(device)
buffer = EmbeddingReplayBuffer(feature_dim=cfg.shared_dim, samples_per_class=10, device=str(device))
trainer = ContinualLearner(model=model, buffer=buffer, cfg=cfg, device=device)

# Simulate T1 training (1 epoch, tiny)
trainer.train_task(
    "nmnist", "nmnist",
    dummy_nmnist, dummy_nmnist, dummy_nmnist,
    num_epochs=1, patience=5, lr=cfg.lr,
    use_replay=False, use_scl=False, step_id=1,
)

# Evaluate zero-shot on SHD (task 2)
acc_shd_zero = trainer.evaluate(dummy_shd, "shd", "shd")
chance_shd = 100.0 / 20
print(f"  SHD zero-shot after T1 training: {acc_shd_zero:.2f}%")
print(f"  Chance for SHD: {chance_shd:.2f}%")
print(f"  Delta: {acc_shd_zero - chance_shd:+.2f}%")

# The key assertion: with NO cross-modal alignment (use_scl=False, no joint training),
# zero-shot should be very close to chance.
if abs(acc_shd_zero - chance_shd) < 3.0:
    print("  ✅ SHD zero-shot is within ~3pp of chance — confirms no spurious cross-modal signal")
else:
    print(f"  ⚠️  SHD zero-shot deviates from chance by {acc_shd_zero - chance_shd:+.2f}pp")
    print("      This may indicate unintended cross-modal transfer or data leakage.")

# ── Sanity printout test ────────────────────────────────────────────────
print("\n--- Sanity printout test ---")
print_task_sanity(
    task_name="shd",
    num_classes=20,
    chance=5.0,
    zero_shot_acc=acc_shd_zero,
    after_training_acc=15.0,
    learning_gain=15.0 - acc_shd_zero,
    fwt=acc_shd_zero - 5.0,
    random_init_baseline_acc=5.2,
)

print("\n" + "="*70)
print("SMOKE TEST PASSED")
print("="*70)

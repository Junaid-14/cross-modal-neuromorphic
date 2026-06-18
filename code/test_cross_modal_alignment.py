#!/usr/bin/env python3
"""Smoke test for cross-modal alignment in m7_trimodal_run_provenance.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import Dataset

from m7_trimodal_run_provenance import (
    ExperimentConfig,
    M7_ContinualAdaptiveModel,
    ContinualLearner,
    EmbeddingReplayBuffer,
    get_device,
    CrossModalSupConLoss,
    orthogonal_penalty,
)


# ── Fake paired dataset ───────────────────────────────────────────────────
class FakePairedDataset(Dataset):
    def __init__(self, n_samples_per_class=20):
        self.n_samples = n_samples_per_class * 10
        self.n_per_class = n_samples_per_class

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        c = idx % 10
        # Visual: (25, 2, 34, 34)
        vis = torch.randn(25, 2, 34, 34) * 0.1 + c * 0.05
        # Audio: (100, 1, 1, 700)
        aud = torch.randn(100, 1, 1, 700) * 0.1 + c * 0.05
        return vis, aud, c


# ── Config ────────────────────────────────────────────────────────────────
cfg = ExperimentConfig(
    seed=42,
    task_order=["nmnist", "shd", "dvs"],
    lr=1e-3,
    batch_size=30,
    shared_dim=64,
    align_shared_dim=16,
    num_patterns=8,
    use_contrastive=True,
    contrastive_weight=0.1,
    align_weight=0.5,
    align_temperature=0.1,
    align_use_orthogonal=True,
    align_orthogonal_weight=0.01,
    device="cpu",
)

TASKS = {
    "nmnist": {"num_classes": 10, "type": "visual"},
    "shd":    {"num_classes": 20, "type": "audio"},
    "dvs":    {"num_classes": 11, "type": "visual_motion"},
}

print("="*70)
print("SMOKE TEST: Cross-Modal Alignment")
print("="*70)
print(f"  shared_dim={cfg.shared_dim}, align_shared_dim={cfg.align_shared_dim}")
print(f"  private_dim={cfg.shared_dim - cfg.align_shared_dim}")

# ── Model ─────────────────────────────────────────────────────────────────
device = get_device(cfg)
model = M7_ContinualAdaptiveModel(tasks=TASKS, cfg=cfg).to(device)

# Verify shared/private projections exist
assert hasattr(model, "proj_shared"), "Model missing proj_shared"
assert hasattr(model, "proj_private"), "Model missing proj_private"
print("  ✅ proj_shared and proj_private exist")

# ── Forward with shared ───────────────────────────────────────────────────
vis = torch.randn(4, 25, 2, 34, 34).to(device)
aud = torch.randn(4, 100, 1, 1, 700).to(device)

logits_vis, gate_vis, full_vis, shared_vis = model.forward_with_shared(vis, "nmnist", "nmnist")
logits_aud, gate_aud, full_aud, shared_aud = model.forward_with_shared(aud, "shd", "shd")

assert full_vis.shape == (4, cfg.shared_dim), f"full_vis shape {full_vis.shape} != (4, {cfg.shared_dim})"
assert shared_vis.shape == (4, cfg.align_shared_dim), f"shared_vis shape {shared_vis.shape} != (4, {cfg.align_shared_dim})"
assert full_aud.shape == (4, cfg.shared_dim)
assert shared_aud.shape == (4, cfg.align_shared_dim)
print("  ✅ forward_with_shared returns correct shapes")

# ── Standard forward unchanged ────────────────────────────────────────────
logits_std, gate_std, features_std = model(vis, "nmnist", "nmnist")
assert logits_std.shape == (4, 10)
assert features_std.shape == (4, cfg.shared_dim)
print("  ✅ Standard forward() signature unchanged")

# ── Orthogonal penalty ────────────────────────────────────────────────────
private_vis = full_vis[:, cfg.align_shared_dim:]
pen = orthogonal_penalty(shared_vis, private_vis)
assert pen.ndim == 0, "Orthogonal penalty should be scalar"
assert pen.item() >= 0, "Orthogonal penalty should be non-negative"
print(f"  ✅ Orthogonal penalty = {pen.item():.4f}")

# ── CrossModalSupConLoss ──────────────────────────────────────────────────
xmscl = CrossModalSupConLoss(temperature=cfg.align_temperature)
shared_cat = torch.cat([shared_vis, shared_aud], dim=0)
labels_cat = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long).to(device)
loss_xmscl = xmscl(shared_cat, labels_cat)
assert loss_xmscl.ndim == 0
assert loss_xmscl.item() > 0, "SupCon loss should be positive"
print(f"  ✅ CrossModalSupConLoss = {loss_xmscl.item():.4f}")

# ── align_cross_modal training ────────────────────────────────────────────
print("\n--- Running alignment phase (2 epochs) ---")
buffer = EmbeddingReplayBuffer(feature_dim=cfg.shared_dim, samples_per_class=10, device=str(device))
trainer = ContinualLearner(model=model, buffer=buffer, cfg=cfg, device=device)

paired_ds = FakePairedDataset(n_samples_per_class=20)
from torch.utils.data import DataLoader
from Dataloaders.paired_digit_loader import paired_digit_collate_fn

paired_loader = DataLoader(
    paired_ds, batch_size=30, shuffle=False, drop_last=True, collate_fn=paired_digit_collate_fn
)

history = trainer.align_cross_modal(
    paired_loader=paired_loader,
    num_epochs=2,
    lr=cfg.lr,
    use_scl=True,
    scl_weight=cfg.contrastive_weight,
    align_weight=cfg.align_weight,
    step_id=99,
)

print(f"  Epoch 1 xmscl_loss: {history['xmscl_loss'][0]:.4f}")
print(f"  Epoch 2 xmscl_loss: {history['xmscl_loss'][1]:.4f}")
assert history['xmscl_loss'][1] < history['xmscl_loss'][0], \
    f"Cross-modal loss should decrease: {history['xmscl_loss']}"
print("  ✅ Cross-modal loss decreased over epochs")

# ── Post-alignment standard task training still works ─────────────────────
print("\n--- Post-alignment task training sanity ---")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
ce = torch.nn.CrossEntropyLoss()

for _ in range(3):
    optimizer.zero_grad()
    logits, gate, features = model(vis, "nmnist", "nmnist")
    loss = ce(logits, torch.randint(0, 10, (4,)))
    loss.backward()
    optimizer.step()

print("  ✅ Post-alignment task training runs normally")

# ── Gate/heads/backbone unchanged ─────────────────────────────────────────
# Verify gate still outputs softmax weights
assert gate.shape == (4, 2), f"Gate shape {gate.shape} != (4, 2)"
assert torch.allclose(gate.sum(dim=1), torch.ones(4).to(device), atol=1e-5), "Gate should softmax"
print("  ✅ Gate outputs valid softmax weights")

print("\n" + "="*70)
print("CROSS-MODAL ALIGNMENT SMOKE TEST PASSED")
print("="*70)

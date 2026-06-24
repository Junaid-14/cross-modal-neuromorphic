#!/usr/bin/env python3
"""Quick MPS benchmark for different batch sizes."""

import sys
import time
sys.path.insert(0, "/Users/agent1/Documents/neuromorphic/m7-trimodal-cl")

import torch
from Models.registry import build_model
from training.pipeline import train_step
from torch.utils.data import TensorDataset, DataLoader

# Simulate N-MNIST-like data
BATCH_SIZES = [32, 64, 128]
NUM_BATCHES = 25
TIME_STEPS = 25
DEVICE = torch.device("mps")

print(f"Device: {DEVICE}")
print(f"Warming up...")

model = build_model("baseline", input_type="nmnist", num_classes=10).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for bs in BATCH_SIZES:
    # Create synthetic data
    x = torch.randn(bs, TIME_STEPS, 2, 34, 34)
    y = torch.randint(0, 10, (bs,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)

    ce_loss = torch.nn.CrossEntropyLoss()

    # Warmup
    model.train()
    for _ in range(3):
        for data, labels in loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            spk_sum, _ = model(data)
            loss = ce_loss(spk_sum, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.mps.synchronize() if hasattr(torch.mps, "synchronize") else None

    # Timed run
    model.train()
    t0 = time.time()
    for i, (data, labels) in enumerate(loader):
        if i >= NUM_BATCHES:
            break
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        spk_sum, _ = model(data)
        loss = ce_loss(spk_sum, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.mps.synchronize() if hasattr(torch.mps, "synchronize") else None
    elapsed = time.time() - t0

    per_batch = elapsed / NUM_BATCHES
    samples_per_sec = bs / per_batch
    print(f"bs={bs:3d}: {elapsed:.2f}s for {NUM_BATCHES} batches = {per_batch:.3f}s/batch = {samples_per_sec:.0f} samples/sec")

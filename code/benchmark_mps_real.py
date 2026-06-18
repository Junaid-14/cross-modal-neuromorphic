#!/usr/bin/env python3
"""Realistic MPS benchmark using actual N-MNIST loader."""

import sys
import time
sys.path.insert(0, "/Users/agent1/Documents/neuromorphic/m7-trimodal-cl")

import torch
from Models.registry import build_model
from torch.utils.data import DataLoader
from Dataloaders.nmnist_loader import get_nmnist_loaders

BATCH_SIZES = [32, 64, 128]
NUM_BATCHES = 50
DEVICE = torch.device("mps")

print(f"Device: {DEVICE}")

for bs in BATCH_SIZES:
    model = build_model("baseline", input_type="nmnist", num_classes=10).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ce_loss = torch.nn.CrossEntropyLoss()

    train_loader, _ = get_nmnist_loaders(batch_size=bs, num_workers=0)

    # Warmup
    model.train()
    for i, (data, labels) in enumerate(train_loader):
        if i >= 3:
            break
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        spk_sum, _ = model(data)
        loss = ce_loss(spk_sum, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Timed run
    model.train()
    t0 = time.time()
    for i, (data, labels) in enumerate(train_loader):
        if i >= NUM_BATCHES:
            break
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        spk_sum, _ = model(data)
        loss = ce_loss(spk_sum, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - t0

    per_batch = elapsed / NUM_BATCHES
    samples_per_sec = bs / per_batch
    print(f"bs={bs:3d}: {elapsed:.2f}s for {NUM_BATCHES} batches = {per_batch:.3f}s/batch = {samples_per_sec:.0f} samples/sec")

#!/usr/bin/env python3
"""Debug script to verify gradient flow in the SNN."""

import sys
sys.path.insert(0, "/Users/agent1/Documents/neuromorphic/m7-trimodal-cl")

import torch
import torch.nn as nn
from Models.registry import build_model

def check_gradients(device_name="cpu"):
    device = torch.device(device_name)
    print(f"\n{'='*60}")
    print(f"Testing on device: {device}")
    print(f"{'='*60}")

    model = build_model("baseline", input_type="nmnist", num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    # Single synthetic batch
    x = torch.randn(4, 25, 2, 34, 34).to(device)
    y = torch.tensor([0, 1, 2, 3]).to(device)

    # Check 1: Forward pass works
    print("\n1. Forward pass...")
    model.train()
    spk_sum, features = model(x)
    print(f"   spk_sum shape: {spk_sum.shape}, dtype: {spk_sum.dtype}")
    print(f"   spk_sum requires_grad: {spk_sum.requires_grad}")
    print(f"   spk_sum min/max/mean: {spk_sum.min():.4f} / {spk_sum.max():.4f} / {spk_sum.mean():.4f}")

    # Check 2: Loss computation
    print("\n2. Loss computation...")
    loss = ce_loss(spk_sum, y)
    print(f"   loss: {loss.item():.4f}")
    print(f"   loss requires_grad: {loss.requires_grad}")

    # Check 3: Backward pass
    print("\n3. Backward pass...")
    optimizer.zero_grad()
    loss.backward()

    has_grad = False
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gnorm = param.grad.norm().item()
            grad_norms.append((name, gnorm))
            if gnorm > 0:
                has_grad = True
        else:
            grad_norms.append((name, None))

    print(f"   Gradients exist: {has_grad}")
    print(f"   Sample gradient norms:")
    for name, gnorm in grad_norms[:6]:
        print(f"      {name}: {gnorm}")

    # Check 4: Optimizer step changes weights
    print("\n4. Optimizer step...")
    weight_before = list(model.parameters())[0].clone()
    optimizer.step()
    weight_after = list(model.parameters())[0]
    changed = not torch.allclose(weight_before, weight_after)
    print(f"   Weights changed: {changed}")

    # Check 5: Overfit test (tiny)
    print("\n5. Tiny overfit test (20 samples, 10 epochs)...")
    model2 = build_model("baseline", input_type="nmnist", num_classes=10).to(device)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

    x_small = torch.randn(20, 25, 2, 34, 34).to(device)
    y_small = torch.arange(20) % 10  # deterministic labels
    y_small = y_small.to(device)

    for epoch in range(10):
        model2.train()
        spk_sum, _ = model2(x_small)
        loss = ce_loss(spk_sum, y_small)
        opt2.zero_grad()
        loss.backward()
        opt2.step()

        acc = (spk_sum.argmax(1) == y_small).float().mean().item() * 100
        if epoch % 2 == 0 or epoch == 9:
            print(f"   Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1f}%")

    print(f"\n   Final accuracy: {acc:.1f}%")
    if acc > 50:
        print("   ✓ Model CAN learn (overfit test passed)")
    else:
        print("   ✗ Model NOT learning (overfit test failed)")

if __name__ == "__main__":
    check_gradients("cpu")
    if torch.backends.mps.is_available():
        check_gradients("mps")

#!/usr/bin/env python3
"""
Verify that the PyTorch discrete-time simulator matches Brian2/PyNN IF_curr_exp.

Generates ONE set of Poisson input spikes for 20 stratified samples (2/class),
feeds the IDENTICAL spike arrays to BOTH simulators with SCALE=10.0, and
compares per-head-neuron spike counts and argmax predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

import pyNN.brian2 as sim
from brian2 import ms

spike_grad = surrogate.atan()


class ConvFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.flat = nn.Flatten()

    def forward(self, x):
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        x = x.permute(1, 0, 2, 3, 4)
        rec = []
        for step in range(x.shape[0]):
            s1 = self.lif1(self.pool1(self.conv1(x[step])))
            s2 = self.lif2(self.pool2(self.conv2(s1)))
            rec.append(self.flat(s2))
        return torch.stack(rec, dim=0).sum(dim=0)


def collect_stratified(test_loader, per_class: int):
    x_list, y_list = [], []
    counts = {i: 0 for i in range(10)}
    for x, y in test_loader:
        label = int(y.item())
        if counts[label] < per_class:
            x_list.append(x)
            y_list.append(label)
            counts[label] += 1
        if all(c >= per_class for c in counts.values()):
            break
    return torch.cat(x_list), np.array(y_list)


def generate_spikes(features, rate_scale, duration_ms, seed):
    """Generate Poisson spikes for all samples. Returns list of (indices, times_ms) per sample."""
    rng = np.random.default_rng(seed)
    n_samples, n_inputs = features.shape
    n_steps = int(duration_ms)
    spike_data = []
    for s in range(n_samples):
        rates = np.clip(features[s] * rate_scale, 0.0, None)
        probs = rates / 1000.0
        spike_mask = rng.random((n_steps, n_inputs)) < probs[None, :]
        indices = []
        times = []
        for nid in range(n_inputs):
            ts = np.where(spike_mask[:, nid])[0]
            for t in ts:
                indices.append(nid)
                times.append(float(t))
        spike_data.append((np.array(indices, dtype=np.int32), np.array(times, dtype=np.float64)))
    return spike_data


# ========================================================================
# PyTorch discrete-time simulator (current implementation)
# ========================================================================

def run_pytorch_sim(features, spike_data, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
                    scale, duration_ms):
    n_samples, n_inputs = features.shape
    n_steps = int(duration_ms)
    device = torch.device("cpu")

    # Convert spike_data to dense tensor (n_samples, n_steps, n_inputs)
    input_spikes = torch.zeros(n_samples, n_steps, n_inputs, dtype=torch.float32, device=device)
    for s, (indices, times) in enumerate(spike_data):
        for nid, t in zip(indices, times):
            input_spikes[s, int(t), nid] = 1.0

    def prep(W, b):
        Wt = torch.from_numpy(W).to(device=device, dtype=torch.float32)
        bt = torch.from_numpy(b).to(device=device, dtype=torch.float32)
        Ws = Wt * scale
        mask = torch.abs(Ws) > 0.001
        Ws = Ws * mask
        return Ws, bt * 10.0

    W1, b1 = prep(W_fc1, b_fc1)
    W2, b2 = prep(W_fc2, b_fc2)
    W3, b3 = prep(W_head, b_head)

    dt = 1.0
    alpha_v = np.exp(-dt / 20.0)
    alpha_ie = np.exp(-dt / 5.0)
    v_factor = (20.0 / 1.0) * (1.0 - alpha_v)
    refrac_steps = int(2.0 / dt)

    v1 = torch.full((n_samples, 1024), -65.0, dtype=torch.float32, device=device)
    ie1 = torch.zeros((n_samples, 1024), dtype=torch.float32, device=device)
    r1 = torch.zeros((n_samples, 1024), dtype=torch.int32, device=device)
    s1_acc = torch.zeros((n_samples, 1024), dtype=torch.int32, device=device)

    v2 = torch.full((n_samples, 512), -65.0, dtype=torch.float32, device=device)
    ie2 = torch.zeros((n_samples, 512), dtype=torch.float32, device=device)
    r2 = torch.zeros((n_samples, 512), dtype=torch.int32, device=device)
    s2_acc = torch.zeros((n_samples, 512), dtype=torch.int32, device=device)

    v3 = torch.full((n_samples, 10), -65.0, dtype=torch.float32, device=device)
    ie3 = torch.zeros((n_samples, 10), dtype=torch.float32, device=device)
    r3 = torch.zeros((n_samples, 10), dtype=torch.int32, device=device)
    s3_acc = torch.zeros((n_samples, 10), dtype=torch.int32, device=device)

    for t in range(n_steps):
        # FC1
        ie1 = ie1 * alpha_ie + input_spikes[:, t] @ W1.T
        active1 = r1 == 0
        v1 = torch.where(active1,
                         -65.0 + (v1 + 65.0) * alpha_v + (ie1 + b1) * v_factor,
                         v1)
        spike1 = active1 & (v1 >= -50.0)
        v1 = torch.where(spike1, torch.full_like(v1, -65.0), v1)
        r1 = torch.where(spike1, torch.full_like(r1, refrac_steps, dtype=torch.int32), r1)
        r1 = torch.clamp(r1 - 1, min=0)
        s1_acc += spike1.to(torch.int32)

        # FC2
        ie2 = ie2 * alpha_ie + spike1.to(torch.float32) @ W2.T
        active2 = r2 == 0
        v2 = torch.where(active2,
                         -65.0 + (v2 + 65.0) * alpha_v + (ie2 + b2) * v_factor,
                         v2)
        spike2 = active2 & (v2 >= -50.0)
        v2 = torch.where(spike2, torch.full_like(v2, -65.0), v2)
        r2 = torch.where(spike2, torch.full_like(r2, refrac_steps, dtype=torch.int32), r2)
        r2 = torch.clamp(r2 - 1, min=0)
        s2_acc += spike2.to(torch.int32)

        # Head
        ie3 = ie3 * alpha_ie + spike2.to(torch.float32) @ W3.T
        active3 = r3 == 0
        v3 = torch.where(active3,
                         -65.0 + (v3 + 65.0) * alpha_v + (ie3 + b3) * v_factor,
                         v3)
        spike3 = active3 & (v3 >= -50.0)
        v3 = torch.where(spike3, torch.full_like(v3, -65.0), v3)
        r3 = torch.where(spike3, torch.full_like(r3, refrac_steps, dtype=torch.int32), r3)
        r3 = torch.clamp(r3 - 1, min=0)
        s3_acc += spike3.to(torch.int32)

    head_counts = s3_acc.cpu().numpy()
    preds = np.argmax(head_counts, axis=1)
    return head_counts, preds


# ========================================================================
# Brian2/PyNN simulator (using SpikeSourceArray with identical spikes)
# ========================================================================

def run_brian2_sim(spike_data, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
                   scale, duration_ms):
    n_samples = len(spike_data)
    sim.setup(timestep=1.0, min_delay=1.0)

    results = []
    for s in range(n_samples):
        indices, times = spike_data[s]

        # Input
        input_pop = sim.Population(1600, sim.SpikeSourceArray(spike_times=[]))
        if len(indices) > 0:
            input_pop.brian2_group.set_spikes(indices, times * ms)

        # FC1
        fc1_pop = sim.Population(1024, sim.IF_curr_exp(
            tau_m=20.0, v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0,
            tau_refrac=2.0, tau_syn_E=5.0, tau_syn_I=5.0, cm=1.0, i_offset=0.0,
        ))
        fc1_pop.initialize(v=-65.0)
        fc1_pop.record(["spikes"])

        # FC2
        fc2_pop = sim.Population(512, sim.IF_curr_exp(
            tau_m=20.0, v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0,
            tau_refrac=2.0, tau_syn_E=5.0, tau_syn_I=5.0, cm=1.0, i_offset=0.0,
        ))
        fc2_pop.initialize(v=-65.0)
        fc2_pop.record(["spikes"])

        # Head
        head_pop = sim.Population(10, sim.IF_curr_exp(
            tau_m=20.0, v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0,
            tau_refrac=2.0, tau_syn_E=5.0, tau_syn_I=5.0, cm=1.0, i_offset=0.0,
        ))
        head_pop.initialize(v=-65.0)
        head_pop.record(["spikes"])

        # Build projections (use chunking to avoid memory issues)
        def build_proj(pre, post, W, b, chunk_post=256):
            n_post, n_pre = W.shape
            for post_start in range(0, n_post, chunk_post):
                post_end = min(post_start + chunk_post, n_post)
                conn = []
                for post_i in range(post_start, post_end):
                    for pre_i in range(n_pre):
                        w = float(W[post_i, pre_i]) * scale
                        if abs(w) > 0.001:
                            conn.append((pre_i, post_i - post_start, w, 1.0))
                if conn:
                    post_chunk = sim.PopulationView(post, list(range(post_start, post_end)))
                    sim.Projection(pre, post_chunk, sim.FromListConnector(conn), sim.StaticSynapse(), receptor_type="excitatory")
            # Bias
            for i in range(n_post):
                post[i:i+1].set(i_offset=float(b[i]) * 10.0)

        build_proj(input_pop, fc1_pop, W_fc1, b_fc1)
        build_proj(fc1_pop, fc2_pop, W_fc2, b_fc2)
        build_proj(fc2_pop, head_pop, W_head, b_head)

        sim.run(duration_ms)

        data = head_pop.get_data("spikes")
        counts = np.array([len(st) for st in data.segments[-1].spiketrains])
        results.append(counts)

        sim.reset()

    sim.end()
    head_counts = np.stack(results, axis=0)
    preds = np.argmax(head_counts, axis=1)
    return head_counts, preds


# ========================================================================
# Main verification
# ========================================================================

def main():
    print("=" * 70)
    print("VERIFICATION: PyTorch discrete-time vs Brian2/PyNN IF_curr_exp")
    print("=" * 70)

    # Load checkpoint
    print("\n1. Loading checkpoint...")
    cp = torch.load("ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt", map_location="cpu")
    sd = cp["model_state_dict"]

    W_fc1 = sd["backbones.nmnist.fc1.weight"].numpy().astype(np.float64)
    b_fc1 = sd["backbones.nmnist.fc1.bias"].numpy().astype(np.float64)
    W_fc2 = sd["backbones.nmnist.fc2.weight"].numpy().astype(np.float64)
    b_fc2 = sd["backbones.nmnist.fc2.bias"].numpy().astype(np.float64)
    W_head = sd["heads.nmnist.1.weight"].numpy().astype(np.float64)
    b_head = sd["heads.nmnist.1.bias"].numpy().astype(np.float64)

    # Load data
    print("2. Loading N-MNIST test data...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from Dataloaders.nmnist_loader import get_nmnist_loaders
    _, test_loader = get_nmnist_loaders(batch_size=1, num_workers=0)
    test_x, test_y = collect_stratified(test_loader, 2)
    print(f"   Samples: {len(test_y)} (2/class)")

    # Extract features
    print("3. Extracting conv features...")
    frontend = ConvFrontend()
    frontend.load_state_dict(
        {k.replace("backbones.nmnist.", ""): v for k, v in sd.items() if k.startswith("backbones.nmnist.") and ("conv" in k or "lif1" in k or "lif2" in k)},
        strict=False,
    )
    frontend.eval()
    with torch.no_grad():
        features = frontend(test_x).numpy()

    # Generate spikes
    print("4. Generating Poisson spikes (seed=42, rate_scale=10.0, duration=100ms)...")
    spike_data = generate_spikes(features, rate_scale=10.0, duration_ms=100.0, seed=42)
    total_spikes = sum(len(t) for _, t in spike_data)
    print(f"   Total input spikes across all samples: {total_spikes}")

    # Run PyTorch sim
    print("\n5. Running PyTorch discrete-time simulator...")
    pt_counts, pt_preds = run_pytorch_sim(
        features, spike_data, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
        scale=10.0, duration_ms=100.0
    )
    print("   Done.")

    # Run Brian2 sim
    print("\n6. Running Brian2/PyNN simulator...")
    b2_counts, b2_preds = run_brian2_sim(
        spike_data, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
        scale=10.0, duration_ms=100.0
    )
    print("   Done.")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Sample':>6} | {'Label':>5} | {'PyTorch counts':>30} | {'Brian2 counts':>30} | {'Match':>5}")
    print("-" * 110)

    all_match = True
    max_diff = 0.0
    for s in range(len(test_y)):
        pt_c = pt_counts[s]
        b2_c = b2_counts[s]
        pt_p = pt_preds[s]
        b2_p = b2_preds[s]
        match = "Y" if pt_p == b2_p else "N"
        if pt_p != b2_p:
            all_match = False
        diff = np.abs(pt_c - b2_c)
        rel_diff = diff / np.maximum(b2_c, 1)
        max_diff = max(max_diff, rel_diff.max())
        pt_str = " ".join(f"{c:3d}" for c in pt_c)
        b2_str = " ".join(f"{c:3d}" for c in b2_c)
        print(f"{s:6d} | {test_y[s]:5d} | {pt_str:>30} | {b2_str:>30} | {match:>5}")

    print("-" * 110)
    total_pt = pt_counts.sum()
    total_b2 = b2_counts.sum()
    print(f"\nTotal head spikes: PyTorch={total_pt}, Brian2={total_b2}, diff={abs(total_pt-total_b2)}")
    print(f"Predictions match: {np.sum(pt_preds == b2_preds)}/{len(test_y)}")
    print(f"Max per-neuron relative diff: {max_diff:.3f}")

    if all_match and max_diff < 0.05:
        print("\n✓ VERIFIED: PyTorch discrete-time simulator matches Brian2/PyNN IF_curr_exp.")
    else:
        print("\n✗ MISMATCH DETECTED. Investigating...")
        # Find the sample with the biggest difference
        diffs = np.abs(pt_counts - b2_counts).sum(axis=1)
        worst = np.argmax(diffs)
        print(f"   Worst sample: {worst} (diff={diffs[worst]})")
        print(f"   PyTorch counts: {pt_counts[worst]}")
        print(f"   Brian2 counts:  {b2_counts[worst]}")

    print("=" * 70)


if __name__ == "__main__":
    main()

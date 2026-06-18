#!/usr/bin/env python3
"""
Verify PyTorch discrete-time simulator against raw Brian2 IF_curr_exp.

Uses raw Brian2 (not PyNN) with vectorized Synapses for speed.
The neuron equations are the exact IF_curr_exp model used by PyNN/SpiNNaker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from brian2 import (
    NeuronGroup, Synapses, SpikeGeneratorGroup, SpikeMonitor,
    ms, mV, nA, nF, second, defaultclock,
    run, start_scope, clear_cache, prefs,
)
prefs.codegen.target = 'numpy'

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
# PyTorch sim — corrected (exact cross-factor)
# ========================================================================

def run_pytorch_sim(features, spike_data, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
                    scale, duration_ms):
    n_samples, n_inputs = features.shape
    n_steps = int(duration_ms)
    device = torch.device("cpu")

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
    v_factor = 20.0 * (1.0 - alpha_v)
    cross_factor = (20.0 * 5.0 / 15.0) * (alpha_v - alpha_ie)
    refrac_steps = 2

    v1 = torch.full((n_samples, 1024), -65.0, dtype=torch.float32, device=device)
    ie1 = torch.zeros((n_samples, 1024), dtype=torch.float32, device=device)
    r1 = torch.zeros((n_samples, 1024), dtype=torch.int32, device=device)

    v2 = torch.full((n_samples, 512), -65.0, dtype=torch.float32, device=device)
    ie2 = torch.zeros((n_samples, 512), dtype=torch.float32, device=device)
    r2 = torch.zeros((n_samples, 512), dtype=torch.int32, device=device)

    v3 = torch.full((n_samples, 10), -65.0, dtype=torch.float32, device=device)
    ie3 = torch.zeros((n_samples, 10), dtype=torch.float32, device=device)
    r3 = torch.zeros((n_samples, 10), dtype=torch.int32, device=device)
    s3_acc = torch.zeros((n_samples, 10), dtype=torch.int32, device=device)

    for t in range(n_steps):
        ie1 = ie1 * alpha_ie + input_spikes[:, t] @ W1.T
        active1 = r1 == 0
        v1 = torch.where(active1,
                         -65.0 + (v1 + 65.0) * alpha_v + ie1 * cross_factor + b1 * v_factor,
                         v1)
        spike1 = active1 & (v1 >= -50.0)
        v1 = torch.where(spike1, torch.full_like(v1, -65.0), v1)
        r1 = torch.where(spike1, torch.full_like(r1, refrac_steps, dtype=torch.int32), r1)
        r1 = torch.clamp(r1 - 1, min=0)

        ie2 = ie2 * alpha_ie + spike1.to(torch.float32) @ W2.T
        active2 = r2 == 0
        v2 = torch.where(active2,
                         -65.0 + (v2 + 65.0) * alpha_v + ie2 * cross_factor + b2 * v_factor,
                         v2)
        spike2 = active2 & (v2 >= -50.0)
        v2 = torch.where(spike2, torch.full_like(v2, -65.0), v2)
        r2 = torch.where(spike2, torch.full_like(r2, refrac_steps, dtype=torch.int32), r2)
        r2 = torch.clamp(r2 - 1, min=0)

        ie3 = ie3 * alpha_ie + spike2.to(torch.float32) @ W3.T
        active3 = r3 == 0
        v3 = torch.where(active3,
                         -65.0 + (v3 + 65.0) * alpha_v + ie3 * cross_factor + b3 * v_factor,
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
# Raw Brian2 sim — vectorized Synapses for speed
# ========================================================================

def run_brian2_sim(spike_data, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
                   scale, duration_ms):
    n_samples = len(spike_data)
    defaultclock.dt = 1.0 * ms

    # IF_curr_exp equations (exact same as PyNN/SpiNNaker)
    eqs = '''
    dv/dt = (v_rest - v)/tau_m + (ie + i_offset)/c_m : volt (unless refractory)
    die/dt = -ie/tau_syn_E : amp
    tau_m : second
    tau_syn_E : second
    c_m : farad
    v_rest : volt
    v_reset : volt
    v_thresh : volt
    tau_refrac : second
    i_offset : amp
    '''

    results = []
    for s in range(n_samples):
        start_scope()

        indices, times = spike_data[s]

        # Input spike generator
        input_gen = SpikeGeneratorGroup(1600, indices, times * ms)

        # FC1: 1600 -> 1024
        fc1 = NeuronGroup(1024, eqs, threshold='v >= v_thresh', reset='v = v_reset',
                          refractory='tau_refrac', method='exact')
        fc1.tau_m = 20.0 * ms
        fc1.tau_syn_E = 5.0 * ms
        fc1.c_m = 1.0 * nF
        fc1.v_rest = -65.0 * mV
        fc1.v_reset = -65.0 * mV
        fc1.v_thresh = -50.0 * mV
        fc1.tau_refrac = 2.0 * ms
        fc1.i_offset = b_fc1 * 10.0 * nA
        fc1.v = -65.0 * mV

        syn1 = Synapses(input_gen, fc1, model='w : amp', on_pre='ie_post += w')
        W1_scaled = W_fc1 * scale
        mask1 = np.abs(W1_scaled) > 0.001
        pre_idx1, post_idx1 = np.nonzero(mask1.T)  # Synapses expects (pre, post)
        syn1.connect(i=pre_idx1, j=post_idx1)
        syn1.w = W1_scaled[post_idx1, pre_idx1] * nA
        syn1.delay = 1.0 * ms

        # FC2: 1024 -> 512
        fc2 = NeuronGroup(512, eqs, threshold='v >= v_thresh', reset='v = v_reset',
                          refractory='tau_refrac', method='exact')
        fc2.tau_m = 20.0 * ms
        fc2.tau_syn_E = 5.0 * ms
        fc2.c_m = 1.0 * nF
        fc2.v_rest = -65.0 * mV
        fc2.v_reset = -65.0 * mV
        fc2.v_thresh = -50.0 * mV
        fc2.tau_refrac = 2.0 * ms
        fc2.i_offset = b_fc2 * 10.0 * nA
        fc2.v = -65.0 * mV

        syn2 = Synapses(fc1, fc2, model='w : amp', on_pre='ie_post += w')
        W2_scaled = W_fc2 * scale
        mask2 = np.abs(W2_scaled) > 0.001
        pre_idx2, post_idx2 = np.nonzero(mask2.T)
        syn2.connect(i=pre_idx2, j=post_idx2)
        syn2.w = W2_scaled[post_idx2, pre_idx2] * nA
        syn2.delay = 1.0 * ms

        # Head: 512 -> 10
        head = NeuronGroup(10, eqs, threshold='v >= v_thresh', reset='v = v_reset',
                           refractory='tau_refrac', method='exact')
        head.tau_m = 20.0 * ms
        head.tau_syn_E = 5.0 * ms
        head.c_m = 1.0 * nF
        head.v_rest = -65.0 * mV
        head.v_reset = -65.0 * mV
        head.v_thresh = -50.0 * mV
        head.tau_refrac = 2.0 * ms
        head.i_offset = b_head * 10.0 * nA
        head.v = -65.0 * mV

        syn3 = Synapses(fc2, head, model='w : amp', on_pre='ie_post += w')
        W3_scaled = W_head * scale
        mask3 = np.abs(W3_scaled) > 0.001
        pre_idx3, post_idx3 = np.nonzero(mask3.T)
        syn3.connect(i=pre_idx3, j=post_idx3)
        syn3.w = W3_scaled[post_idx3, pre_idx3] * nA
        syn3.delay = 1.0 * ms

        # Monitor
        head_mon = SpikeMonitor(head)

        run(duration_ms * ms)

        counts = np.array(head_mon.count, dtype=np.int32)
        results.append(counts)

    head_counts = np.stack(results, axis=0)
    preds = np.argmax(head_counts, axis=1)
    return head_counts, preds


# ========================================================================
# Main
# ========================================================================

def main():
    print("=" * 90)
    print("FULL NETWORK VERIFICATION: PyTorch vs raw Brian2 IF_curr_exp")
    print("=" * 90)

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
    print(f"   Total input spikes: {total_spikes}")

    # Run PyTorch sim
    print("\n5. Running PyTorch discrete-time simulator...")
    pt_counts, pt_preds = run_pytorch_sim(
        features, spike_data, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
        scale=10.0, duration_ms=100.0
    )
    print("   Done.")

    # Run Brian2 sim
    print("\n6. Running raw Brian2 simulator...")
    b2_counts, b2_preds = run_brian2_sim(
        spike_data, W_fc1, b_fc1, W_fc2, b_fc2, W_head, b_head,
        scale=10.0, duration_ms=100.0
    )
    print("   Done.")

    # Compare
    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Sample':>6} | {'Label':>5} | {'PyTorch pred':>12} | {'Brian2 pred':>11} | {'Match':>5} | {'Count diff':>10}")
    print("-" * 70)

    match_count = 0
    mismatches = []
    for s in range(len(test_y)):
        pt_c = pt_counts[s]
        b2_c = b2_counts[s]
        pt_p = pt_preds[s]
        b2_p = b2_preds[s]
        match = pt_p == b2_p
        if match:
            match_count += 1
        else:
            mismatches.append(s)
        count_diff = np.abs(pt_c - b2_c).sum()
        print(f"{s:6d} | {test_y[s]:5d} | {pt_p:12d} | {b2_p:11d} | {'Y' if match else 'N':>5} | {count_diff:10d}")

    print("-" * 70)
    print(f"\nPrediction agreement: {match_count}/{len(test_y)}")
    if mismatches:
        print(f"Mismatched samples: {mismatches}")
        for s in mismatches:
            print(f"  Sample {s} (label={test_y[s]}): PyTorch pred={pt_preds[s]}, counts={pt_counts[s]}; Brian2 pred={b2_preds[s]}, counts={b2_counts[s]}")
    else:
        print("All 20 samples produced identical argmax predictions.")

    print("=" * 90)


if __name__ == "__main__":
    main()

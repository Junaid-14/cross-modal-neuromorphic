#!/usr/bin/env python3
"""Debug script to understand why exact SpiNNaker simulation is silent."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

import pyNN.brian2 as sim
from brian2 import Synapses, nA, ms, defaultclock

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


def main():
    print("Loading checkpoint...")
    cp = torch.load("ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt", map_location="cpu")
    sd = cp["model_state_dict"]

    W_fc1 = sd["backbones.nmnist.fc1.weight"].numpy().astype(np.float32)
    b_fc1 = sd["backbones.nmnist.fc1.bias"].numpy().astype(np.float32)
    W_fc2 = sd["backbones.nmnist.fc2.weight"].numpy().astype(np.float32)
    b_fc2 = sd["backbones.nmnist.fc2.bias"].numpy().astype(np.float32)
    W_head = sd["heads.nmnist.1.weight"].numpy().astype(np.float32)
    b_head = sd["heads.nmnist.1.bias"].numpy().astype(np.float32)

    print(f"FC1: {W_fc1.shape}, bias range: [{b_fc1.min():.4f}, {b_fc1.max():.4f}]")
    print(f"FC2: {W_fc2.shape}, bias range: [{b_fc2.min():.4f}, {b_fc2.max():.4f}]")
    print(f"Head: {W_head.shape}, bias range: [{b_head.min():.4f}, {b_head.max():.4f}]")

    # Load data
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from Dataloaders.nmnist_loader import get_nmnist_loaders
    _, test_loader = get_nmnist_loaders(batch_size=1, num_workers=0)

    # Get first sample of each class for debugging
    samples = {}
    for x, y in test_loader:
        label = int(y.item())
        if label not in samples:
            samples[label] = x
        if len(samples) == 10:
            break

    # Extract features
    frontend = ConvFrontend()
    frontend.load_state_dict(
        {k.replace("backbones.nmnist.", ""): v for k, v in sd.items() if k.startswith("backbones.nmnist.") and ("conv" in k or "lif1" in k or "lif2" in k)},
        strict=False,
    )
    frontend.eval()

    features = {}
    with torch.no_grad():
        for label, x in samples.items():
            features[label] = frontend(x).numpy()[0]

    # Build network with SCALE=2.0 using PyNN native projections
    scale = 2.0
    thresh = 0.001
    duration = 100.0
    rate_scale = 10.0

    sim.setup(timestep=1.0, min_delay=1.0)
    print(f"\nBrian2 defaultclock.dt = {defaultclock.dt}")

    # Input
    input_pop = sim.Population(1600, sim.SpikeSourceArray(spike_times=[]))

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

    # Projections using PyNN native API
    def make_proj(pre, post, W, b):
        conn = []
        for post_i in range(W.shape[0]):
            for pre_i in range(W.shape[1]):
                w = float(W[post_i, pre_i]) * scale
                if abs(w) > thresh:
                    conn.append((pre_i, post_i, w, 1.0))
        print(f"  Connections: {len(conn)} / {W.size}")
        if conn:
            proj = sim.Projection(pre, post, sim.FromListConnector(conn), sim.StaticSynapse(), receptor_type="excitatory")
            print(f"  Projection created: {proj}")
        # Bias
        for i in range(W.shape[0]):
            post[i:i+1].set(i_offset=float(b[i]) * 10.0)
        print(f"  Bias i_offset range: [{b.min()*10:.4f}, {b.max()*10:.4f}] nA")

    print("\nBuilding FC1 projection...")
    make_proj(input_pop, fc1_pop, W_fc1, b_fc1)

    print("\nBuilding FC2 projection...")
    make_proj(fc1_pop, fc2_pop, W_fc2, b_fc2)

    print("\nBuilding Head projection...")
    make_proj(fc2_pop, head_pop, W_head, b_head)

    # Run one sample
    label = 0
    feat = features[label]
    print(f"\n{'='*60}")
    print(f"Running sample: label={label}")
    print(f"Feature stats: nonzero={np.count_nonzero(feat)}, sum={feat.sum():.1f}, max={feat.max():.1f}")

    rates = np.clip(feat * rate_scale, 0.0, None)
    n_steps = int(duration)
    rng = np.random.default_rng(42)
    input_spikes = rng.poisson(rates / 1000.0, size=(n_steps, 1600)) > 0

    indices = []
    times = []
    for nid in range(1600):
        ts = np.where(input_spikes[:, nid])[0]
        for t in ts:
            indices.append(nid)
            times.append(float(t))

    print(f"Total input spikes: {len(times)}")

    input_pop.brian2_group.set_spikes(np.array(indices), np.array(times) * ms)

    sim.run(duration)

    # Get spike counts
    def get_counts(pop, name):
        data = pop.get_data("spikes")
        counts = [len(st) for st in data.segments[-1].spiketrains]
        print(f"  {name}: total_spikes={sum(counts)}, per_neuron={counts}")
        return counts

    fc1_counts = get_counts(fc1_pop, "FC1")
    fc2_counts = get_counts(fc2_pop, "FC2")
    head_counts = get_counts(head_pop, "Head")

    if sum(head_counts) > 0:
        pred = int(np.argmax(head_counts))
        print(f"  Prediction: {pred} (correct={pred==label})")
    else:
        print(f"  Prediction: NONE (silent)")

    sim.end()
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

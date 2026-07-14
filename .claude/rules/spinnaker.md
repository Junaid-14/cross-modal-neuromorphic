---
paths:
  - "code/*spinnaker*"
  - "code/*pynn*"
  - "docs/SpiNNaker*"
  - "docs/HARDWARE*"
  - "reports/**"
---

# SpiNNaker port — the constraints are binding, not abstract

Source of truth: `docs/SpiNNaker_Port_Plan.md` (v1.0) and
`docs/HARDWARE_ACCESS_CHECKLIST.md`. This file is the summary you must not get wrong.

## The one thing people forget

**SpiNNaker runs PyNN, not PyTorch.** There is no backprop, no AdamW, no autograd. The
model must be *reimplemented*, not exported. snnTorch and sPyNNaker have different neuron
semantics — **parameters do not transfer automatically.** Flag every portability gap
rather than assuming it closes.

## Hardware budget

| | SpiNN-3 | SpiNN-5 |
|---|---|---|
| Chips | 4 | 48 |
| Usable cores | **72** | **~864** |

- ARM968E-S @ ~200 MHz, **32 KB DTCM + 64 KB ITCM per core**, 1 ms timestep.
- FC1 alone: 1,638,400 weights × 4 B = 6.55 MB → **~205 cores**. (The port plan's §8
  self-review corrected an earlier estimate that was **4× too low** — do not resurrect it.)
- **Full M7 ≈ 600 cores.** SpiNN-3 (72 cores) is roughly **8× short** and cannot host the
  model without compressing 512→256 hidden.

FC1's fan-in was contested (2-conv vs 3-conv) until 2026-07-14. **Resolved: it is 2-conv**,
`fc1 = Linear(1600, 1024)` = 1,638,400 weights — which is exactly what the ~205-core figure
assumes, so the budget is sound. `EXPERIMENTAL_MANIFEST.md` had claimed 3-conv and was wrong.

## Component mapping

| PyTorch / snnTorch | PyNN / sPyNNaker | Risk |
|---|---|---|
| `snn.Leaky(beta=0.9)` | `IF_curr_exp(tau_m ≈ 10 ms)` via **β = exp(−dt/τ_m)** | low |
| `nn.Linear` | `Projection` + `AllToAllConnector` | low |
| `nn.Conv2d` | `ConvolutionConnector` | **medium** (R1) |
| `MaxPool` | `PoolingConnector` / subsample | medium |
| `LayerNorm` | **not native** — omit or approximate | high |
| `Softmax` | **not native** — WTA circuit or rate-decode | high |
| `ModernHopfieldLayer` | rate-based WTA (~20 cores) | medium |
| `ImprovedHGRNGate` — `r * h_prev` | **NOT EXPRESSIBLE.** Elementwise multiplication of spike trains is unsupported | **highest (R3)** |
| `SupervisedContrastiveLoss` | **omit on-chip** — train off-chip | n/a |
| Replay buffer (~4 MB) | **host-mediated** — exceeds per-core memory ~100× | n/a |

**R3 is the one that matters for the paper.** If HGRN's multiplicative reset gate cannot be
realised, the adaptive-gate story — the core novelty claim — changes. Do not quietly
substitute an additive gate and call it "the same architecture."

## The three phases (and what "success" means for each)

| Phase | Weeks | Goal | Success criterion |
|---|---|---|---|
| 1 — inference only | 1–3 | Export weights → PyNN | within **±1 %** of the PyTorch baseline on all 3 test sets |
| 2 — host-mediated CL | 4–6 | Host computes updates, injects into live sim | forgetting / FWT within **±2 pp** of baseline |
| 3 — on-chip STDP | 7–10 (stretch) | Neuromodulated STDP | *"any non-random accuracy"* — deliberately modest |

## Say what actually happened

- Nothing has run on a board. `docs/HARDWARE_ACCESS_CHECKLIST.md` has **every box
  unchecked** — hardware access is not confirmed.
- The committed `reports/` directory contains sPyNNaker provenance from local runs,
  **including one that errored during `setup()`**. It is not evidence of a working port.
- Permitted claim if only the simulator runs: *"Validated on the sPyNNaker simulator, ready
  for hardware deployment"* — with **no energy numbers**, because a simulator cannot
  produce them.
- **Mapping ≠ inference ≠ real hardware.** These are three different claims. Never collapse
  them.

Software: `sPyNNaker == 7.0.0`, Python 3.8–3.11.

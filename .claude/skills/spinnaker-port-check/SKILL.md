---
description: Check whether a PyTorch/snnTorch component can actually run on SpiNNaker — its PyNN equivalent, its core and DTCM cost, and whether it is expressible at all. Use when working on the hardware port, writing PyNN/sPyNNaker code, estimating whether the model fits on a board, or whenever someone assumes a layer "should map fine".
argument-hint: "<component, e.g. ImprovedHGRNGate or 'the whole model'>"
allowed-tools: Bash(python3 *), Bash(grep *), Read, Grep, Glob
---

# SpiNNaker port check: `$ARGUMENTS`

## The premise

**SpiNNaker runs PyNN, not PyTorch.** No backprop, no AdamW, no autograd. Nothing is
"exported" — it is **reimplemented**. snnTorch and sPyNNaker have different neuron
semantics, so **parameters do not transfer automatically**. Every mapping is a hypothesis
until it has been demonstrated.

Source of truth: `docs/SpiNNaker_Port_Plan.md`, `docs/HARDWARE_ACCESS_CHECKLIST.md`.

## The architecture (settled — but re-verify before you budget)

**Backbone: RESOLVED 2026-07-14 — it is 2 conv layers.** Read from source:
`conv1(2->32, k5) -> MaxPool -> conv2(32->64, k5) -> MaxPool -> fc1(1600->1024) -> fc2(1024->512)`.
`EXPERIMENTAL_MANIFEST.md` claimed 3-conv and was wrong (now corrected); the deep-dive and the
port plan were right. **FC1 = 1600 x 1024 = 1,638,400 weights** -- exactly the number the
~205-core budget rests on, so that budget is sound.

```bash
grep -n "Conv2d\|class SNN_Backbone" code/m7_trimodal_run.py   # confirm it hasn't drifted
```

Always state which backbone you assumed. A capacity estimate built on the wrong one is not a
conservative estimate — it's a wrong one.

## Budget

| | SpiNN-3 | SpiNN-5 |
|---|---|---|
| Usable cores | **72** | **~864** |

ARM968E-S @ ~200 MHz · **32 KB DTCM + 64 KB ITCM per core** · 1 ms timestep.

- **FC1 alone ≈ 205 cores** (1,638,400 weights × 4 B = 6.55 MB). The port plan's §8
  self-review corrected an earlier estimate that was **4× too low** — do not resurrect it.
- **Full M7 ≈ 600 cores → SpiNN-5 required.** SpiNN-3 is ~8× short and cannot host the model
  without compressing 512 → 256 hidden.

## Mapping table

| PyTorch / snnTorch | PyNN / sPyNNaker | Risk |
|---|---|---|
| `snn.Leaky(beta=0.9)` | `IF_curr_exp(tau_m ≈ 10 ms)` — **β = exp(−dt/τ_m)** | low |
| `nn.Linear` | `Projection` + `AllToAllConnector` | low |
| `nn.Conv2d` | `ConvolutionConnector` | **medium (R1)** |
| `MaxPool` | `PoolingConnector` / subsample | medium |
| `LayerNorm` | **not native** — omit or approximate | high |
| `Softmax` (the adaptive gate!) | **not native** — WTA or rate-decode | high |
| `ModernHopfieldLayer` | rate-based WTA (~20 cores) | medium |
| `ImprovedHGRNGate` — `r * h_prev` | **NOT EXPRESSIBLE.** Elementwise multiplication of spike trains is unsupported | **highest (R3)** |
| `SupervisedContrastiveLoss` | **omit on-chip** — train off-chip | n/a |
| Replay buffer (~4 MB) | **host-mediated** — exceeds per-core memory ~100× | n/a |

## R3 is the one that can cost you the paper

If HGRN's multiplicative reset gate cannot be realised on-chip, the **adaptive-gate story —
the core novelty claim — changes.** Do not quietly substitute an additive gate and call it
"the same architecture." If you propose a substitution, state plainly **which claim it
invalidates** and what would have to be re-run.

## Say only what is true

- **Nothing has ever run on a board.** `HARDWARE_ACCESS_CHECKLIST.md` has every box unchecked.
- `reports/` holds sPyNNaker provenance from local runs — **one of which errored during
  `setup()`**. It is not evidence of a working port.
- Simulator-only ⇒ the permitted claim is *"validated on the sPyNNaker simulator, ready for
  hardware deployment"* — **with no energy numbers.** A simulator cannot produce them.
- **Mapping ≠ inference ≠ real hardware.** Three separate claims. Never collapse them.
- `code/test_spinnaker_*.py` **does not exist** (gitignored). Don't tell anyone to run it.

## Output

Per component: the PyNN construct, core + DTCM cost, risk, and — if it doesn't map — what
breaks and **which paper claim depends on it**. Give a running core total against 72
(SpiNN-3) and ~864 (SpiNN-5). State your assumptions, especially the backbone depth.

For a full component-by-component audit, use `.claude/workflows/spinnaker-feasibility-audit.js`.

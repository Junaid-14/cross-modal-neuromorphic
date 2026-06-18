# Session Checkpoint — 2026-06-07

> Save point before PC restart. Covers the unfrozen-backbone + raw-input rehearsal experiment.

---

## 1. Experiment Run: `C_rehearsal` (seed 42)

**Script:** `code/run_cl_rehearsal.py`
**Output dir:** `results/cl_rehearsal/`
**Checkpoints:** `results/cl_rehearsal/checkpoints/C_rehearsal_seed42/`

### What it tests
Unfrozen backbone + raw-input rehearsal (replay_weight=1.0).
Sequence: T1 (N-MNIST) → cross-modal alignment (+ T1 rehearsal) → T2 (SHD, + T1 rehearsal) → T3 (DVS, + T1+T2 rehearsal).

### Results (extracted from log + transfer eval)

| Metric | Value |
|--------|-------|
| T1 after T1 | 97.50% |
| T1 after T2 | 95.81% |
| T1 after T3 | **95.65%** |
| T2 after T2 | 72.26% |
| T2 after T3 | 73.14% |
| T3 after T3 | 76.89% |
| **T1 forgetting** | **1.85%** |
| V2A transfer | **14.40%** |
| A2V transfer | **10.30%** |
| Prototype alignment | 0.183 |
| Silhouette (visual) | 0.321 |
| Silhouette (audio) | 0.061 |

**CSV:** `results/cl_rehearsal/cl_rehearsal_results.csv`

### Key finding
Raw-input rehearsal **completely prevents catastrophic forgetting** but **collapses cross-modal transfer** to the same level as frozen backbones (~14%).

The visual backbone must drift toward audio during alignment to build a meaningful shared subspace. Rehearsal pins it to old-task data, preserving old accuracy at the cost of transfer.

---

## 2. Bug Fix Applied

**File:** `code/run_cl_rehearsal.py` (line ~324)
**Bug:** `.norm()` called on a Long tensor (BatchNorm `num_batches_tracked`) → `RuntimeError`
**Fix:** Added `.is_floating_point()` guard:
```python
if k.startswith("backbones.nmnist") and sd_t1[k].is_floating_point():
```

---

## 3. Helper Script Created

**File:** `code/run_rehearsal_transfer_eval_only.py`
Loads the final T3 checkpoint and runs `evaluate_cross_modal_transfer()` + writes CSV. Useful if you need to re-extract metrics without rerunning training.

---

## 4. Comparative Summary (all conditions, seed 42)

| Condition | T1 post-T3 | V2A | Notes |
|-----------|-----------|-----|-------|
| Unfrozen, no rehearsal | ~60% | 36.9% | Strong transfer, catastrophic forgetting |
| Frozen backbone | 13.3% | 14.8% | No transfer, still forgets via shared drift |
| **Unfrozen + rehearsal** | **95.7%** | **14.4%** | Zero forgetting, no transfer |

---

## 5. Open Questions / Next Steps

- Is there a middle ground? (e.g., replay_weight < 1.0, or only rehearse during alignment not T2/T3)
- Can we decouple backbone drift from shared-subspace alignment? (e.g., separate alignment networks, adversarial constraints)
- Should we try rehearsal with frozen backbones to see if shared-component drift alone is the culprit?
- The orthogonal penalty in alignment had insane values in epochs 1-2 (`orth=2.9e8` then `5.4e6`) before settling. Worth investigating if this is numerically stable.

---

*Session ended: PC restart requested.*

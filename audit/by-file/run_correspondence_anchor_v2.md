# Audit — `code/run_correspondence_anchor_v2.py` (749 lines)

Cluster **E-alignment** · shard `w3_algn` · IDs **ALGN-001 … ALGN-014**

## What this file is

"Cross-Modal Correspondence v2 — Hardened Discovery Claim." It loads a frozen
tri-modal checkpoint, trains small projection heads (MLP or linear) that map the
frozen N-MNIST (visual) and SHD (audio) backbone features onto a shared set of
QR-orthonormal anchors, and asks whether digit *identity* transfers across modality.
It runs three sub-configs per seed — a main 0-9 config, a held-out 0-5→6-9 config,
and a label-shuffled control — plus a parameter-free Procrustes corroboration, over
8 seeds × {mlp, linear}. Output goes to `results/hypergraph_anchor_v2/`.

## Safety posture (good news first)

- Has a proper `if __name__ == "__main__"` guard (line 748); nothing runs on import.
- Imports only the **import-safe** `m7_trimodal_run_provenance` (guarded) and
  `utils.cross_modal_transfer`. It does **not** touch `code/m7_trimodal_run.py`.
- The invariant asserts in `run_one_seed` (560-566) are genuinely good practice: they
  hash the frozen backbones/heads and assert accuracy is unchanged after training the
  projection heads.

The problems are logic, provenance, and missing-artifact issues, not import-time hazards.

## Critical

### ALGN-001 — the held-out training path is doubly broken
`train_correspondence` builds `ce = nn.CrossEntropyLoss()` (line 287, default
`reduction="mean"`). In `run_one_seed` the held-out model is built with
`num_anchors=6` (line 575), so `model.anchors` is `(6, anchor_dim)` and
`logits_vis/logits_aud` have **6 columns** (315-316). But the training set from
`build_paired_feature_dataset` contains **labels 0-9** (it loops `range(10)`,
251/261). So `ce(6-col logits, target=9)` raises `RuntimeError: Target out of bounds`
and the held-out training crashes on **every** seed.

Even if it did not crash, the masking is a **no-op**: `loss_vis`/`loss_aud` are already
scalars (mean over the batch) *before* the mask is applied, so

```
loss = loss_vis + loss_aud          # scalar
loss = loss * mask.float()          # scalar broadcast to (B,)
loss = loss.sum() / mask.sum()      # collapses back to the same scalar
```

exclusion of classes 6-9 never happens. `main()` wraps `run_one_seed` in a
`try/except` (669-672), so the crash is swallowed with a red ❌ and **no per-seed CSV
is ever written**. The script's central "hardened discovery" deliverable — held-out
6-9 retrieval — cannot run in this code. Any held-out-retrieval-at-chance number in the
paper did not come from this path.

**Fix:** filter `logits` and `labels` by the keep-mask *before* `ce` (or use
`reduction="none"` and average only kept per-sample losses); train the held-out config
on 0-5 samples so targets fit the 6 anchors. That single fix removes both the crash and
the no-op.

## High

### ALGN-002 — `CKPT_SOURCE` does not exist
Line 50 hardcodes
`results/cl_confirmation/checkpoints/A_no_align_seed42/final_shd_shd.pt`. There is **no
`results/` tree** in the repo, so `torch.load(CKPT_SOURCE)` (533) fails before any work.
That checkpoint is meant to be produced by the `cl_confirmation` pipeline, which is only
reachable through a broken archived import (ALGN-027/ALGN-029).

### ALGN-003 — feature cache does not exist
`_load_cached_features` (147-157, called at 645) requires four
`results/hypergraph_anchor/cache/*_features.pt` files and otherwise raises
`FileNotFoundError("Run v1 first…")`. `results/` is absent and "v1" is
`code/archive/run_correspondence_anchor.py` (archived, off `sys.path`), so the documented
recovery is itself broken.

### ALGN-004 — hardcoded baseline table printed as results (RULE 1)
Lines 716-718 print a "BASELINE COMPARISON" with six literal numbers
(`~97.0`, `~13.3`, `13.3`, `14.8`, `95.7`, `14.4`) directly beneath the freshly-computed
Correspondence row (715). They read as comparable measured results but trace to no CSV.
This is a direct RULE 1 violation.

### ALGN-005 — `sklearn` missing from `requirements.txt` (whole cluster)
All four cluster files import `utils.cross_modal_transfer`, which imports
`sklearn.{linear_model,metrics,metrics.pairwise}` (utils/cross_modal_transfer.py:34-36).
`code/requirements.txt` has no scikit-learn pin, so a clean env → `ImportError` on every
one of these scripts. (`seaborn` is *not* actually imported here; sklearn is the real gap.)

## Medium

- **ALGN-006 (dead-code):** `_extract_backbone_features` (120-144) is never called — a
  verbatim clone of the archived v1. Delete it.
- **ALGN-007 (duplication):** `_get_backbone_features` (190-207) hand-reimplements the
  audio ModuleDict forward, duplicating `M7_ContinualAdaptiveModel._get_features`
  (provenance 343-360). Verified numerically consistent *today* (audio backbone is
  `fc1→1024, fc2→1024, fc3→512` with `lif1-3`), but it will silently drift if the model
  changes. Call the model's own feature method.
- **ALGN-008 (architecture):** cluster-wide helper duplication — `compute_held_out_metrics`
  ×3 (v2:372, `run_joint_magnitude_ablation.py:572`, `run_joint_shared_encoder.py:653`),
  `_hash_params` ×2, `build_paired_feature_dataset`/`train_correspondence` ×2 (v2+archive),
  `run_procrustes` in v2 and `run_joint_shared_encoder`, `_events_to_dense` in ~8 files.
- **ALGN-009 (naming):** module-level `run_one_seed(seed, cfg, loaders, cached, device,
  head_type)` (524) collides with `m7_trimodal_run_provenance.run_one_seed(cfg,
  data_loaders, results_dir, ckpt_dir)` and yet another in `run_ordinal_anchor`. Three
  same-named functions, three contracts.

## Low / info

- **ALGN-010 (quality):** unused `List` import (23).
- **ALGN-011 (correctness):** `_events_to_dense` (93-103) has a dead `else` and divides by
  zero if all events are at `t==0`.
- **ALGN-012 (safety):** `torch.load` on disk paths without `weights_only=` (155, 533).
  Mitigated by torch 2.11's default, but make it explicit.
- **ALGN-013 (correctness/info):** retrieval candidate sets differ — 4-way in
  `compute_held_out_metrics` (chance 0.25) vs 10-way in `run_procrustes` (chance 0.1) —
  yet reported side by side; verdict threshold `0.25 + 0.1` is a magic number and the
  4-class retrieval statistic is quantised to {0,.25,.5,.75,1}.
- **ALGN-014 (research-integrity/info):** the per-seed CSV (595-615) records accuracies
  and retrieval % but never spike-rate / sparsity — "accuracy never travels alone."

## Metric-correctness notes (what is actually right)

- Held-out metrics and Procrustes read from `cached["*_test"]` — **test**, not
  validation. No val/test confusion here.
- `evaluate_cross_modal_transfer` (in `utils/`) is a legitimate linear-probe transfer
  protocol on **test** loaders; the prototype cosine alignment and silhouette are
  standard. The concern is the missing sklearn pin (ALGN-005), not the math.
- The QR-anchor construction is seeded (`manual_seed(42)`, 182) and registered as a
  buffer, and the asserts confirm anchors don't move — solid.

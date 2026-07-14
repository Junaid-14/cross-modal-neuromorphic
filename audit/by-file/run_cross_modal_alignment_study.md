# Audit — `code/run_cross_modal_alignment_study.py` (534 lines)

Cluster **E-alignment** · shard `w3_algn` · IDs **ALGN-021 … ALGN-026**

## What this file is

The full tri-modal (N-MNIST → SHD → DVS) alignment study across three conditions:
**A** (no alignment), **B** (full 512-d shared space with cross-modal SupCon), and
**C** (128-d shared + 384-d private with an orthogonal penalty). For each seed it trains
T1/T2/T3 with per-step checkpointing, runs the Paper-1 cross-modal transfer eval after
T1, after the alignment phase, and after T2, then aggregates mean ± std / 95% CI and
paired Wilcoxon tests (B-vs-A, C-vs-A). Output: `results/cross_modal_alignment/`.

## Safety posture

- Has a `__main__` guard (533); imports the guarded provenance module. No import-time run.
- Uses **test** accuracy for T1/T2/T3 (`trainer.evaluate(..., ["test"], ...)`, e.g. line
  290) — RULE 4 compliant, unlike `run_digit_only_alignment.py`.
- Running it still needs the ~23 GB datasets (via `build_data_loaders`), which are absent;
  that is expected for a full run, not a code defect.

## High

### ALGN-021 — checkpoint-skip staleness trap (T1/T2/T3)
Each stage skips training whenever `{condition}_seed{seed}_T{n}.pt` exists, loading it
with only a `⏩ Loading checkpoint` print (212-214, 276-278, 323-325). This is the
documented repo-wide trap: a checkpoint from older code silently yields stale results
presented as fresh, with no code-version/hash guard. Combined with ALGN-022, a stale run
can enter the summary unnoticed.

## Medium

### ALGN-022 — `master_results.csv` accumulates duplicates; summary n/std/CI corrupted
`_save_seed_result` (68-83) appends each row to `master_results.csv`
(read → concat → rewrite) with **no dedup** on `(seed, condition)`.
`_aggregate_alignment_results` then computes `n = len(sub)`, `std(ddof=1)` and 95% CI over
all rows (99-118), so a rerun inflates `n` and shrinks/skews the CI — the same class of
bug flagged for `generate_figures_3seed`. It is **asymmetric**: the Wilcoxon pivot uses
`aggfunc="first"` (131) so the significance test dedups, but the summary statistics do
not. Fix: `drop_duplicates` on `(seed, condition)` (keep last) before the summary loop.

### ALGN-023 — condition-A silhouettes hardcoded to 0.0 and fed into the summary
For the no-align branch, `transfer_after_align` is fabricated: every key is copied from
`transfer_after_t1` **except** `sil_visual`/`sil_audio`, which are forced to `0.0`
(260-265). So `ALIGN_sil_vis`/`ALIGN_sil_aud` (388-389) are `0.0` for every A seed, and
those columns are in the aggregation list (522) → condition A summarizes as
"after-alignment silhouette = 0.0 ± 0.0". That reads as a *measured* zero cluster
separation but is a placeholder (A has no alignment phase). The other `ALIGN_*` values
for A are copies of the T1 numbers, not independent measurements.

### ALGN-024 — `random_init_baseline` recomputed per condition × seed
`_run_alignment_seed` recomputes the random-init baseline for all three tasks at the start
of every condition-seed (201-206). The RI baseline is condition-independent; with 3
conditions × 5 seeds that is 135 baseline evaluations (3 random models × 3 tasks × 15)
where 3 would do. On MPS with DVS-Gesture this is a large redundant cost. Compute once in
`main()` and pass in.

## Low

- **ALGN-025 (research-integrity):** `base_cfg` sets `conditions=["no_replay","replay"]`
  (442) — a copy-paste leftover from the replay CL study — but this is an A/B/C study.
  `base_cfg.to_json("master_config.json")` (528) therefore persists a `conditions` field
  that contradicts the actual run, poisoning provenance for anyone trusting the JSON.
- **ALGN-026 (quality):** unused imports `copy` (32), `os` (36), `Dict` (38). Also the
  seed set here is `[42, 123, 456, 789, 101112]` (413), where `101112` looks like a typo
  divergence from the `1011` used in `run_ordinal_anchor` and `run_digit_only_alignment`.

## Shared findings that also apply here

- **sklearn missing from requirements** (ALGN-005) — imports `utils.cross_modal_transfer`
  at line 61.
- **Accuracy without spike-rate/sparsity cost** (ALGN-014) — the flattened result dict
  (355-397) records accuracies, forgetting, FWT and transfer % but no SNN cost metrics.
- **`torch.load` without `weights_only`** (ALGN-012) — lines 214, 278, 325.

## Metric-correctness notes (what is right)

- FWT (`acc_t2_zero_shot - ri_baselines[task2]`, 297) and forgetting deltas (373-375) are
  defined consistently with the project's CL conventions.
- Zero-shot is evaluated on the **test** split (271, 318); post-training accuracies are
  **test** (290, 337). Transfer evals use test loaders. No val/test confusion.
- The Wilcoxon guard for the all-zero-difference case (149-158) is handled explicitly
  rather than letting `scipy` raise — good.

# Audit — `code/run_cl_confirmation_fixed.py` (408 lines)

Cluster **F-cl-eval** · shard `w5_cle` · prefix `CLE`

## Narrative

The "confirmation fix" trainer: one full continual-learning run of `C_split_128` seed 42 with
**frozen old-task backbones/heads** — T1 (N-MNIST) → cross-modal alignment → T2 (SHD) → T3
(DVS) — writing to `results/cl_confirmation_fixed/`. It is the more carefully engineered of
the two run scripts: it has a `main()` + `__main__` guard, an atomic CSV write
(`_atomic_write_csv`, 63-67), a resume check (`_is_complete_result`), an `EXPECTED_COLUMNS`
schema, and it correctly reports **test** accuracy via `trainer.evaluate(..., ["test"])`
(RULE 4 satisfied). Unlike the eval-only script, its result CSV is entirely live — no
hardcoded accuracies.

The concerns are provenance and metric-semantics rather than fabrication:

- The saved config **misrepresents the training length**: `cfg.max_epochs=50, patience=12`
  (368) but every `train_task` call hardcodes `num_epochs=15, patience=5` (226/266/302).
  `cfg.to_json` persists 50/12 alongside each checkpoint, so reproducing from the saved config
  would not reproduce the checkpoints (CLE-020).
- The checkpoint-skip-on-exists pattern (220-222, 260-262, 296-298) is the documented repo
  staleness trap, replicated per task: a stale `.pt` yields stale results that look fresh
  (CLE-029).
- The forgetting metrics have a naming trap: `forgetting_T1_T3` (327) is actually the
  **incremental** post-T2→post-T3 drop, while `total_forgetting_T1_T3` (329) is the cumulative
  one — easy to quote the wrong one in the paper (CLE-021). And `forgetting_T2_T3` and
  `total_forgetting_T2_T3` (328, 330) are computed by identical expressions, so they are
  always equal — a redundant column and asymmetric with the T1 pair (CLE-022).
- The result CSV carries accuracy/forgetting but no spike-rate/sparsity cost (CLE-024).

`build_full_loaders` (69-179) is a ~110-line near-clone of the same function in
`run_cl_rehearsal.py` (CLE-030), and the `ExperimentConfig` literal is one of three copies
(CLE-032).

## Findings by severity

### MEDIUM
- **CLE-020** (224-231, 264-271, 300-307 vs 368, research-integrity) — **Saved config
  misrepresents actual training length.** `cfg.max_epochs=50/patience=12` but training uses
  hardcoded `num_epochs=15/patience=5`; the persisted config (`to_json` at 231/271/307 and
  `master_config.json` at 402) records 50/12. Reproduction defect. Drive epochs from cfg.
  Same pattern in `run_cl_rehearsal.py`.
- **CLE-021** (326-329, correctness/naming) — **`forgetting_T1_T3` is the incremental
  post-T2→post-T3 drop, not the total** the name implies (total is a separate column).
  Mislabel risk when a number is lifted into the paper. Rename / add a data dictionary.
- **CLE-024** (332-347, research-integrity) — **Results CSV stores accuracy/FWT/forgetting
  with no spike-rate/sparsity cost.** Only `time_steps` survives, and only via the separately
  saved config. Add measured spike-rate/sparsity columns. Same for `run_cl_rehearsal.py`.
- **CLE-029** (220-222, 260-262, 296-298, safety) — **Per-task checkpoint-skip silently loads
  stale weights.** `if ckpt.exists(): skip training; load_state_dict(...)` — the documented
  repo trap, per task, warning only via a transient print. Gate on a version/config hash or an
  explicit `--resume`. Same in `run_cl_rehearsal.py`.

### LOW
- **CLE-022** (328, 330, duplication) — `forgetting_T2_T3` and `total_forgetting_T2_T3` are
  identical expressions → always-equal redundant columns. Drop one or compute the intended
  distinct quantity.
- **CLE-023** (53-60, correctness) — `_is_complete_result` checks only column presence, not
  row count; a header-only/truncated CSV counts as "complete" and the cell is silently skipped
  (186). Also require `len(df) >= 1`.
- **CLE-019** (88-95 etc. vs 371, quality) — `cfg.pin_memory=True` and `cfg.num_workers` are
  never passed to the DataLoaders (hardcoded `num_workers=0`, no `pin_memory`), so the saved
  config advertises settings the run did not use. (num_workers=0 itself is correct on macOS.)
- **CLE-017** (10, dead-code) — `import dataclasses` unused.
- **CLE-023 / CLE-031** (105, dead-code) — dead `else 1` in `_events_to_dense`.
- **CLE-030** (69-179, duplication) — `build_full_loaders` near-identical to
  `run_cl_rehearsal.py:47-157`; extract to a shared module.
- **CLE-032** (364-377, duplication) — `ExperimentConfig` literal triplicated across the
  run/eval scripts.

## Cross-checks performed
- Confirmed `__main__` guard + `main()`; atomic write via `.csv.tmp` + `rename` (63-67).
- Confirmed reported accuracy uses the **test** loader (233, 273-277, 312-315) — RULE 4 OK.
- Confirmed the seed is set for both torch and numpy (189-190); split generator fixed at
  `manual_seed(42)` per repo policy (splits fixed across seeds).
- Confirmed `cfg.max_epochs/patience` (50/12) never reach `train_task` (grep: only the cfg
  literal has 50/12; all call sites use 15/5).
- Confirmed `dataclasses` has zero `dataclasses.*` uses.
- Confirmed the result CSV columns are all populated from live variables — **no hardcoded
  accuracy literals in this file.**
- Confirmed imported provenance symbols (`random_init_baseline`, `print_task_sanity`,
  `ContinualLearner`, etc.) exist in `m7_trimodal_run_provenance.py`.

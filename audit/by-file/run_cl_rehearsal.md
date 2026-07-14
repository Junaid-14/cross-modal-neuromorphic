# Audit — `code/run_cl_rehearsal.py` (452 lines)

Cluster **F-cl-eval** · shard `w5_cle` · prefix `CLE`

## Narrative

The rehearsal trainer: the same `C_split_128` seed-42 sequence as
`run_cl_confirmation_fixed.py` but with an **unfrozen backbone + raw-input rehearsal**
(`RawInputReplayBuffer`, `raw_replay_weight=1.0`), testing whether rehearsal recovers
cross-modal transfer without catastrophic forgetting. It is properly guarded (`main()` +
`__main__`), reports **test** accuracy (RULE 4 OK), and — importantly — writes an **entirely
live** result CSV (341-356) and computes its final transfer eval live (411-427). Its backbone
drift sanity check (318-333) is a nice touch.

The headline problem is the **COMPARISON TABLE printed at the end** (433-444). It answers the
assignment's question "is any other script writing hardcoded numbers to a CSV/log?" — **yes**:
the `A_no_align`, `C_unfrozen` and `C_frozen` columns are hardcoded float literals (e.g.
`97.21`, `54.85`, `84.20`, `36.92`, `0.59`, `0.21`), printed to stdout interleaved with the
live `res[...]`/`xfer[...]` values in identical column formatting. Under RULE 1 these are
unsourced numbers with no CSV backing, and a reader copying this table into the paper cannot
tell them apart from the measured C_rehearsal column (CLE-025). It is a log rather than a CSV,
so it is **high** rather than critical — but it is the same class of hazard as the fabricated
CSV in `run_rehearsal_transfer_eval_only.py`.

The remaining issues are the ones shared with the confirmation script (checkpoint-skip
staleness, config/epoch provenance mismatch, forgetting-metric naming/duplication,
accuracy-without-cost), plus a few local magic numbers and a minor robustness gap in the drift
loop.

## Findings by severity

### HIGH
- **CLE-025** (433-444, research-integrity) — **Hardcoded comparison-table numbers printed to
  the log alongside live results.** The `A_no_align`/`C_unfrozen`/`C_frozen` columns are float
  literals embedded in the print f-strings; only the C_rehearsal column and the transfer row
  are live. No CSV provenance; indistinguishable from measured values. Load those columns from
  their own result CSVs or print `n/a`/`unsourced`.

### MEDIUM
- **CLE-020** (199-207, 247-257, 289-299 vs 381, research-integrity) — Config/epoch provenance
  mismatch: `cfg.max_epochs=50/patience=12` but `train_task` uses hardcoded `15/5`; saved
  config records 50/12. (Shared with `run_cl_confirmation_fixed.py`.)
- **CLE-024** (341-356, research-integrity) — Result CSV stores accuracy/FWT/forgetting with
  no spike-rate/sparsity cost. (Shared.)
- **CLE-029** (195-197, 243-245, 285-287, safety) — Per-task checkpoint-skip silently loads
  stale weights instead of retraining. (Shared.)
- **CLE-033** (44, dependencies) — `evaluate_cross_modal_transfer` (imported line 44)
  transitively requires scikit-learn, absent from `code/requirements.txt`. (Shared.)

### LOW
- **CLE-021** (335-338, correctness/naming) — `forgetting_T1_T3` (336) is the incremental
  post-T2→post-T3 drop, not the total the name implies. (Shared.)
- **CLE-022** (337, 339, duplication) — `forgetting_T2_T3` and `total_forgetting_T2_T3`
  identical → always-equal redundant columns. (Shared.)
- **CLE-027** (178, 214, 275, 330, numerics) — Magic numbers: `RawInputReplayBuffer(
  samples_per_class=200)` (178) vs `cfg.replay_samples_per_class=50`; `max_batches=100`
  (214, 275); drift threshold `total_l2 < 1e-6` (330). Promote to named/config values;
  reconcile the 200-vs-50 discrepancy.
- **CLE-028** (324-328, correctness) — Backbone-drift loop indexes `sd_t3[k]` by `sd_t1`'s
  keys with no membership check → `KeyError` if key sets differ. `phase1_sanity_check
  .compare_state_dicts` (65-66) is the guarded pattern to mirror. (Confidence low.)
- **CLE-026** (15, dead-code) — `import dataclasses` unused.
- **CLE-031** (83, dead-code) — dead `else 1` in `_events_to_dense`.
- **CLE-030** (47-157, duplication) — `build_full_loaders` near-identical to the confirmation
  script; extract to a shared module.
- **CLE-032** (377-390, duplication) — `ExperimentConfig` literal triplicated.

## Cross-checks performed
- Confirmed `__main__` guard + `main()`.
- Confirmed reported accuracy uses the **test** loader (209, 259-263, 301-307) — RULE 4 OK.
- Confirmed the result CSV (341-356) is populated entirely from live variables — **no
  hardcoded accuracy literals written to the CSV** (contrast the stdout comparison table).
- Confirmed torch + numpy seeds set (163-164); split generator fixed at `manual_seed(42)`.
- Confirmed `cfg.max_epochs/patience` (50/12) never reach `train_task` (call sites use 15/5).
- Located the hardcoded stdout literals precisely at lines 435-444 via grep of the
  `:>14.2f`/`:>14.4f` format fields.
- Confirmed `RawInputReplayBuffer` exists in `m7_trimodal_run_provenance.py` (class at
  line 467) and `dataclasses` has zero `dataclasses.*` uses.

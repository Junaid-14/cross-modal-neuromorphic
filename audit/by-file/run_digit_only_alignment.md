# Audit — `code/run_digit_only_alignment.py` (678 lines)

Cluster **E-alignment** · shard `w3_algn` · IDs **ALGN-027 … ALGN-032**

## What this file is

The digit-only (N-MNIST + SHD, **DVS never instantiated**) version of the A/B/C alignment
ablation. Per `(condition, seed)` cell: train T1 (N-MNIST), optionally run the alignment
phase (B/C) with a post-align transfer eval, train T2 (SHD), then a post-T2 transfer eval.
It is **resumable** (skips complete cells) and uses **atomic writes** (temp file +
rename). It also carries a CLI: `--cell CONDITION SEED` runs one cell, `--confirm` is
meant to run the full T1→T2→T3 CL confirmation. Output:
`results/cross_modal_alignment_digit/`.

## Safety posture

- Has a `__main__` guard (607) with an `argparse` entrypoint; imports the guarded
  provenance module. No import-time run.
- The atomic per-cell CSV design (`_atomic_write_csv`, 197-201; one file per cell) means
  reruns **do not** accumulate duplicate rows — a genuine improvement over
  `run_cross_modal_alignment_study.py` (contrast ALGN-022).

## High

### ALGN-027 — `--confirm` imports a module that lives only in `code/archive/`
```
if args.confirm:
    import run_cl_confirmation      # line 616
    run_cl_confirmation.main()
```
`run_cl_confirmation.py` exists **only** at `code/archive/run_cl_confirmation.py`; it was
moved out of `code/`, and `archive/` is **not** on `sys.path` (only `REPO_ROOT/code` and
`REPO_ROOT` are inserted, 50-51). So `--confirm` raises `ModuleNotFoundError`. A
differently-named `code/run_cl_confirmation_fixed.py` exists but is not what the import
targets, and the file's own `run_full_cl_confirmation` (461) is never wired to the flag.
The documented `--confirm` path is dead. *(This is the pre-verified seed finding — the
line number is 616.)*

### ALGN-028 — SHD accuracy reported and Wilcoxon-tested on VALIDATION, never test (RULE 4)
`run_alignment_cell` records `T1_val_acc = t1_result["best_val"]` (277, 331) and
`SHD_val_acc = t2_result["best_val"]` (318, 333). `trainer.evaluate` on the **test** split
is used only for SHD zero-shot (308); after training, SHD accuracy is **never** evaluated
on test. `SHD_val_acc` then flows into `summary.csv` and is one of the paired-Wilcoxon
`test_metrics` (416). Per RULE 4 the SHD validation/test gap is **15-19 pp**, so the
headline retention metric *and* the significance test run on an inflated split. The
columns are honestly named `_val_` (mitigation), but validation is the only post-training
accuracy this script produces. (Provenance: `train_task` returns `{"best_val": …}` at
`m7_trimodal_run_provenance.py:673`, which is the best **validation** accuracy.)

## Medium

### ALGN-029 — `run_full_cl_confirmation` is orphaned yet is the cluster's checkpoint producer
`run_full_cl_confirmation` (461-505) is documented "NOT called by main() — invoke
explicitly" and is invoked by nothing (the `--confirm` flag imports the archived module
instead, ALGN-027). It writes `results/cl_confirmation/` checkpoints via the provenance
`run_one_seed` — which is exactly the `CKPT_SOURCE`
(`results/cl_confirmation/checkpoints/A_no_align_seed42/final_shd_shd.pt`) that
`run_correspondence_anchor_v2` and `run_ordinal_anchor` load (ALGN-002/ALGN-017). So the
entire cluster's data dependency rests on a function nothing calls, and the one CLI flag
that should call it is broken.

### ALGN-030 — couples to the legacy bi-modal `Dataloaders` package (boundary)
`build_digit_loaders` does `from Dataloaders.paired_digit_loader import
get_paired_digit_loaders` (152). `Dataloaders/` is the legacy Paper-2 package that
`CLAUDE.md` / `python-numerics.md` place outside the M7 reproducibility boundary. The
import resolves (the module exists), but it couples an M7-era study to legacy code, so a
change there silently affects these results.

## Low

- **ALGN-031 (correctness):** `aggregate_digit_results` silently drops any per-cell CSV
  missing an `EXPECTED_COLUMN` (368-374) — a stale-schema cell disappears from the summary
  and Wilcoxon and `n` shrinks with no visible signal. (Condition-A cells are safe: their
  `post_align_*` values are `NaN` but the columns still exist, so the membership check
  passes.) Surface skipped cells and the final per-condition `n`.
- **ALGN-032 (quality):** unused imports `copy` (31), `os` (34), `time` (36), and
  `List`/`Dict` (38).

## Shared findings that also apply here

- **sklearn missing from requirements** (ALGN-005) — imports `utils.cross_modal_transfer`
  at line 64.
- **Accuracy without spike-rate/sparsity cost** (ALGN-014) — the result dict (328-345)
  records val accuracies and transfer % with no SNN cost metrics.

## Metric-correctness notes

- `_events_to_dense` (122-132) is the same duplicated helper flagged in ALGN-008/ALGN-011
  (dead `else`, divide-by-zero if all events at `t==0`).
- The transfer eval itself (`_run_transfer_eval`, 219-234) runs on the **test** loaders
  (`data_loaders["nmnist"]["test"]`, `["shd"]["test"]`) — so the transfer metrics are on
  test even though the task-accuracy columns are on validation (ALGN-028). Worth keeping
  those two facts distinct when reading the CSV.
- Summary stats use `std(ddof=1)` and a 1.96·SE 95% CI (398-407); the CI is labeled and
  computed correctly given `n` (and `n` is not inflated here, unlike ALGN-022).

# Audit — `code/eval_transfer_fixed.py` (141 lines)

Cluster **F-cl-eval** · shard `w5_cle` · prefix `CLE`

## Narrative

An eval-only script that scores cross-modal transfer on the `C_split_128` seed-42 checkpoints
at two stages (`post_T2` = `final_shd_shd.pt`, `post_T3` = `final_dvs_dvs.pt`). Structurally
it is the healthy counterpart to `run_rehearsal_transfer_eval_only.py`: it **has a proper
`main()` + `__main__` guard**, it **loads its config from disk** rather than re-declaring it,
and its transfer metrics are all **computed live** — there are no fabricated numbers written
anywhere. The problems here are correctness/robustness on the empty-data path, a magic-number
verdict, and the shared missing-dependency and duplication issues.

The empty-path defect is the notable one. If the config loads but the checkpoints are missing
(exactly today's repo state for the checkpoints, though the config open fails first), the
`results` list stays empty, `pd.DataFrame([])` has no columns, and the verdict line

```
post_t2_v2a = df[df["stage"]=="post_T2"]["v2a"].values[0] if "post_T2" in df["stage"].values else 0
```

raises `KeyError: 'stage'` — because Python evaluates the ternary **condition**
(`df["stage"].values`) first, before the `else 0` can act. So the fallback that was clearly
meant to handle "no post_T2" never runs. A sibling defect: when `df` is non-empty but lacks a
`post_T2` row, the `else 0` prints "Transfer DID NOT survive. V2A = 0.0%" — rendering
*absence of data* as a *negative scientific result*.

## Findings by severity

### HIGH
- **CLE-008** (60-63, safety) — **Missing `config_nmnist.json` crashes `main()` immediately.**
  `CKPT_DIR = results/cl_confirmation_fixed/checkpoints/C_split_128_seed42` does not exist;
  `open(config_path)` (61-62) raises `FileNotFoundError` as the first statement of `main()`.
  Guard with a clear "run/seed must exist first" message.
- **CLE-009** (126, correctness) — **Empty-DataFrame `KeyError`.** With `results == []`,
  `df["stage"]` has no such column and the ternary condition raises before `else 0`. Check
  `df.empty` / column presence before indexing; represent "not evaluated" as `None`.

### MEDIUM
- **CLE-010** (126-137, correctness) — **Misleading verdict when `post_T2` absent but `df`
  non-empty.** `else 0` → prints a 0.0% "did not survive" conclusion for a stage that was
  never scored. Distinguish "not evaluated" from a measured low value before emitting a
  verdict.
- **CLE-011** (127, 131, numerics) — **Magic verdict thresholds `>20` and `>12`** with no
  derivation. Chance is `100/num_classes = 10%` (num_classes=10, line 95), so these are ~2x
  and ~1.2x chance — but that is undocumented and unnamed. Define thresholds relative to the
  eval's own chance + a documented margin.
- **CLE-033** (29, dependencies) — `evaluate_cross_modal_transfer` (imported line 29)
  transitively requires scikit-learn, absent from `code/requirements.txt`; import fails on a
  clean env. Shared cluster issue.

### LOW
- **CLE-012** (11, 14, dead-code) — `import numpy as np` (11) and `random_split` (14) unused.
- **CLE-031** (50, dead-code) — dead `else 1` in `_events_to_dense` (one of four copies).
- **CLE-030** (33-58, duplication) — `build_nmnist_test_loader` / `build_shd_test_loader` /
  `_events_to_dense` duplicate the same helpers in `phase1_sanity_check.py` and the two run
  scripts. Consolidate into a shared module.

## Cross-checks performed
- Confirmed `__main__` guard + `main()` present (grep count 1 each).
- Confirmed config path `results/cl_confirmation_fixed/...` does not exist.
- Confirmed `np` and `random_split` have zero uses beyond their import.
- Confirmed transfer keys used (`v2a`, `a2v`, `prototype_alignment`,
  `prototype_alignment_off_diag`, `sil_visual`, `sil_audio`, `chance`) match the return dict
  of `utils/cross_modal_transfer.evaluate_cross_modal_transfer`.
- No fabricated/hardcoded result numbers written by this file (contrast
  `run_rehearsal_transfer_eval_only.py`).

# Audit deep-dive — `code/test_provenance_smoke.py`

- **Cluster:** H-tests · **Shard:** w7_tst · **Prefix:** TST · **Status:** live
- **Length:** 110 lines · **Target under test:** `code/m7_trimodal_run_provenance.py` (the real, import-safe shared library)
- **Read-only static analysis. Nothing was imported or executed.**

## What this file is

A "smoke test" that exercises `m7_trimodal_run_provenance.py` end-to-end on CPU with a
`DummyLoader` of random tensors (lines 36-54): it checks the random-init baseline is near
chance for each task (lines 57-63), trains one task for one epoch and evaluates SHD
zero-shot (lines 66-93), and calls `print_task_sanity` (lines 96-106). It prints
`SMOKE TEST PASSED` at the end (lines 108-110).

Call-site verification (all confirmed against the source): `ExperimentConfig`,
`M7_ContinualAdaptiveModel`, `random_init_baseline`, `get_device`, `TASKS`,
`print_task_sanity`, `ContinualLearner`, `EmbeddingReplayBuffer` **all exist** in
`m7_trimodal_run_provenance.py`, and every keyword/positional argument used here matches
the real signatures (`random_init_baseline` at :804, `print_task_sanity` at :829,
`train_task` at :570, `evaluate` at :540, `EmbeddingReplayBuffer.__init__` at :418). So
this file would, in principle, run — the problems are structural and about assertion
quality, not broken references.

## Safety verdict (important)

Importing this file does **not** trigger the 4-5 hour run or any dataset download. It forces
`device="cpu"` (line 31), uses only synthetic tensors, and its single heavy import
(`m7_trimodal_run_provenance`) is `__main__`-guarded (that module, line 1378), has no
module-level dataset/DataLoader/mkdir, and does **not** import the dangerous unguarded
`m7_trimodal_run.py`. See finding **TST-021**. It is therefore safe to relocate into
`tests/`.

---

## Findings by severity

### Medium

- **TST-001 — Not a real pytest test, and not in the validation loop.** No `test_`-prefixed
  functions anywhere in the file, so `pytest` collects 0 items. And `pytest.ini:2` pins
  `testpaths = tests`, so the canonical loop `pytest tests/` never even scans `code/`. This
  file (and its two siblings) give a false impression of coverage. *Cross-cutting — applies
  to all three files in this shard.*

- **TST-002 — Whole body runs at import time.** No `__main__` guard, no functions; every
  top-level statement (lines 17-110) executes on import, including model construction and
  one epoch of `ContinualLearner.train_task`. Import == execution, so pytest collection runs
  training and a failed assertion becomes a collection/import error rather than a clean test
  failure. (Safe otherwise — see safety verdict.)

- **TST-003 — Asserted tolerance contradicts the printed claim.** Line 62 enforces
  `abs(ri - chance) < 8.0` but line 63 prints "within ~3pp of chance". 8 pp is ~3× the
  claimed 3 pp. The `8.0` is also a magic number with no derivation and is under-powered:
  each `random_init_baseline` evaluates ~40 samples (`DummyLoader(10, …)` → 10 batches × 4,
  lines 52-54, 42-48) averaged over `n_seeds=2` (line 59), so the estimate has a wide spread
  and the bound is arbitrary. Printing "~3pp" while enforcing 8 pp is a mislabeled claim in a
  RULE-1 repo.

- **TST-004 — The "key assertion" is not an assertion.** The comment on lines 87-88 calls the
  near-chance zero-shot check "The key assertion", but lines 89-93 implement it as an
  `if/else` that only prints — it never fails. `SMOKE TEST PASSED` (line 109) prints
  regardless. The one property this test most wants to guarantee (no spurious cross-modal
  signal / no leakage) is unenforced. The `< 3.0` band is also tighter than the noise floor
  of a ~40-sample `evaluate` (`dummy_shd = DummyLoader(10, 20)`, line 53), so the warning
  branch fires often yet nothing fails.

### Low

- **TST-005 — Hardcoded accuracy-like literals into `print_task_sanity`** (lines 103-105):
  `after_training_acc=15.0`, `random_init_baseline_acc=5.2`, `learning_gain=15.0 - …`. Only
  printed (never persisted to CSV), so low impact, but hardcoded accuracy-shaped numbers in
  a "SANITY" printout are exactly the RULE-1 smell, and could be misread as measurements in
  stdout.

- **TST-006 — Unused import** `import numpy as np` (line 7); no `np.` usage anywhere.

- **TST-007 — `DummyLoader` hardcodes batch size 4 and tensor shapes** (lines 44-48),
  ignoring `cfg.batch_size` (=4, line 27). They agree only by coincidence; changing the
  config would silently not change the batches. Magic shapes duplicated from dataset specs.

- **TST-008 — Zero-shot section is not seeded.** The test never calls `torch.manual_seed` /
  `np.random.seed` in its own body; the model built at line 68 consumes leftover RNG state
  from the preceding `random_init_baseline` loop (which reseeds internally, lines 58-62). So
  the zero-shot accuracy feeding the near-chance check is not reproducible-by-design, which
  compounds the tight/under-powered tolerances of TST-003/004.

### Info

- **TST-021 — Verified import-safety** (see safety verdict above). No action; guidance for
  relocation into `tests/`.

---

## Suggested remediation order

1. Fix the structural root cause once for the whole shard (TST-001/002): wrap the body in
   `test_*` functions + a `__main__` guard, and move this dataset-free CPU test into `tests/`
   so `pytest tests/` runs it.
2. Make the checks real and honest (TST-003/004): promote the "key assertion" to an `assert`,
   reconcile the tolerance with the printed claim, and raise the DummyLoader sample count so a
   tight bound is statistically defensible.
3. Seed the test body (TST-008), then clean up TST-005/006/007.

# Audit deep-dive — `code/test_cross_modal_alignment.py`

- **Cluster:** H-tests · **Shard:** w7_tst · **Prefix:** TST · **Status:** live
- **Length:** 169 lines · **Target under test:** `code/m7_trimodal_run_provenance.py` cross-modal alignment path
- **Read-only static analysis. Nothing was imported or executed.**

## What this file is

A smoke test for the cross-modal alignment feature. It builds an
`M7_ContinualAdaptiveModel` with a small config (`shared_dim=64`, `align_shared_dim=16`,
lines 44-59), checks `proj_shared`/`proj_private` exist (lines 78-79), verifies
`forward_with_shared` and standard `forward` shapes (lines 86-99), exercises
`orthogonal_penalty` and `CrossModalSupConLoss` (lines 101-115), runs a 2-epoch
`align_cross_modal` on a `FakePairedDataset` (lines 117-144), does 3 post-alignment training
steps (lines 146-159), and checks the gate still softmaxes (lines 161-165).

Call-site verification (all confirmed against source): every imported symbol
(`ExperimentConfig`, `M7_ContinualAdaptiveModel`, `ContinualLearner`,
`EmbeddingReplayBuffer`, `get_device`, `CrossModalSupConLoss`, `orthogonal_penalty`) exists
in `m7_trimodal_run_provenance.py`; `forward_with_shared` returns the 4-tuple this file
unpacks (:372, :399); `align_cross_modal` accepts the keywords passed and returns a
`history` dict with `xmscl_loss` (:675, :707); and `paired_digit_collate_fn` (imported
mid-file at line 124) exists in `Dataloaders/paired_digit_loader.py:231` and is compatible
with `FakePairedDataset`'s `(vis, aud, label)` items. So this file would run — the issues
are structural, flakiness, and weak assertions, not broken references.

## Safety verdict

Safe to import: `device="cpu"` (line 58), synthetic data only, and it imports only
`m7_trimodal_run_provenance` (guarded, no module-level side effects) and
`Dataloaders.paired_digit_loader`. It does **not** pull in the unguarded `m7_trimodal_run`.
See **TST-021**.

---

## Findings by severity

### Medium

- **TST-009 — Whole body runs at import time.** No `__main__` guard and no test functions;
  model build (line 75), forward passes, the 2-epoch `align_cross_modal` run (lines 130-138),
  and 3 optimizer steps (lines 152-157) all execute on import. Same import==execution hazard
  as the provenance smoke test; also the reason `pytest` collects 0 items here (see TST-001).

- **TST-010 — Flaky monotonic-decrease assertion.** Line 142 asserts
  `history['xmscl_loss'][1] < history['xmscl_loss'][0]` — a *strict* drop between just two
  epochs, on `FakePairedDataset` whose per-class signal is `c*0.05` over `randn*0.1` noise
  (lines 37-39), optimized with AdamW under `CosineAnnealingLR(T_max=2)`
  (`m7_trimodal_run_provenance.py:702`). A strict two-point decrease on near-noise data is
  not guaranteed, so this can fail spuriously — a false regression signal.

### Low

- **TST-011 — Local `TASKS` duplicates the canonical one.** Lines 61-65 hand-define `TASKS`
  rather than importing it; it is byte-identical to `m7_trimodal_run_provenance.py:955-959`
  *today*, so no current bug, but it is a second source of truth that will drift silently if
  the real task schema/class-counts change. (Not literal shadowing — this file never imports
  `TASKS` — it is a divergent copy.)

- **TST-012 — Vacuous assertions.** Line 105 `assert pen.item() >= 0` can never fail:
  `orthogonal_penalty` returns `cross.pow(2).sum()`
  (`m7_trimodal_run_provenance.py:218-219`), a sum of squares. Line 114
  `assert loss_xmscl.item() > 0` is likewise essentially always true for a SupCon loss. These
  confirm only that the calls return a scalar; they validate no correctness.

- **TST-013 — Orthogonal-penalty test uses a raw slice, not `proj_private`.** `private_vis`
  is `full_vis[:, cfg.align_shared_dim:]` (line 102), i.e. tail features — not the model's
  learned `proj_private` projection (`m7_trimodal_run_provenance.py:339`). So it exercises a
  different quantity than training uses and (given the vacuous ≥0 check) would not catch a
  `proj_private` wiring bug.

- **TST-014 — Unused import** `import numpy as np` (line 11).

- **TST-015 — Mid-module imports.** `DataLoader` (line 123) and `paired_digit_collate_fn`
  (line 124) are imported in the middle of the executable body, relying on the repo-root
  `sys.path.insert` at line 8; `Dataset` was already imported at line 12. Hurts readability
  and is easy to miss if the body is refactored into functions.

### Info

- **TST-021 — Verified import-safety** (shared with the provenance test). No action.

## Assertions that are actually good (no finding)

- The `forward_with_shared` / `forward` **shape** assertions (lines 89-99) and the **gate
  softmax** assertion (lines 163-164, sums to 1 within `atol=1e-5`) are meaningful and worth
  keeping.

---

## Suggested remediation order

1. Structural fix with the shard (TST-001/009): `test_*` functions + `__main__` guard; this
   test is dataset-free and belongs in `tests/`.
2. De-flake TST-010 (tolerance / trend-over-more-epochs / seed the run).
3. Strengthen the weak assertions (TST-012/013) into checks that can actually fail; then
   TST-011/014/015 cleanup.

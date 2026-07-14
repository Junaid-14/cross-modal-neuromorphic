# Audit deep-dive — `code/test_ordinal_anchor_smoke.py`

- **Cluster:** H-tests · **Shard:** w7_tst · **Prefix:** TST · **Status:** live
- **Length:** 84 lines · **Target under test:** `code/run_ordinal_anchor.py` (Stage 0 RDM + 1-seed MLP)
- **Read-only static analysis. Nothing was imported or executed.**

## What this file is

A smoke test that imports helpers from `run_ordinal_anchor`, builds a full production-style
`ExperimentConfig` (lines 23-56), builds digit loaders, loads cached features, runs the
Stage-0 RDM diagnostic, and runs one seed of the MLP head (`main`, lines 22-80). Unlike the
other two files in this shard it **has** a `__main__` guard (line 83) and a `main()`
function — so importing it does not run the body. But it still crashes, for a chain of
reasons, and it is not a pytest test.

## The crash chain (three layered blockers)

1. **TST-016 (HIGH, the live crash) — ImportError at line 14.** The import
   `from run_ordinal_anchor import (…, _build_digit_loaders, …)` fails because
   `run_ordinal_anchor` neither defines nor re-exports `_build_digit_loaders`. That symbol
   lives in `run_correspondence_anchor_v2.py:67`. `run_ordinal_anchor` *does* import a batch
   of helpers from that module (`run_ordinal_anchor.py:51-58`: `_hash_params`,
   `_load_cached_features`, `build_paired_feature_dataset`, `compute_held_out_metrics`,
   `run_procrustes`, `train_correspondence`) but **not** `_build_digit_loaders`; it only
   defines a differently-named `_build_test_loaders` (`run_ordinal_anchor.py:75`). Result:
   `ImportError: cannot import name '_build_digit_loaders' from 'run_ordinal_anchor'`, raised
   before `main()` ever runs. **This is the actual failure today** — the `__main__` guard is
   moot, and it fires earlier than the dataset/checkpoint gaps the original seed note
   described. (The other six imported names — `ExperimentConfig`, `M7_ContinualAdaptiveModel`,
   `get_device` from provenance via :40-47; `_load_cached_features` from
   run_correspondence_anchor_v2 via :51-58; `compute_rdms` at :171 and `run_one_seed` at :395
   defined locally — all resolve correctly.)

2. **TST-017 (MEDIUM, masked) — missing datasets / download risk.** If the import is fixed,
   `_build_digit_loaders(cfg)` (line 61) constructs `tonic.datasets.NMNIST` and `SHD` with
   `save_to='./ICONS_M7/datasets'` and `mkdir(parents=True)`
   (`run_correspondence_anchor_v2.py:67-91`). The repo has no `ICONS_M7` datasets, so this
   triggers a multi-GB tonic **download** (or crashes offline), and the path is CWD-relative
   (only correct from `code/`). Currently masked by TST-016 — but a naive "fix the import and
   run it" hits this.

3. **TST-018 (MEDIUM, masked) — missing cache + checkpoint.** `_load_cached_features(device)`
   (line 64) needs a cached-feature artifact, and `run_one_seed(…)` (line 73) does
   `torch.load(CKPT_SOURCE)` where `CKPT_SOURCE =
   results/cl_confirmation/checkpoints/A_no_align_seed42/final_shd_shd.pt`
   (`run_ordinal_anchor.py:63-70, :410`). Verified: the repo has **no `results/` directory
   and no `*.pt` files**, so this raises `FileNotFoundError`. The `run_one_seed(42, cfg,
   loaders, cached, device, head_type="mlp")` call itself (line 73) matches
   `run_ordinal_anchor.py`'s 6-arg `run_one_seed` signature (:395-402) — the crash is the
   missing artifact, not the call.

## Safety verdict

Importing this file is safe (heavy work is behind `main()`), and the import chain
(`run_ordinal_anchor` → `run_correspondence_anchor_v2`, `utils.cross_modal_transfer`,
`m7_trimodal_run_provenance`) does **not** reach the unguarded `m7_trimodal_run` — verified.
The download in TST-017 only happens if you *run* the file after fixing TST-016, not on
import.

---

## Findings by severity

### High

- **TST-016 — ImportError on `_build_digit_loaders`** (line 14). Broken reference; the test
  cannot be imported. Fix: import it from `run_correspondence_anchor_v2` (or add it to
  `run_ordinal_anchor`'s re-export list at :51-58) — and confirm whether the intended loader
  is really `_build_digit_loaders` or the locally-defined `_build_test_loaders`.

### Medium

- **TST-017 — Requires absent N-MNIST/SHD datasets; becomes a large tonic download** once the
  import is fixed (line 61). For a real smoke test, feed a tiny fake paired loader (as
  `test_cross_modal_alignment.py` does) or gate the real-data path behind an explicit flag /
  `pytest.skip`.

- **TST-018 — Requires an absent cached-feature file and checkpoint** (lines 64, 73 →
  `run_ordinal_anchor.py:63-70, :410`). Provide synthetic fixtures / construct a fresh model,
  or skip gracefully when the artifacts are missing.

### Low

- **TST-019 — Has a `__main__` guard + `main()` (good) but is still not a pytest test and
  asserts nothing.** No `test_*` function (0 items collected — see TST-001), and no `assert`
  statements at all: it just runs and prints "Smoke test passed." (line 80). It is a manual
  demo script mis-named as a test.

- **TST-020 — Inline `ExperimentConfig` duplicates the full production hyperparameter set**
  (lines 23-56): 34 fields, most equal to the dataclass defaults
  (`m7_trimodal_run_provenance.py:34-74`). Redundant and a second place production values are
  pinned. A smoke test should override only the few small/cheap fields it needs.

## Cross-cutting

- **TST-001** applies here too: named `test_*.py`, collected by nothing (`pytest.ini:2`
  `testpaths = tests`).

## Note (out-of-scope but relevant for the fixer)

There are two incompatible `run_one_seed` functions in the codebase:
`m7_trimodal_run_provenance.py:962` takes `(cfg, data_loaders, results_dir, ckpt_dir)`,
while `run_ordinal_anchor.py:395` takes `(seed, cfg, loaders, cached, device, head_type)`.
This test correctly targets the latter, so it is not a bug *in the test* — but the
same-named, different-signature functions are a maintenance trap worth flagging to whoever
owns those modules.

---

## Suggested remediation order

1. **TST-016 first** — it is the only thing failing today; fix the import source.
2. Then decide the test's identity: either make it a dataset-free `test_*` with synthetic
   fixtures (addresses TST-017/018/019/020 together), or rename it to a `run_`/`demo_` script
   and remove it from any `test_*` expectation.

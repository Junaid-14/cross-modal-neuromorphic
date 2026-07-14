# Retirement audit — `code/archive/` (9 files)

**Shard:** `audit/raw/w_arch.jsonl` · **Prefix:** `ARCH` · **Status:** `archive` (all findings)
**Scope:** static, read-only. No file was imported or executed. Findings ARCH-001..ARCH-023.

These nine files are all already superseded. This audit confirms *why* each is dead and gives a
per-file action (keep-archived vs delete). It captures **unique** issues only; four cross-cutting
defects (missing dependency, duplicated class, missing checkpoint, sim≠hardware framing) are
formalized once as cluster findings and referenced back.

## Retirement verdicts at a glance

| File | Role | Runnable today? | Verdict / action |
|---|---|---|---|
| `run_correspondence_anchor.py` | v1 of correspondence-anchor experiment | No (missing ckpt) | **DELETE** — v2 supersedes; carries a held-out bug |
| `run_cl_confirmation.py` | older T1→T2→T3 A/C confirmation | No (as `import` from live) | **KEEP-ARCHIVED** until live ref fixed (ALGN) |
| `verify_full.py` | full-net PyTorch↔Brian2 verify (build-once) | No (deps + ckpt) | KEEP-ARCHIVED or delete — superseded by live `final_evaluation.py` |
| `verify_simulators.py` | earlier full-net verify (per-sample rebuild) | No (deps + ckpt) | **DELETE** — superseded by `verify_full.py` |
| `verify_brian2_raw.py` | raw-Brian2 speed variant | No (deps + ckpt) | KEEP-ARCHIVED or delete — redundant |
| `verify_tiny.py` | tiny synthetic variant-search POC | No (deps only) | **KEEP-ARCHIVED** — self-contained methodology record |
| `debug_brian2_exact.py` | one-off "why silent" debug | No (deps + ckpt) | **DELETE** — debug scratch |
| `debug_gradient.py` | legacy gradient-flow sanity | Marginal (stale path) | **DELETE** — legacy, covered by `tests/` |
| `benchmark_mps.py` | legacy MPS throughput bench | Marginal (stale path, mps-only) | **DELETE** — legacy micro-bench |

## Cross-cutting cluster findings (formalized once)

- **ARCH-010 [dependencies, medium]** — the Brian2/PyNN verify cluster imports `brian2` and/or
  `pyNN.brian2`, **neither of which is in `code/requirements.txt`** (pins are torch/snntorch/tonic/
  transformers/numpy/pandas/matplotlib/tqdm/h5py/scipy/pytest only). All five raise
  `ModuleNotFoundError` at import → **un-runnable in the declared env**. Sites: `verify_full.py:20-21`,
  `verify_simulators.py:21-22`, `verify_brian2_raw.py:20-25`, `verify_tiny.py:8-9`,
  `debug_brian2_exact.py:13-14`.
- **ARCH-011 [duplication, medium]** — `ConvFrontend` is **copy-pasted verbatim in 5 files**:
  `verify_full.py:26-46`, `verify_simulators.py:27-47`, `verify_brian2_raw.py:30-50`,
  `debug_brian2_exact.py:19-39`, and the **LIVE** `code/final_evaluation.py:33-53`. A numerics fix to
  one (β, reset order) will not propagate. Extract one shared, import-safe copy; verify the extraction
  is numerically inert before touching the live file.
- **ARCH-012 [dependencies, medium]** — four scripts `torch.load(
  "ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt")`; `ICONS_M7/` does not exist →
  `FileNotFoundError`. Sites: `verify_full.py:259`, `verify_simulators.py:265`,
  `verify_brian2_raw.py:290`, `debug_brian2_exact.py:44`. (`verify_tiny.py` is exempt — synthetic.)
  Also note `torch.load` here is without `weights_only=True`.
- **ARCH-013 [research-integrity, medium]** — the cluster verifies a PyTorch discrete-time simulator
  against **Brian2, a software simulator**, and prints "VERIFIED … matches Brian2/PyNN". Comments call
  the equations "exact same as PyNN/SpiNNaker" (`verify_brian2_raw.py:6,180`). This is **sim-vs-sim
  agreement, not SpiNNaker-hardware verification**; no energy/latency is or can be produced. No
  hardcoded *result* numbers were found in the cluster itself. Do not let a "VERIFIED" line be cited as
  on-chip correctness (mapping ≠ inference ≠ real hardware; nothing has run on a board).

---

## `run_correspondence_anchor.py` (767) — correspondence-anchor **v1**

**Role.** Trains per-modality projection heads over a frozen A_no_align checkpoint to map digits to
fixed orthonormal anchors, then runs held-out-class and label-shuffled controls. Import-safe (`main()`
guarded, l.766). **Superseded by the live `code/run_correspondence_anchor_v2.py` (34 KB).**

**Verdict: DELETE.** v2 is the record. Beyond being v1, it contains a genuine correctness bug and a
RULE-1 print, both good reasons not to resurrect it as-is.

### High
- **ARCH-015 — held-out-class masking is a no-op; held-out digits are still trained on.**
  `ce = nn.CrossEntropyLoss()` (l.377, default `reduction='mean'`) makes `loss` a **scalar** (l.412-414).
  The masking `loss = loss * mask.float()` then `loss.sum()/mask.sum()` (l.418-419) algebraically
  returns that same scalar mean over **all** samples — held-out classes are never excluded. And
  `build_paired_feature_dataset` takes `held_out_classes` (l.299) but never uses it, so held-out samples
  are in the training set too. The step-7 held-out test (digits 8,9; l.616-628) therefore measures
  alignment on classes that *were* trained → invalid, along with the l.651 "discovered vs engineered"
  verdict. Fix: `reduction='none'` before masking, or filter in the dataset builder.

### Medium
- **ARCH-014 — hardcoded numbers in a "BASELINE COMPARISON" table (RULE 1).** l.757-761 print baseline
  rows as literals (`~97.0%`, `~13.3%`, `13.3%`, `14.8%`, `95.7%`, `14.4%`) beside the run's computed
  `mean_retention`/`mean_v2a`. Hardcoded numbers presented as results is the exact RULE-1 anti-pattern
  (stdout-only, archived → bounded harm).
- **ARCH-001 — superseded v1 (whole file).** Dead; DELETE.

### Low
- **ARCH-016 — dead `vis_indices`/`aud_indices` loop that still consumes the RNG.** l.313-322 build
  index lists via `rng.choice` and never read them; the paired selection recomputes at l.326-343 with
  the **same** `rng` (l.312). The dead loop is *not* inert — removing it shifts the RNG stream and
  changes which samples train the heads.
- **ARCH-017 — null baseline shuffles only 2 held-out classes.** `np.random.shuffle` (l.526) of a
  length-2 list is the identity ~50% of the time, making `null_baseline` equal to same-class similarity
  and the l.651 verdict unreliable; also uses the global numpy RNG, not the local one.

---

## `run_cl_confirmation.py` (627) — older CL confirmation (A & C)

**Role.** Full T1→T2→T3 sequence for conditions A (no align) and C (split-128 + orthogonal), writing
measured metrics to per-cell CSVs (no hardcoded results). Import-safe (`main()` guarded, l.626);
datasets load in `main()`, not at import.

**Verdict: KEEP-ARCHIVED until the live reference is fixed.**

### Medium
- **ARCH-002 — archived but still imported by the LIVE `run_digit_only_alignment.py:616`.**
  The `--confirm` branch does `import run_cl_confirmation; run_cl_confirmation.main()` (l.616-617), but
  the live file only adds `REPO_ROOT` and `REPO_ROOT/code` to `sys.path` (l.50-51), **not**
  `code/archive/`, so the import raises `ModuleNotFoundError` — the whole `--confirm` path is dead.
  This is a stale live→archive coupling. **Related: the live ALGN finding.** Resolve there (delete the
  dead branch, or move this file back to `code/` and fix the import) *before* deleting this file.
- **ARCH-018 — `forgetting_T1_T3` stores the incremental T2→T3 drop, not total T1→T3.** l.412
  `= acc_t1_after_t2 - acc_t1_after_t3` (self-labeled "incremental"); the true total is
  `total_forgetting_T1_T3` (l.414). The column name misleads anyone mining old CSVs — sensitive under
  RULE 5 (forgetting is a headline claim). Rename to make incremental-vs-total explicit.

### Low
- **ARCH-019 — duplicate metric columns.** `forgetting_T2_T3` (l.413) and `total_forgetting_T2_T3`
  (l.415) have identical RHS → always equal; both are in `EXPECTED_COLUMNS` and the aggregation list.
- **ARCH-020 — "95% CI" uses normal z=1.96 at n≤3.** l.497 `ci95 = 1.96*std/sqrt(n)` with seeds
  `[42,123,456]`; a t-multiplier would be much wider. Treat any CI from this script as under-wide.

---

## `verify_full.py` (355) — full-net PyTorch↔Brian2/PyNN verify (build-once)

**Role.** Loads the N-MNIST backbone from checkpoint, extracts conv features, generates Poisson spikes,
and compares a PyTorch discrete-time IF sim (Variant D, cross-factor, l.136) against a **build-once**
Brian2/PyNN network reused across 20 stratified samples via `sim.reset()` (l.240).

**Verdict: KEEP-ARCHIVED or delete** — superseded by the live `final_evaluation.py`; it is the closest
provenance record of the real-network verification.

- **ARCH-003 [low]** — superseded status.
- Shares **ARCH-010** (deps), **ARCH-011** (ConvFrontend l.26-46), **ARCH-012** (ckpt l.259),
  **ARCH-013** (verdict text l.342-349).

---

## `verify_simulators.py` (362) — earlier full-net verify (per-sample rebuild)

**Role.** Same intent as `verify_full.py` but an earlier iteration: PyTorch Variant A
(`(ie+b)*v_factor`, l.139) and the PyNN network is **rebuilt from scratch for every sample**.

**Verdict: DELETE** — strictly superseded by `verify_full.py`, no unique capability.

- **ARCH-004 [low]** — superseded status.
- **ARCH-021 [performance, low]** — rebuilds all Populations + Projections inside the per-sample loop
  (l.186-238), re-enumerating every above-threshold weight per sample; this is exactly what
  `verify_full.py` fixed with build-once + `sim.reset()`.
- Shares **ARCH-010** (deps l.21-22), **ARCH-011** (ConvFrontend l.27-47), **ARCH-012** (ckpt l.265).

---

## `verify_brian2_raw.py` (376) — raw-Brian2 speed variant

**Role.** Same verification using **raw Brian2** (`NeuronGroup`/`Synapses`, `prefs.codegen.target='numpy'`,
l.25) instead of the PyNN wrapper, for speed; PyTorch side uses the cross-factor variant (l.138).

**Verdict: KEEP-ARCHIVED or delete** — functionally redundant with `verify_full.py`.

- **ARCH-005 [low]** — redundant-variant status.
- Shares **ARCH-010** (brian2 l.20-25), **ARCH-012** (ckpt l.290), **ARCH-011** (ConvFrontend l.30-50).
- Carries the strongest **ARCH-013** overclaim comment: "exact same as PyNN/SpiNNaker" (l.180) and
  docstring (l.6).

---

## `verify_tiny.py` (216) — tiny synthetic variant-search POC

**Role.** A 5→3→2 synthetic network sweeps four discretization variants **A/D/H/I** against Brian2/PyNN
to determine which matches `IF_curr_exp` exactly — this is what justified Variant D downstream. **No
checkpoint, no dataset, no ConvFrontend** — fully synthetic.

**Verdict: KEEP-ARCHIVED.** Self-contained methodology record; its only blocker is the shared
brian2/pyNN dependency.

- **ARCH-006 [info]** — POC/keep status.
- Shares **ARCH-010** only (deps l.8-9). Not affected by ARCH-011/012.

---

## `debug_brian2_exact.py` (195) — one-off "why is it silent" debug

**Role.** Single-sample (`label=0`, l.148) debugging of why the exact PyNN sim produced no spikes;
reaches into Brian2 internals via `input_pop.brian2_group.set_spikes` (l.169).

**Verdict: DELETE** — transient debug artifact; conclusion folded into `verify_full.py`/`final_evaluation.py`.

- **ARCH-007 [low]** — debug-scratch status; also unused imports `Synapses`, `nA` (l.14).
- Shares **ARCH-010** (deps l.13-14), **ARCH-011** (ConvFrontend l.19-39), **ARCH-012** (ckpt l.44).

---

## `debug_gradient.py` (98) — legacy gradient-flow sanity

**Role.** Forward/backward/overfit sanity check on the **legacy bi-modal** model via
`Models.registry.build_model('baseline', …)` (l.9,17), synthetic data, `__main__`-guarded, runs on
cpu then mps. Does not touch M7; `tests/` now covers the legacy package.

**Verdict: DELETE.**

- **ARCH-008 [low]** — legacy/superseded status.
- **ARCH-022 [dependencies, medium]** — stale hardcoded `sys.path.insert(0,
  '/Users/agent1/Documents/neuromorphic/m7-trimodal-cl')` (l.5) — a different machine/user; the path
  does not exist here, so the `Models` import only resolves if CWD is the repo root (portable by
  accident).

---

## `benchmark_mps.py` (63) — legacy MPS throughput benchmark

**Role.** Times batch sizes 32/64/128 on the **legacy** `Models.registry` model (l.9) with synthetic
data; prints samples/sec (timing, computed — not a research number). `__main__`-guarded.

**Verdict: DELETE.**

- **ARCH-009 [low]** — legacy/superseded status.
- **ARCH-022 [dependencies, medium]** — same stale `/Users/agent1/...` `sys.path` (l.6).
- **ARCH-023 [quality, low]** — `DEVICE = torch.device('mps')` hardcoded (l.17), no CPU/CUDA fallback;
  crashes off Apple Silicon instead of using the repo's `get_device()` auto-select.

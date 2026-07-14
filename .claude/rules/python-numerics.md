---
paths:
  - "code/**/*.py"
  - "Models/**/*.py"
  - "Dataloaders/**/*.py"
  - "training/**/*.py"
  - "utils/**/*.py"
  - "tests/**/*.py"
---

# Working on the Python / numerics

## The traps in this codebase, in order of how much time they cost

### 1. Importing `m7_trimodal_run` starts a 4–5 hour run

`code/m7_trimodal_run.py` has **no `if __name__ == "__main__"` guard** and loads its
datasets at module import time. `python -c "import m7_trimodal_run"` is not an import
check — it is the full tri-modal experiment. Read the file instead.

(`.claude/hooks/guard_long_runs.py` will stop and ask, but don't rely on it.)

### 2. Training is SILENTLY SKIPPED when a checkpoint exists

`run_one_seed()` checks for `final_{task}_{task}_seed{seed}_{condition}.pt` and, if it
finds one, **loads it and skips training entirely** — no warning. A stale checkpoint from
an older code version therefore produces stale results that look fresh.

Before trusting any run: check whether checkpoints predate the code changes you care about.

### 3. The results CSV has two schemas in one file

`cl_trimodal_results.csv` receives:
- **incremental rows** written after each step — these have a `step` value;
- **summary rows** written at the end of each seed — these have **no** `step`.

Reruns append, so duplicates accumulate. The mandatory ritual before any analysis:

```python
df = df[df["step"].isna()].drop_duplicates()
```

Do not re-implement this. Use `.claude/skills/analyze-cl-results/scripts/load_results.py`,
which is the single canonical loader.

### 4. Seeds

Policy is `[42, 123, 456]`, and `code/m7_trimodal_run.py` now matches it (fixed 2026-07-14;
it was `[123, 456]`, so seed 42 never ran and every "3-seed" number was really a 2-seed
number). **Any result generated before that fix is a 2-seed result.** Always set **and
report** both the torch and numpy seeds. Splits are fixed with `torch.Generator().manual_seed(42)` across all seeds —
only model init and DataLoader shuffling vary.

### 5. macOS / MPS

- Device auto-selects MPS → CUDA → CPU.
- **`num_workers=0` is required on macOS** (tonic's multiprocessing doesn't survive fork).
  Do not "optimise" this to a higher value.
- Do not set `PYTORCH_ENABLE_MPS_FALLBACK` to work around an unsupported op without saying
  so — silent CPU fallback changes both timing and, occasionally, numerics.

### 6. Relative paths

The scripts use `BASE_DIR = './ICONS_M7'`. That only resolves correctly when the working
directory is `code/`. Running `python code/m7_trimodal_run.py` from the repo root puts the
outputs somewhere different from what the docs describe.

---

## Numerical-integrity rules for any edit here

- **Refactors must be numerically inert.** If a change could move a number — operation
  order, dtype, reduction, RNG consumption, an `eps`, a seed call — say so *before* making
  it. Never fold a behavioural change into a cleanup.
- **Accuracy never travels alone.** Any accuracy you compute, log, or report must come with
  spike rate / sparsity and the timestep count.
- **Test accuracy, not validation.** SHD's val/test gap is 15–19 pp. When you pull a number
  out of code, confirm which split produced it.
- Keep `code/m7_trimodal_run.py` **self-contained** — it is the reproducibility entry point.
  No imports from `Models/`, `Dataloaders/`, `training/`, or `utils/`.

## The safe validation loop

```bash
pytest tests/     # 6 suites, tonic mocked — fast, needs no datasets
```

This is the only quick check that works without the ~23 GB of data. Use it by default.
`tests/` imports the **legacy bi-modal** package (`Models/`, `Dataloaders/`, `training/`) —
which is *not* what `code/m7_trimodal_run.py` uses. Passing tests do not validate M7.

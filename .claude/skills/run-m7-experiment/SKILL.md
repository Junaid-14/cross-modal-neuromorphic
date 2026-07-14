---
description: Pre-flight and launch a tri-modal continual-learning training run safely. Use before running m7_trimodal_run.py or any long M7 experiment. Checks the venv, datasets, disk, seed policy and stale checkpoints FIRST, because the script silently skips training when a checkpoint exists and silently starts a 4-5 hour run on import.
argument-hint: "[seeds] [condition: replay|no_replay|both]"
allowed-tools: Bash(python3 *), Bash(ls *), Bash(df *), Bash(find *), Read, Grep
---

# Running an M7 experiment

A run costs **4–5 hours on MPS**. The failure modes here are expensive and silent, so the
pre-flight is not optional.

## Pre-flight — every item, before launching anything

### 1. Is there an environment?
```bash
ls code/.venv/bin/python 2>/dev/null || echo "NO VENV — run /bootstrap-env first"
```

### 2. Are the datasets there?
```bash
ls -d ICONS_M7 code/ICONS_M7 2>/dev/null || echo "NO DATASETS — ~23 GB will be downloaded"
df -h .    # do you actually have room?
```

### 3. ⚠️ Stale checkpoints — the silent-staleness trap
```bash
find . -name 'final_*.pt' -not -path './.git/*' 2>/dev/null
```
`run_one_seed()` checks for `final_{task}_{task}_seed{seed}_{condition}.pt` and, **if it
finds one, loads it and skips training entirely — with no warning.**

So if any checkpoint predates your code changes, the run will quietly reuse it and produce
**stale results that look fresh.** If you changed the model, delete the stale checkpoints or
you are not testing your change. Confirm this with the user before deleting anything.

### 4. Seeds — fixed, but confirm it hasn't regressed
```bash
grep -n 'SEEDS' code/m7_trimodal_run.py     # expect: SEEDS = [42, 123, 456]
```
Until 2026-07-14 this read `SEEDS = [123, 456]` — **seed 42 never ran**, so every result the
script produced was a **2-seed** number while being reported as 3-seed. It now runs all three.

⚠️ **Any CSV or figure produced before that date is 2-seed data**, whatever it is labelled.
If you are comparing against an older run, you are not comparing like with like.

### 5. ⚠️ Never import the module to "check" it
```bash
python -c "import m7_trimodal_run"   # ← this STARTS THE 4-5 HOUR RUN
```
There is no `if __name__ == "__main__"` guard and datasets load at import time. To inspect
the module, **read the file.** (`.claude/hooks/guard_long_runs.py` will stop and ask, but
don't lean on it.)

### 6. Working directory
The script uses `BASE_DIR = './ICONS_M7'`, which only resolves as documented when the CWD is
`code/`. Running it from the repo root puts outputs somewhere else.

## Launch

```bash
cd code
../code/.venv/bin/python m7_trimodal_run.py 2>&1 | tee run_$(date +%Y%m%d_%H%M).log
```

Keep the log. It is the only record of what actually happened if the CSV write is
interrupted.

## During and after

- **RULE 2 — back up to HuggingFace after every seed.** Checkpoints and results are
  gitignored; git will not save you, and **this team has already lost data once** to a
  terminated cloud instance. A finished run that isn't backed up may as well not have
  happened.
- Results are appended incrementally, so a killed run leaves a **partial, duplicate-laden**
  CSV. Always analyse via `/analyze-cl-results` — never read the CSV raw.
- **Do not run `code/generate_figures_3seed.py` until `/analyze-cl-results` says both the
  `no_replay` and `replay` conditions are present.** It indexes both positionally and raises
  `IndexError` otherwise.

## When you report the results

- **Test** accuracy, not validation (SHD's val/test gap is 15–19 pp).
- Accuracy travels with **spike rate / sparsity and timestep count**.
- Say how many seeds actually ran — and if it was 2, say 2.

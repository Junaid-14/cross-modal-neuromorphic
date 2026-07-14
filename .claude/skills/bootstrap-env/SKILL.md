---
description: Create the Python virtualenv this repo needs but does not have, install the pinned dependencies, and verify the scientific stack imports. Use when setting up the repo, when a command fails because code/.venv is missing, when "source code/.venv/bin/activate" errors, or when torch/snntorch/tonic won't import.
argument-hint: "[--force to recreate an existing venv]"
allowed-tools: Bash(python3.11 *), Bash(python3 *), Bash(source *), Bash(*/pip *), Bash(ls *), Read
---

# Bootstrapping the environment

**Nothing in this repo runs today.** `code/.venv` does not exist, yet every runbook in
`CLAUDE.md`, `AGENTS.md`, and `docs/` opens with `source code/.venv/bin/activate`. This
skill fixes that.

## Version constraint — this matters

The project pins **Python 3.11**. The system default here is **3.14**, which will not
satisfy the pins (`torch==2.11.0`, `snntorch==0.9.4`, `tonic==1.6.0`). Use `python3.11`
explicitly. It is installed (3.11.12); confirm before starting:

```bash
python3.11 --version    # expect 3.11.x
```

If it's missing, stop and say so — do not silently fall back to 3.14 and produce an env
that half-works.

## Steps

```bash
# 1. Create the venv with the RIGHT interpreter (not the system default)
python3.11 -m venv code/.venv

# 2. Install the pinned deps
code/.venv/bin/python -m pip install --upgrade pip
code/.venv/bin/python -m pip install -r code/requirements.txt

# 3. Verify the stack actually imports and the accelerator is visible
code/.venv/bin/python - <<'PY'
import sys, torch
print("python     ", sys.version.split()[0])
print("torch      ", torch.__version__)
print("mps        ", torch.backends.mps.is_available())
print("cuda       ", torch.cuda.is_available())
import snntorch, tonic
print("snntorch   ", snntorch.__version__)
print("tonic      ", tonic.__version__)
PY

# 4. Confirm the safe validation loop works
code/.venv/bin/python -m pytest tests/ -q
```

## What "done" looks like

- `code/.venv/bin/python --version` reports **3.11.x**
- `torch`, `snntorch`, `tonic` all import
- On Apple Silicon, `torch.backends.mps.is_available()` is `True`
- `pytest tests/` passes (6 suites; tonic is mocked, so this needs **no datasets**)

Report the actual versions you observed. Do not assert success without showing the output.

## What this does NOT get you

- **The datasets.** `ICONS_M7/` (~23 GB of N-MNIST / SHD / DVS-Gesture) is not in the repo
  and is not downloaded here. Training will try to fetch it.
- **Any results.** There are no CSVs and no checkpoints. `pytest tests/` passing does not
  validate M7 — those tests exercise the **legacy bi-modal** package (`Models/`,
  `Dataloaders/`, `training/`), not `code/m7_trimodal_run.py`.

## Note on the pins

`code/requirements.txt` carries some unusual pins (e.g. `pandas==3.0.2`, `numpy==1.26.4`).
If resolution fails, **report the conflict — do not "fix" it by relaxing a pin.** Changing a
dependency version can move numerical results, and this project's first rule is that numbers
don't move silently.

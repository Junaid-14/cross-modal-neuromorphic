---
description: Report what actually exists in this repo versus what the documentation claims, and find drift between the docs and the code. Use when orienting in the repo, when a documented path or command doesn't work, when the docs and the tree seem to disagree, or before trusting any instruction written in CLAUDE.md, AGENTS.md, README.md or docs/.
allowed-tools: Bash(python3 *), Bash(find *), Bash(ls *), Bash(git *), Bash(grep *), Read, Glob, Grep
---

# Checking the real state of the repo

This repo has a history of **documentation describing things that do not exist**. Before
acting on any documented instruction, check it.

## Fast path

```bash
python3 .claude/hooks/session_state.py \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['hookSpecificOutput']['additionalContext'])"
```

That is the same ground-truth snapshot injected at every session start: git branch, venv,
datasets, results CSVs (with seeds/conditions), checkpoints, notebooks.

## Known drift — verified, and the reason this skill exists

Prose in the repo has claimed all of the following. **None of it is true.**

| Claimed somewhere in the docs | Reality |
|---|---|
| `configs/`, `results/`, `figures/`, `notebook/`, `experiments/`, `ICONS_M7/` | **None exist** |
| `code/.venv/` — every runbook starts by activating it | **Does not exist** |
| "The notebook is canonical" (Cell 11 → Cell 30) | **No `.ipynb` in the repo.** `code/modify_notebook.py` is the only surviving record of its structure — and its target file is gone |
| `python code/test_spinnaker_hello.py` | **File does not exist** (gitignored: `code/test_spinnaker_*.py`) |
| "All 3 seeds `[42, 123, 456]` must run" | ✅ Fixed 2026-07-14 — the code now runs all three. It previously ran only `[123, 456]`. |
| "no tests are currently present" (AGENTS.md) | `tests/` has **6 working pytest suites** — the only fast, safe validation loop |
| `Models/model_1_baseline.py` (AGENTS.md) | The file is `Models/baseline.py` |
| A `.bib` file for the paper | **None.** `paper/*.tex` uses an inline bibliography, mostly `\todo{verify}` |

`CLAUDE.md` and `AGENTS.md` have since been corrected. If you find a *new* instance of
drift, that is a finding — report it, and fix the doc rather than working around it.

## Checking a specific claim

Before following any documented command or path:

```bash
ls -la <the path the doc mentions>       # does it exist at all?
git log --oneline -5 -- <path>           # was it deleted? when? why?
```

`git log` is often the answer: several of these directories were removed in the
`Repo hygiene` / `Remove stale info/ and experiments/ directories` commits, and the docs
were never updated to match.

## Deeper audit

For a full doc↔code reconciliation across every file, run the multi-agent workflow rather
than doing it by hand:

```
.claude/workflows/doc-code-reconcile.js
```

It fans out over every doc, extracts each asserted path / command / hyperparameter / number,
verifies it against the real tree and the real code, and reports the drift.

## What actually works today

```bash
pytest tests/     # 6 suites, tonic mocked — fast, safe, needs no datasets
```

That is the **only** quick check that works without a venv and without the ~23 GB of data.
Note it exercises the legacy bi-modal package, **not** M7 — passing tests do not validate
`code/m7_trimodal_run.py`.

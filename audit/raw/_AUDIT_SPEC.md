# AUDIT_SPEC — every audit agent MUST follow this exactly

## SAFETY (read-only, static analysis only)
- **NEVER import or execute ANY Python file.** In particular NEVER import/run
  `code/m7_trimodal_run.py` — it has no `__main__` guard and loads ~23 GB of datasets at
  import time, starting a 4–5 h run. Read files as **TEXT only** (Read/Grep/Glob).
- Ignore `code/__pycache__`. **Do not modify any `code/` file.** You only write the two
  output files below.

## YOUR OUTPUT — exactly two kinds of file, at these paths (create nothing else)
1. `audit/raw/<SHARD>.jsonl` — **one JSON object per line**, matching the schema below. This
   is the machine-readable spine downstream fixing agents parse. Emit STRICT JSON: double
   quotes, `\n`-escaped newlines inside strings, no trailing commas, exactly one object per
   line, no blank lines, no markdown fences.
2. `audit/by-file/<basename>.md` — one human-readable deep-dive **per file** you audit
   (basename without `.py`). Narrative + the same findings expanded, grouped by severity.

## FINDING SCHEMA (every JSONL line)
```
{"id":"<PREFIX>-001","file":"code/X.py","line":N,"line_end":M,"cluster":"<CLUSTER>","category":"<cat>","severity":"<sev>","status":"live","title":"...","description":"...","evidence":"...","fix":"...","fix_locations":["code/X.py:N-M"],"effort":"...","confidence":"...","related":[],"tags":[]}
```
- `id`: use YOUR assigned `<PREFIX>`, numbered `001,002,...` (unique by construction — ignore other agents).
- `category` ∈ `correctness, research-integrity, numerics, safety, architecture, duplication, dead-code, quality, dependencies, performance, naming, docs`
- `severity` ∈ `critical, high, medium, low, info`:
  - **critical** = breaks correctness/research-integrity, destroys data, or triggers the 4–5 h run / large download unexpectedly.
  - **high** = wrong results or a crash on a normal path; fabricated/mislabeled output.
  - **medium** = fragile, silently wrong under some inputs, or a real maintainability hazard.
  - **low** = style / minor dead code / naming. **info** = observation, no action required.
- `status` = `"live"` for your files unless your assignment says otherwise.
- `fix` = concise "what to change and why" — **NO full patch/diff**. `fix_locations` = every
  site to edit (`path:line` or `path:start-end`), including clones in other files if you know them.
- `confidence` ∈ `high, medium, low`; `effort` ∈ `trivial, small, medium, large`.
- **Required**: `id, file, line, category, severity, status, title, description, fix, fix_locations, confidence`. Others optional but encouraged.
- Whole-file findings: use `line:1`.

## COVERAGE — audit EVERY angle for each file; do not stop at the obvious
- **correctness / logic**: wrong metric, off-by-one, silent wrong branch, dead branch, mutable default arg, variable shadowing, swallowed exceptions, empty-container edge cases.
- **research-integrity (RULE 1–5)**: hardcoded/fabricated numbers, validation-vs-test confusion, accuracy reported without spike-rate/sparsity + timestep cost, mislabeled statistics (e.g. CI vs spread), seed-policy deviations.
- **numerics**: seed/RNG handling and consumption order, device/MPS traps, dtype/reduction/`eps` choices that could move results, determinism gaps.
- **safety**: import-time side effects (no `__main__` guard), destructive/in-place file writes, `torch.load` of paths, `subprocess`/`os.system`/`kill`, network/download at import.
- **architecture**: duplication across files, coupling, module boundaries, dead/orphan code, signature collisions.
- **quality**: dead code, unused imports/vars, magic numbers, naming, complexity, missing/misleading docstrings, stale comments.
- **dependencies**: third-party imports absent from `code/requirements.txt`; repo-local imports needing `sys.path` hacks; broken references (files/paths/modules that don't exist).
- **performance**: clear inefficiency (eager full-dataset materialization, redundant recompute) — only where genuinely real.

## RULE 1 — the audit invents NO numbers
If you cite a number it must be one **literally present in the source**, and you must say
"hardcoded in code" — never imply it is a measured result. If unsure something is a bug, set
`confidence:"low"` and say why. Prefer precision over volume, but do not miss anything real.
Ground every finding in an exact `file:line`.

## Verified repo context you can rely on
- NO results CSV, NO checkpoints (`*.pt`), NO `ICONS_M7/` datasets, NO `.ipynb` anywhere.
- `BASE_DIR='./ICONS_M7'` is CWD-relative; only resolves when run from `code/`.
- `code/` has no `__init__.py`; scripts `sys.path.insert` the repo root.
- The real shared library is `m7_trimodal_run_provenance.py` (has a `__main__` guard,
  import-safe). `m7_trimodal_run.py` is a self-contained orphan copy (imported by nobody, no
  guard). The legacy bi-modal package lives at repo root (`Models/`, `Dataloaders/`,
  `training/`, `utils/`) and is out of scope except where a `code/` file breaks because of it.

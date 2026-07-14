# Dependency reconciliation — `code/` (cluster DEP-config)

**Shard:** `w8_dep` · **Prefix:** `DEP` · **Primary files:** `code/requirements.txt` (13 lines),
`code/.gitignore` · **Method:** static/text only — nothing imported or executed.

This is a repo-wide reconciliation of what `code/` *imports* against what
`code/requirements.txt` *declares*, plus a `.gitignore` audit. Findings are machine-readable in
`audit/raw/w8_dep.jsonl` (`DEP-001` … `DEP-018`); this document is the narrative and the same
findings expanded, grouped by severity.

## How the evidence was gathered

- Enumerated every `import` / `from … import` across **all 40 `.py` files** in `code/`
  (31 top-level + 9 in `code/archive/`) — 706 import lines total.
- Reduced to unique top-level modules and classified each as **stdlib**, **third-party**
  (must be declared), or **repo-local**.
- Cross-referenced every third-party module against the 13 pins in `requirements.txt`.
- Located every `sys.path.insert/append` and every `REPO_ROOT = …` to judge how repo-local
  imports actually resolve.
- Confirmed which declared pins are never imported, and verified the legacy repo-root packages
  (`Models/`, `Dataloaders/`, `training/`, `utils/`) physically exist.

## Classification of every top-level module imported by `code/`

**Third-party — DECLARED in `requirements.txt` (8):** `torch`, `matplotlib`, `snntorch`,
`numpy`, `tonic`, `pandas`, `tqdm`, `scipy`. All are imported; all correctly present.

**Third-party — NOT declared (6):** `seaborn` (DEP-004), `sklearn`→scikit-learn (DEP-005),
`huggingface_hub` (DEP-006), `brian2` (DEP-007), `pyNN` (DEP-008), `wandb` (DEP-009).

**Declared but NEVER imported anywhere in `code/` (unused pins, 4):** `transformers`,
`torchaudio`, `torchvision`, `h5py` — see DEP-010. (`pytest` is also declared and never imported
under `code/`; handled in DEP-003 because it is also a version-floor concern.)

**Repo-local — legacy package at repo root (resolve only via `sys.path` hacks):** `Models`,
`Dataloaders`, `training`, `utils`. All target modules exist on disk (verified). See DEP-011.

**Repo-local — modules inside `code/`:** `m7_trimodal_run_provenance`,
`run_joint_magnitude_ablation`, `run_correspondence_anchor_v2`, `run_ordinal_anchor`,
`run_cl_confirmation`. The last one is a broken reference — see DEP-014.

**Stdlib (no action):** `os, sys, time, json, pathlib, typing, __future__, dataclasses, csv,
copy, traceback, subprocess, datetime, argparse, math, gc, collections, warnings, itertools,
hashlib, re, pickle`.

> **RULE 1 note:** every version number quoted below is a *pin literally present in
> `code/requirements.txt`* — not a measured result. No "correct" replacement version is asserted;
> where a pin is suspect the fix is to **verify it resolves for CPython 3.11**, which is the
> documented target interpreter (system default is 3.14 and will not satisfy the pins).

---

## HIGH severity

### DEP-001 — Future-dated pins likely unresolvable on PyPI for py3.11
`code/requirements.txt:1-3,6,8,9,12`

Seven pins are dated ahead of the Jan-2026 knowledge cutoff and cannot be confirmed to exist for
`cp311`:

| Line | Pin |
|---|---|
| 1 | `torch==2.11.0` |
| 2 | `torchaudio==2.11.0` |
| 3 | `torchvision==0.26.0` |
| 6 | `transformers==5.7.0` |
| 8 | `pandas==3.0.2` |
| 9 | `matplotlib==3.10.9` |
| 12 | `scipy==1.17.1` |

Because the repo has **no venv**, `pip install -r code/requirements.txt` *is* the documented
bootstrap. If any single version does not exist for `cp311`, the whole resolve fails and the
environment cannot be built. **Fix:** dry-run each (`pip index versions <pkg>` /
`pip install --dry-run`) against py3.11; replace any nonexistent version with the newest release
that supports 3.11; pin the `torch`/`torchaudio`/`torchvision` trio to a mutually compatible,
published triple.

### DEP-002 — `numpy==1.26.4` is inconsistent with the rest of the stack
`code/requirements.txt:7`

`numpy==1.26.4` is a numpy **1.x** release that predates every other (future-dated) pin. Modern
`scipy` (1.17.1, L12) and `pandas` (3.0.2, L8) almost certainly floor at `numpy>=2`, so a resolver
will likely reject 1.26.4 alongside them; even if it resolved, a numpy-1.x ABI under 2026-era
`torch`/`scipy` binaries risks an import-time binary-incompatibility. **Fix:** reconcile numpy
against the numpy floors of the chosen `scipy`/`pandas`/`torch` — most likely a numpy 2.x that
satisfies all three (verified for cp311) — rather than pinning numpy independently.

### DEP-004 — `seaborn` imported, not declared
`code/m7_mps.py:43`, `code/m7_raw.py:43`, `code/m7_smoke_test.py:43`, `code/m7_trimodal_cl.py:43`

`import seaborn as sns` at **module top level** (unguarded) in four live scripts. A clean install
then importing/running any of them raises `ImportError` before any work. **Fix:** add `seaborn`
to `requirements.txt`.

### DEP-005 — scikit-learn (`sklearn`) imported, not declared
`code/m7_mps.py:70-71`, `code/m7_raw.py:62-63`, `code/m7_smoke_test.py:70-71`, `code/m7_trimodal_cl.py:70-71`

`from sklearn.manifold import TSNE` and `from sklearn.metrics import (...)` at module top level in
four live scripts. **Naming trap:** the pip package is **`scikit-learn`**, import name `sklearn` —
the requirement must read `scikit-learn`. **Fix:** add `scikit-learn`.

### DEP-006 — `huggingface_hub` not declared → RULE 2 backups silently skip
`code/m7_mps.py:2170`, `code/m7_raw.py:2147`, `code/m7_trimodal_cl.py:2097`

`from huggingface_hub import HfApi` lives inside `upload_to_hf()` and the whole block is wrapped in
`try: … except Exception as e: print("… HF upload skipped …")`. On a clean install (no
`huggingface_hub`) the `ImportError` is **swallowed** — the run prints a skip message and continues
with **no backup**. RULE 2 mandates a HuggingFace backup after every seed and the team has lost
data once, so a missing pin turns every backup into a silent no-op. (Note: the canonical runners
`m7_trimodal_run.py` and `m7_trimodal_run_provenance.py` contain no HF code at all.) **Fix:** add
`huggingface_hub`; and narrow the `except` so a missing backup dependency is *loud* rather than
silently swallowed.

---

## MEDIUM severity

### DEP-003 — `pytest>=9.0` floor may be unsatisfiable, and pytest is unused in `code/`
`code/requirements.txt:13`

At cutoff pytest was 8.x; a `>=9.0` floor may not resolve, breaking the install. Separately,
**`pytest` is never imported anywhere under `code/`** (0 hits) — the suites live in the repo-root
`tests/` dir, out of scope for `code/requirements.txt`. **Fix:** verify a pytest 9.x exists for
py3.11 (else relax the floor); consider moving it to a dev/test extra.

### DEP-007 — `brian2` imported, not declared (archive)
`code/archive/{debug_brian2_exact.py:14, verify_brian2_raw.py:20, verify_full.py:21, verify_simulators.py:22, verify_tiny.py:9}`

Five archive Brian2/SpiNNaker verification scripts import `brian2`; undeclared → `ImportError` on
a clean install. **Fix:** add `Brian2` as an optional/hardware extra (not core runtime).

### DEP-008 — `pyNN` imported, not declared (archive)
`code/archive/{debug_brian2_exact.py:13, verify_full.py:20, verify_simulators.py:21, verify_tiny.py:8}`

`import pyNN.brian2 as sim` in four archive scripts; pip package `PyNN`; undeclared. **Fix:** add
`PyNN` as an optional/hardware extra (it also pulls Brian2).

### DEP-010 — Declared pins never imported anywhere in `code/`
`code/requirements.txt:2,3,6,11`

| Line | Pin | Status | Recommendation |
|---|---|---|---|
| 6 | `transformers==5.7.0` | **Never imported.** Appears only in stale notebook cruft — `m7_raw.py:22` `!pip install -q transformers`, and `… :27` `print("Newly installed: … transformers, datasets")`. | **Drop it.** Heavy dep, adds resolver risk for nothing. |
| 2 | `torchaudio==2.11.0` | Not imported directly. | Likely a **transitive** dep of the torch/tonic stack — verify, keep if so. |
| 3 | `torchvision==0.26.0` | Not imported directly. | Likely **transitive** (tonic) — verify, keep if so. |
| 11 | `h5py==3.16.0` | Not imported directly. | tonic reads SHD from HDF5 → likely needed **transitively** — verify, keep if so. |

Only `transformers` is clearly removable. Do not blindly delete the three transitive candidates.

### DEP-011 — Legacy repo-root imports resolve only via runtime `sys.path` hacks + correct CWD
`code/run_ablation.py:35-45` and the `utils.cross_modal_transfer` importers
(`eval_transfer_fixed.py:29`, `run_cl_rehearsal.py:44`, `run_correspondence_anchor_v2.py:45`,
`run_cross_modal_alignment_study.py:61`, `run_digit_only_alignment.py:64`,
`run_ordinal_anchor.py:48`, `run_rehearsal_transfer_eval_only.py:19`)

`code/` has **no `__init__.py`**, so scripts reach the repo-root legacy package (`Models/`,
`Dataloaders/`, `training/`, `utils/`) by `sys.path.insert(0, REPO_ROOT)` at runtime. All target
modules exist (`Models/registry.py`, `training/pipeline.py`, `training/settings.py`,
`Dataloaders/*_loader.py`, `utils/cross_modal_transfer.py` — verified), so these resolve **when**
the insert runs and the CWD is right. But the pattern is fragile: imports must physically follow
the insert (hence the scattered `# noqa: E402`), and any tool importing the module without the
insert (linter, IDE, test collector, different CWD) fails. ~17 `code/` files perform the
`sys.path.insert(0, str(REPO_ROOT))` dance. **Fix:** make the legacy code a real installable
package (`pip install -e`), or add a `conftest`/path shim, or convert `code/` into a package.

### DEP-012 — Hardcoded foreign absolute path in `sys.path.insert`
`code/benchmark_mps_real.py:6` (live) · `code/archive/benchmark_mps.py:6` · `code/archive/debug_gradient.py:5`

All three do `sys.path.insert(0, "/Users/agent1/Documents/neuromorphic/m7-trimodal-cl")` —
**another user's directory** that does not exist here (repo is under `/Users/zainhassan/…`). The
subsequent `from Models.registry import build_model` / `from training.pipeline import train_step`
then cannot resolve → **guaranteed `ImportError`** when the live `benchmark_mps_real.py` is run,
and a leak of a foreign filesystem layout. **Fix:** use `Path(__file__).resolve().parent.parent`
or proper packaging.

### DEP-013 — Archive scripts compute `REPO_ROOT` one level too shallow
`code/archive/run_cl_confirmation.py:49-51` · `code/archive/run_correspondence_anchor.py:39-41`

`REPO_ROOT = Path(__file__).resolve().parent.parent`. For a file at `code/archive/X.py` this
resolves to `code/`, **not** the repo root. So `sys.path.insert(REPO_ROOT / "code")` → `code/code/`
(nonexistent) and `sys.path.insert(REPO_ROOT)` adds `code/`. The `from m7_trimodal_run_provenance
import …` still works (that module is in `code/`), but `from Dataloaders…` (`run_cl_confirmation.py:204`)
/ `from utils.cross_modal_transfer…` (`run_correspondence_anchor.py:51`) cannot — the legacy
package is at the repo root, which is never added. The depth was not updated when the files were
moved into `archive/`. **Fix:** use `parent.parent.parent` for archive-depth files (or search
upward for a repo-root marker).

### DEP-014 — `import run_cl_confirmation` targets a module that exists only in `archive/`
`code/run_digit_only_alignment.py:616-617`

Under the `--confirm` CLI branch: `import run_cl_confirmation; run_cl_confirmation.main()`. There is
**no `run_cl_confirmation.py` in `code/`** (only `code/run_cl_confirmation_fixed.py`); the sole
`run_cl_confirmation.py` lives in `code/archive/`, which is never added to `sys.path` (the script
inserts only `REPO_ROOT/"code"` and `REPO_ROOT`). Running with `--confirm` raises `ImportError`.
Almost certainly a stale reference left after the rename to `run_cl_confirmation_fixed.py`.
**Fix:** import `run_cl_confirmation_fixed` (verify it exposes `main()`), or relocate the module
onto the path.

### DEP-015 — Provenance shared lib imports `Dataloaders` without fixing its own `sys.path`
`code/m7_trimodal_run_provenance.py:1251`

`from Dataloaders.paired_digit_loader import get_paired_digit_loaders` (inside a function) but the
module never inserts the repo root itself. It works only because every current importer runs
`sys.path.insert(0, REPO_ROOT)` first. Invoked via its own `__main__` guard, or imported by
anything that has not fixed the path, this deferred import raises `ImportError`. **Fix:** guarded
self-insert at top of module (or packaging, DEP-011).

---

## LOW / INFO

### DEP-009 (low) — `wandb` imported but not declared (optional, guarded)
`code/m7_mps.py:84`, `code/m7_raw.py:76`, `code/m7_smoke_test.py:84`, `code/m7_trimodal_cl.py:84`

`import wandb` is inside `try/except ImportError` with a `WANDB_AVAILABLE` flag → degrades
gracefully, no crash. **Not in the seed list — added here.** But because it is never declared, W&B
logging is silently unavailable on any clean install. **Fix:** declare `wandb` as an optional
extra.

### DEP-016 (low) — Stale `.gitignore` comment about notebooks
`code/.gitignore:34-35`

Comment reads `# Jupyter checkpoints (keep notebooks, ignore checkpoints)`, implying tracked
notebooks. Per CLAUDE.md there is **no `.ipynb` anywhere** in the repo. Misleading comment; the
`.ipynb_checkpoints/` rule itself is harmless. **Fix:** drop/reword the comment.

### DEP-017 (info) — `code/.gitignore` is narrower than repo-root; global HDF5 ignore
`code/.gitignore:16-21,45-46`

No defect. `code/.gitignore` ignores only `ICONS_M7/datasets/` and `ICONS_M7/results/checkpoints/`,
plus `*.pt/*.pth/*.ckpt/*.h5/*.hdf5`; it does **not** ignore results CSVs or figure PNGs. Those are
nonetheless protected because the **repo-root** `.gitignore` ignores `ICONS_M7/`, `code/ICONS_M7`
and `results/`. So the two files overlap, and `code/.gitignore` alone would not shield a CSV
written from a different CWD. Also the global `*.h5/*.hdf5` rule would exclude any legitimately
tracked HDF5 fixture. **Fix (optional):** consolidate; scope the HDF5 ignore to dataset dirs if a
tracked `.h5` is ever needed.

### DEP-018 (info) — Seed correction: `datasets` is NOT a missing dependency
`code/m7_mps.py:67` (+ `m7_nmnist_smoke_test.py:16`, `m7_raw.py:59`, `m7_smoke_test.py:67`, `m7_trimodal_cl.py:67`)

A prior seed listed `datasets` among undeclared third-party imports. **False positive — do not act
on it.** `datasets` appears only as `from tonic import datasets, transforms` — the `datasets`
*submodule of tonic*, which IS declared (`tonic==1.6.0`). The HuggingFace `datasets` package is
never imported; the bare word also shows up only in stale `print`/`!pip install` cruft. **Do NOT
add a `datasets` package to `requirements.txt`.**

---

## Concrete fix guidance for `code/requirements.txt`

**Add (missing runtime imports):**

```
seaborn                 # DEP-004 — m7_mps/m7_raw/m7_smoke_test/m7_trimodal_cl top-level import
scikit-learn            # DEP-005 — import name is sklearn; pip name is scikit-learn
huggingface_hub         # DEP-006 — required for RULE 2 backups (currently silently skipped)
```

**Add as optional/extras (guarded or archive-only):**

```
wandb                   # DEP-009 — optional logging (try/except guarded)
Brian2                  # DEP-007 — archive hardware-sim scripts only
PyNN                    # DEP-008 — archive hardware-sim scripts only
```

**Sanity-check every pin against CPython 3.11 (verify it resolves — do not assume a number):**
`torch==2.11.0`, `torchaudio==2.11.0`, `torchvision==0.26.0`, `transformers==5.7.0`,
`pandas==3.0.2`, `matplotlib==3.10.9`, `scipy==1.17.1` (DEP-001); reconcile `numpy==1.26.4`
against the scipy/pandas/torch numpy floors (DEP-002); confirm `pytest>=9.0` is satisfiable and
move it to a dev extra (DEP-003).

**Consider removing / re-scoping:** `transformers` (DEP-010, unused cruft); verify whether
`torchaudio` / `torchvision` / `h5py` are needed transitively before removing (they most likely
are — keep if so).

## Seed reconciliation

- **Version pins (seed, high):** confirmed and itemized — DEP-001 (7 future-dated pins), DEP-002
  (numpy inconsistency), DEP-003 (pytest floor). Target interpreter is python3.11 as stated.
- **Missing third-party (seed, high):** confirmed `seaborn`, `scikit-learn`, `huggingface_hub`,
  `brian2`, `pyNN` (DEP-004/005/006/007/008) with exact importer lines. **Corrected:** `datasets`
  is *not* a real missing dep (it is `tonic.datasets`) — DEP-018.
- **Expanded beyond the seed:** `wandb` undeclared (DEP-009); four unused pins incl.
  transformers-as-cruft (DEP-010); the `sys.path`/repo-local fragility family
  (DEP-011/012/013/014/015), notably the hardcoded `/Users/agent1` path in a **live** file and the
  broken `import run_cl_confirmation`; `.gitignore` findings (DEP-016/017).

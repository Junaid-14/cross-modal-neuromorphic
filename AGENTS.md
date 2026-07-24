# Repository Guidelines

> **Claude Code reads `CLAUDE.md`, not this file.** `CLAUDE.md` is the authoritative map and
> is verified against the actual tree. Keep the two in sync; where they differ, `CLAUDE.md`
> wins.

## Existing Project Description
- The existing repository contains experiments supporting the paper: **"Modality-Dependent Memory Mechanisms in Cross-Modal Neuromorphic Computing"** (IEEE Computer, minor revisions).
- The implementation benchmarks 5 models (`Models/baseline.py`, `Models/model_2_scl.py`, `Models/model_3_hopfield.py`, `Models/model_4_hgrn.py`, and `Models/model_5_hybrid.py`) against datasets N-MNIST and SHD. This is the **legacy bi-modal codebase**.
- The **current production system** is the M7 tri-modal continual learning pipeline in `code/m7_trimodal_run.py`, which learns vision (N-MNIST) → audio (SHD) → motion (DVS-Gesture) sequentially without catastrophic forgetting. This is the NeurIPS 2026 submission.
- Legacy notebook driver: **removed from the repo.** No `.ipynb` exists any more. `code/modify_notebook.py` is the only surviving record of its cell structure, and its target file is gone — so references elsewhere to "Cell 11" / "Cell 30" refer to nothing.
- Production driver code: `code/m7_trimodal_run.py` and `code/m7_mps.py`.

## Goal
- Maintain and extend the M7 tri-modal continual learning system.
- Run ablation studies and new experiments using the `code/` pipeline.
- You may need to write new Dataloaders in `Dataloaders/` to support additional Tonic datasets.
- You may need to refactor existing code to improve modularization or adapt to new experiments.

## Project Structure & Module Organization
```
cross-modal-neuromorphic/
├── code/           all executable code (training scripts, figure generation); code/archive/ = superseded
├── docs/           documentation & lab notes (8 flat files; NOT the multi-folder vault the indexes claim)
├── Models/         legacy bi-modal SNN architectures + memory mechanisms (baseline, SCL, Hopfield, HGRN, hybrid)
├── Dataloaders/    legacy dataset loaders (N-MNIST, SHD, SSC, DVS-Gesture, CIFAR10-DVS, N-TIDIGITS18)
├── training/       legacy training utilities (losses, pipeline, settings)
├── utils/          legacy plotting, dataset checks, production helpers
├── tests/          pytest suites (backbones, models, dataloaders, training, transfer) — tonic mocked
├── paper/          cross_modal_correspondence.tex (the controlled-negative-result paper)
├── reports/        sPyNNaker auto-generated provenance SQLite — not research output
└── .claude/        the research harness (agents, skills, hooks, rules, workflows)
```

**Directories that do NOT exist**, despite being referenced by older docs:
`configs/` · `results/` · `figures/` · `notebook/` · `experiments/` · `ICONS_M7/` · `code/.venv/`.
Also absent: any `.ipynb`, any `.bib`, any results CSV, any `.pt` checkpoint, and
`code/test_spinnaker_*.py` (gitignored). Run `/check-repo-state` to see the live truth.

### Key files in `code/`
| File | Purpose |
|------|---------|
| `m7_trimodal_run.py` | Main production training script (3 seeds, MPS/CUDA/CPU) |
| `m7_mps.py` | MPS-compatible monolithic script (Apple Silicon) |
| `generate_figures_3seed.py` | Publication-quality figure generation |
| `generate_figures.py` | Legacy single-seed figure generation |
| `m7_smoke_test.py` | Quick sanity checks for all three modalities |
| `convert_mps.py` | Utility: convert `m7_raw.py` → `m7_mps.py` |
| `modify_notebook.py` | Utility: injected DVS-Gesture support into the legacy notebook. **Its target notebook no longer exists** — running it will fail or corrupt. Kept only as a record of the notebook's cell structure. |

> `code/test_spinnaker_*.py` is **gitignored and not present**. Older docs tell you to run it;
> you cannot.

### Key files in `tests/`
| File | Purpose |
|------|---------|
| `test_backbones.py` | Forward-pass shape tests for VisualBackbone, AudioBackbone, build_backbone |
| `test_models.py` | Forward-pass tests for all 5 registered model variants |
| `test_dataloaders.py` | Collate/shape tests for all loaders (uses mocked tonic datasets) |
| `test_training.py` | 1-epoch training loop smoke tests + early-stopping validation |

### Key files in `docs/`
| File | Purpose |
|------|---------|
| `M7_Architecture_Deep_Dive.md` | System architecture and design rationale |
| `EXPERIMENTAL_MANIFEST.md` | Reproducibility doc (seeds, hyperparameters, expected numbers) |
| `SpiNNaker_Port_Plan.md` | Neuromorphic hardware deployment plan |

## Build, Test, and Development Commands
There is no build system or CLI entrypoint. **There is also no virtualenv** — `code/.venv`
does not exist and must be created first. Use `python3.11` (the system default is 3.14, which
will not satisfy the pins).

```bash
# Create the environment (once) — or use the /bootstrap-env skill
python3.11 -m venv code/.venv
source code/.venv/bin/activate
pip install -r code/requirements.txt

# Run tests — the ONLY fast, safe validation loop (tonic is mocked; no datasets needed)
pytest tests/

# Run main training — 4-5 hrs on MPS; needs ~23 GB of datasets that are NOT in the repo
python code/m7_trimodal_run.py

# Generate publication figures — requires a results CSV, which does not exist yet
python code/generate_figures_3seed.py
```

⚠️ **`code/m7_trimodal_run.py` has no `if __name__ == "__main__"` guard and loads datasets at
import.** `python -c "import m7_trimodal_run"` does not check the imports — it starts the full
4–5 hour run. To inspect the module, read the file.

⚠️ **It also silently skips training when a matching `final_*.pt` checkpoint exists**, loading
the checkpoint instead. Stale checkpoints produce stale results with no warning.

Execute ad hoc scripts by importing modules in `Models/` and `Dataloaders/` from a Python
session — note these are the **legacy** package and are not used by `code/m7_trimodal_run.py`.

## Coding Style & Naming Conventions
- Follow PEP 8; keep functions/classes descriptive and aligned with the existing `Models/` naming (e.g., `model_2_scl.py`).
- Add docstrings to public functions/classes; type hints are encouraged.
- Use lowercase module filenames with underscores; keep new model files alongside existing ones in `Models/`.
- Scripts in `code/` should be runnable from the repository root (e.g., `python code/script.py`).

## Testing Guidelines
- **`tests/` exists and contains 6 working pytest suites** (`test_backbones.py`, `test_models.py`,
  `test_dataloaders.py`, `test_training.py`, `test_cross_modal_transfer.py`,
  `test_paired_digit_loader.py`). Tonic is mocked, so they run fast and need **no datasets**.
  Run with `pytest tests/` (config in `pytest.ini`).
- ⚠️ They exercise the **legacy bi-modal package** (`Models/`, `Dataloaders/`, `training/`) —
  **not** `code/m7_trimodal_run.py`. **Passing tests do not validate M7.** Do not defend an M7
  change on the grounds that the tests pass.
- If you add tests, place them in `tests/` and run: `pytest tests/`.
- Aim for >80% coverage on new code where practical.

## Commit & Pull Request Guidelines
- Commit messages should be clear and concise; the suggested format is `Add feature: description`.
- Use feature branches (e.g., `feature/your-feature-name`).
- PRs should include a detailed description, link related issues, and note any performance changes; ensure tests pass before submission.

## Research Resources
- `resources/research/` is a symlink to the external Obsidian research vault. Do not move or rename it without updating `resources/rag/config.py`.
- The local RAG (`resources/rag/`) indexes `01_Literature`, `02_Concepts`, `03_Projects`, and `04_Meetings_and_Logs`.
- Run `python code/rag_ingest.py` after adding or editing notes in the vault.
- Claude skills in `.claude/skills/` rely on the RAG; keep them in sync with any RAG schema changes.

## Research/Analysis Notes
- New memory mechanisms or datasets should include a matching loader or model module, documentation updates, and tests in `tests/`.
- When adding new datasets from Tonic, create a loader in `Dataloaders/`, wire it into `Models/backbones.py`, and add mocked pytest tests to `tests/test_dataloaders.py`.
- If results change, document the comparison in the PR description. (Results and figures are
  gitignored and produced under `ICONS_M7/results/` at runtime — there is no tracked `results/`
  or `figures/` directory.)

## Research integrity — non-negotiable
These are enforced by the harness in `.claude/`; see `.claude/rules/research-integrity.md`.
1. **Never put a number anywhere that is not in a CSV.** No estimates.
2. **Back up to HuggingFace after every seed** — results/checkpoints are gitignored and this
   team has already lost data once.
3. **Anonymous submission** — self-cite as `\citep{anon2026modality}`.
4. **Report test accuracy, not validation** (SHD's val/test gap is 15–19 pp).
5. **The forgetting claim is ~0 pp**, not "100 % reduction."

Also: hypotheses are not results; accuracy always travels with spike rate / sparsity and
timestep count; never refactor in a way that silently changes numerics; never write outside
this repo.

## Automation Layer
- The assistants documented in `docs/AUTOMATION_TUTORIAL.md` are implemented in `code/automation/` and registered as Claude Code slash commands in `.claude/commands/`.
- All automation outputs must land under `knowledge/`; never write automation reports into `results/`, `code/`, or `paper/`.
- Automation scripts are allowed to read production code and results, but they must only **suggest** changes — they never edit `code/`, `Models/`, `paper/`, or numeric artifacts on their own.
- If you add a new automation assistant, add the matching `.claude/commands/<name>.md` file and update both `CLAUDE.md` and `docs/AUTOMATION_TUTORIAL.md`.
- Validate automation changes by running the script from the repo root (e.g., `python code/automation/improve_repo.py --dry-run`).

## Running environment
- This code base is deployed to a Runpod pod.
- runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

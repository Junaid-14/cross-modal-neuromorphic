# Repository Guidelines

## Existing Project Description
- The existing repository contains experiments supporting the paper: **"Modality-Dependent Memory Mechanisms in Cross-Modal Neuromorphic Computing"** (NeurIPS 2026 submission).
- The implementation benchmarks 5 models (`Models/model_1_baseline.py`, `Models/model_2_scl.py`, `Models/model_3_hopfield.py`, `Models/model_4_hgrn.py`, and `Models/model_5_hybrid.py`) against datasets N-MNIST and SHD. This is the **legacy bi-modal codebase**.
- The **current production system** is the M7 tri-modal continual learning pipeline in `code/m7_trimodal_run.py`, which learns vision (N-MNIST) → audio (SHD) → motion (DVS-Gesture) sequentially without catastrophic forgetting.
- Legacy driver code: `notebook/cross_modal_experiments_complete (5).ipynb` (superseded; kept for historical reference).
- Production driver code: `code/m7_trimodal_run.py` and `code/m7_mps.py`.

## Goal
- Maintain and extend the M7 tri-modal continual learning system.
- Run ablation studies and new experiments using the `code/` pipeline.
- You may need to write new Dataloaders in `Dataloaders/` to support additional Tonic datasets.
- You may need to refactor existing code to improve modularization or adapt to new experiments.

## Project Structure & Module Organization
```
m7-trimodal-cl/
├── code/           all executable code (training scripts, figure generation, SpiNNaker tests)
├── docs/           documentation & lab notes (Obsidian vault)
├── Models/         core SNN architectures and memory mechanisms (baseline, SCL, Hopfield, HGRN, hybrid)
├── Dataloaders/    dataset loaders (N-MNIST, SHD, SSC, DVS-Gesture, CIFAR10-DVS, N-TIDIGITS18)
├── tests/          pytest smoke tests (backbones, models, dataloaders, training pipeline)
├── notebook/       legacy Jupyter notebooks (superseded by code/ scripts)
├── training/       training utilities (losses, pipeline, settings)
├── experiments/    experiment runner and dataset configurations
├── utils/          plotting, dataset checks, production helpers
├── results/        output CSVs, pickles, and summary tables
├── figures/        generated plots and confusion matrices
├── configs/        experiment configurations
└── ICONS_M7/       datasets + checkpoints (gitignored; symlinked from code/ICONS_M7)
```

### Key files in `code/`
| File | Purpose |
|------|---------|
| `m7_trimodal_run.py` | Main production training script (3 seeds, MPS/CUDA/CPU) |
| `m7_mps.py` | MPS-compatible monolithic script (Apple Silicon) |
| `generate_figures_3seed.py` | Publication-quality figure generation |
| `generate_figures.py` | Legacy single-seed figure generation |
| `m7_smoke_test.py` | Quick sanity checks for all three modalities |
| `test_spinnaker_*.py` | SpiNNaker hardware port validation tests |
| `convert_mps.py` | Utility: convert `m7_raw.py` → `m7_mps.py` |
| `modify_notebook.py` | Utility: inject DVS-Gesture support into legacy notebook |

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
There is no build system or CLI entrypoint. Typical workflows are:

```bash
# Activate environment
source code/.venv/bin/activate

# Install dependencies
pip install -r code/requirements.txt

# Run tests
pytest tests/

# Run main training
python code/m7_trimodal_run.py

# Generate publication figures
python code/generate_figures_3seed.py

# Run smoke tests
python code/m7_smoke_test.py
```

For legacy bi-modal experiments:
```bash
jupyter notebook "notebook/cross_modal_experiments_complete (5).ipynb"
```

Execute ad hoc scripts by importing modules in `Models/` and `Dataloaders/` from a Python session.

## Coding Style & Naming Conventions
- Follow PEP 8; keep functions/classes descriptive and aligned with the existing `Models/` naming (e.g., `model_2_scl.py`).
- Add docstrings to public functions/classes; type hints are encouraged.
- Use lowercase module filenames with underscores; keep new model files alongside existing ones in `Models/`.
- Scripts in `code/` should be runnable from the repository root (e.g., `python code/script.py`).

## Testing Guidelines
- The repository references `pytest` usage and a `tests/` directory in `CONTRIBUTING.md`, but no tests are currently present.
- If you add tests, place them in `tests/` and run: `pytest tests/`.
- Aim for >80% coverage on new code where practical.

## Commit & Pull Request Guidelines
- Commit messages should be clear and concise; the suggested format is `Add feature: description`.
- Use feature branches (e.g., `feature/your-feature-name`).
- PRs should include a detailed description, link related issues, and note any performance changes; ensure tests pass before submission.

## Research/Analysis Notes
- New memory mechanisms or datasets should include a matching loader or model module, documentation updates, and tests in `tests/`.
- When adding new datasets from Tonic, create a loader in `Dataloaders/`, wire it into `Models/backbones.py`, and add mocked pytest tests to `tests/test_dataloaders.py`.
- If results change, document comparisons in the PR description and update `results/` or `figures/` as needed.
- When adding new datasets from Tonic, create a loader in `Dataloaders/` and wire it into `code/m7_trimodal_run.py`.

## Running environment
- This code base is deployed to a Runpod pod.
- runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

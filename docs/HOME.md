---
tags: [home, index, moc]
updated: 2026-06-02
---

# 🏠 M7 Project Home

> **One-line:** A single spiking neural network (M7) that learns three sensory
> modalities in sequence — vision → audio → motion — without catastrophic
> forgetting, using per-task heads + embedding replay, with a pipeline to deploy
> onto SpiNNaker neuromorphic hardware. Target venue: NeurIPS 2026.

This is the home page. Start here. Every fact below is grounded in the actual
repo — filenames, checkpoint names, seed values, and test scripts are real.

---

## 📌 Current Status (2026-06-02)

| Track | State | Next action |
|-------|-------|-------------|
| **Repo / infra** | ✅ Stable | Keep `neuromorphic 2` backup until results are committed |
| **EXP-1 Core M7 (CL)** | ✅ Trained, 3-seed checkpoints on disk | Write paper figures; validate CSV dedup |
| **EXP-2 Ablation study** | ⏸️ Paused | Pipeline verified (50-sample overfit → 100%); resume after SpiNNaker |
| **EXP-3 SpiNNaker deploy** | 🔄 In progress | Make e2e test do real inference (accuracy, not just mapping) |
| **Paper** | ✍️ Not started | Keep sim-vs-hardware claims separate |

---

## 🧪 Experiments

### EXP-1 — Core M7 Tri-Modal Continual Learning

**The main contribution.** A single SNN learns T1→T2→T3 without catastrophic
forgetting, using per-task heads + embedding replay (~4 MB of 512-d embeddings
vs ~80 MB raw data).

- **Tasks:** T1 N-MNIST (visual, 10 classes) → T2 SHD (audio, 20 classes) → T3 DVS-Gesture (visual-motion, 11 classes)
- **Mechanism:** Per-task heads isolate gradients; Modern Hopfield + HGRN memory modules; adaptive gate learns `[w_hop, w_hgrn]` routing per modality
- **Seeds:** `[42, 123, 456]` — all three complete for both `no_replay` and `replay` conditions
- **Checkpoints:** `ICONS_M7/results/checkpoints/` — 21 `.pt` files total (18 `final_*` + 3 `best_*` seed 42), named `final_{task}_{modality}_seed{N}_{condition}.pt`
- **CSV results:** `ICONS_M7/results/cl_trimodal_results.csv` — contains duplicate rows from reruns; filter `step.isna()` then `drop_duplicates()` before analysis
- **3-seed mean ± std (replay):** SHD 80.00 ± 1.54%, DVS 82.83 ± 1.96%, FWT 75.11 ± 1.24pp; forgetting negligible
- **Entry point:** `code/m7_trimodal_run.py` (self-contained, ~4–5 hrs on MPS)
- **Status:** ✅ Complete for all seeds
- **Detail:** [[M7 Project Overview]] · [[Tri-Modal CL Sequence]] · [[Experiment Log]] · [[EXPERIMENTAL_MANIFEST]]

**Key files:**

| File | Role |
|------|------|
| `code/m7_trimodal_run.py` | Reproducibility runner (3 seeds, both conditions) |
| `code/NeurIPS2026_M7_clean (3) (1).ipynb` | Canonical 55-cell notebook |
| `code/generate_figures_3seed.py` | Publication figures (mean ± std across seeds) |
| `Models/backbones.py` | `SNN_Backbone` (Conv2D visual) + `SNN_Backbone_SHD` (MLP audio) |
| `Models/hopfield_layer.py` | `ModernHopfieldLayer` (256 patterns × 512-d) |
| `Models/hgrn_layer.py` | `ImprovedHGRNGate` (512→512 with LayerNorm) |
| `training/pipeline.py` | `train_model()` loop with early stopping |
| `Dataloaders/nmnist_loader.py` | N-MNIST (34×34, T=25, 10 classes) |
| `Dataloaders/shd_loader.py` | SHD (700-ch, 100 bins, 20 classes) |
| `Dataloaders/dvs_gesture_loader.py` | DVS-Gesture (128→34 pool, T=25, 11 classes) |

---

### EXP-2 — Ablation Study (PyTorch)

**Supporting evidence.** Trains 5 registered model variants × 6 datasets, logging
accuracy + spike-rate / sparsity / timesteps, to show which M7 components matter.

- **Models:** `baseline`, `scl`, `hopfield`, `hgrn`, `hybrid` (registered in `Models/registry.py`)
- **Datasets:** N-MNIST (10), SHD (20), DVS-Gesture (11), SSC (35), CIFAR10-DVS (10), N-TIDIGITS18 (10)
- **Runner:** `code/run_ablation.py` — uses `SpikeRecorder` hooks on every `snn.Leaky` layer
- **Default seeds:** `[42, 123, 456]` (same as EXP-1)
- **Output:** `results/ablation_results.csv` — one row per (dataset, model, seed) with spike metrics
- **Status:** ⏸️ Paused. Pipeline verified (50-sample overfit test → 100%); SNNs on sparse event data learn slowly — 1 epoch at chance is expected, not a bug
- **⚠️ Note:** CIFAR10-DVS needs manual download (broken upstream mirror)
- **Detail:** [[Experiment Results]] · [[Datasets]]

**Key files:**

| File | Role |
|------|------|
| `code/run_ablation.py` | CLI runner — `--quick` for smoke test, full grid for paper |
| `Models/registry.py` | `build_model()` factory for all 5 variants |
| `Models/baseline.py` → `model_5_hybrid.py` | M1–M5 architecture definitions |
| `Dataloaders/*.py` | One loader per dataset (6 loaders) |
| `results/ablation_results.csv` | Output CSV (incremental save after every run) |

**CLI examples:**

```bash
python code/run_ablation.py --quick                                    # 1 epoch smoke
python code/run_ablation.py --datasets nmnist shd --models baseline scl # subset
python code/run_ablation.py --use-contrastive                          # with SCL
```

---

### EXP-3 — SpiNNaker Hardware Deployment

**The stretch-goal contribution.** Take trained M7 checkpoints and run them on
SpiNNaker neuromorphic hardware to demonstrate energy-efficient inference and
(optionally) on-chip continual learning.

- **Approach:** 3 phases — (1) inference-only, (2) host-mediated CL, (3) on-chip STDP (stretch)
- **Done:** Toolchain installed; FC backbones mapped with real weights (visual backbone 1.49M
  synapses, audio backbone 1.27M synapses); stress tests resolved (FromListConnector 256-row
  limit → chunked workaround in `test_spinnaker_fc1600to1024_chunked.py`)
- **⚠️ Critical gap:** `test_spinnaker_backbone_e2e.py` only *maps* weights onto a virtual
  board — it feeds Poisson noise and never decodes predictions. **Mapping ≠ inference.**
  The checkpoint it loads (`final_nmnist_nmnist_replay.pt`) may also be a stale path; verify
  against the actual files in `ICONS_M7/results/checkpoints/`.
- **Current task:** Extend e2e test to feed real N-MNIST data (pre-computed 1600-d conv features
  → Poisson rates), decode head spikes → argmax, report accuracy vs PyTorch baseline
- **⚠️ Watch:** `SCALE=0.05` and `THRESH=0.001` in e2e tests are arbitrary constants — calibrate
  before concluding the model fails on SpiNNaker
- **Hardware:** Everything so far runs on the **virtual board (simulator)**, NOT real silicon.
  Full M7 needs ~600 cores → requires SpiNN-5 (48 chips); SpiNN-3 (72 cores) is insufficient
- **Conv blocker:** sPyNNaker `ConvolutionConnector` flagged as unusable → host-side conv
  workaround planned (verify the bug claim against source before any bug report)
- **Detail:** [[SpiNNaker_Port_Plan]] · [[HARDWARE_ACCESS_CHECKLIST]]

**Key files (all in `code/`):**

| File | What it tests |
|------|---------------|
| `test_spinnaker_hello.py` | Toolchain validation (50 Poisson → 100 LIF, virtual board) |
| `test_spinnaker_fc512to10.py` | Smallest real-weight test (head: 512→10) |
| `test_spinnaker_fc1600to1024.py` | FC1 layer with real weights |
| `test_spinnaker_fc1600to1024_alltoall.py` | Same with `AllToAllConnector` |
| `test_spinnaker_fc1600to1024_chunked.py` | Chunked FC1 (fixes 256-row FromListConnector limit) |
| `test_spinnaker_backbone_e2e.py` | Full visual backbone (1600→1024→512→10) — maps only, no accuracy |
| `test_spinnaker_backbone_e2e_inference.py` | Inference variant (check status) |
| `test_spinnaker_audio_backbone.py` | Audio backbone mapping (700→1024→1024→512) |

---

## 🗂️ Repo Map

```
m7-trimodal-cl/
├── code/                          # All executable code
│   ├── m7_trimodal_run.py         # Reproducibility entry point (3 seeds)
│   ├── run_ablation.py            # Ablation study runner (5 models × 6 datasets)
│   ├── generate_figures.py        # Figures from seed 42 only
│   ├── generate_figures_3seed.py  # Publication figures (3-seed mean±std)
│   ├── test_spinnaker_*.py        # 8 SpiNNaker integration tests
│   ├── benchmark_mps.py           # MPS performance benchmarks
│   ├── NeurIPS2026_M7_clean (3) (1).ipynb  # Canonical 55-cell notebook
│   └── requirements.txt           # Python deps (torch, snntorch, tonic, etc.)
│
├── Models/                        # Model definitions
│   ├── registry.py                # build_model() factory
│   ├── baseline.py                # M1: Baseline SNN
│   ├── model_2_scl.py             # M2: +Supervised Contrastive Loss
│   ├── model_3_hopfield.py        # M3: +Hopfield
│   ├── model_4_hgrn.py            # M4: +HGRN
│   ├── model_5_hybrid.py          # M5: Hybrid (Hopfield+HGRN)
│   ├── backbones.py               # SNN_Backbone (Conv2D) + SNN_Backbone_SHD (MLP)
│   ├── hopfield_layer.py          # ModernHopfieldLayer (256×512-d)
│   ├── hgrn_layer.py              # ImprovedHGRNGate
│   └── engram_analysis.py         # Embedding/engram analysis utilities
│
├── Dataloaders/                   # One file per dataset
│   ├── nmnist_loader.py           # N-MNIST (visual, 10 classes, 34×34)
│   ├── shd_loader.py              # SHD (audio, 20 classes, 700-ch)
│   ├── dvs_gesture_loader.py      # DVS-Gesture (motion, 11 classes, 128→34)
│   ├── cifar10dvs_loader.py       # CIFAR10-DVS (visual, 10 classes)
│   ├── ssc_loader.py              # SSC (audio, 35 classes)
│   └── ntidigits18_loader.py      # N-TIDIGITS18 (audio, 10 classes)
│
├── training/                      # Training infrastructure
│   ├── pipeline.py                # train_model() with early stopping
│   ├── settings.py                # TrainingConfig dataclass
│   └── losses.py                  # SupervisedContrastiveLoss + replay loss
│
├── utils/                         # Shared utilities
│   ├── plotting.py                # Figure helpers
│   ├── dataset_checks.py          # Dataset validation
│   └── production.py              # Production utilities
│
├── tests/                         # Unit tests (pytest, mocked)
│   ├── test_dataloaders.py
│   ├── test_backbones.py
│   ├── test_models.py
│   └── test_training.py
│
├── experiments/                   # Experiment helpers
│   ├── datasets.py
│   └── runner.py
│
├── notebook/                      # Additional Jupyter notebooks
│   ├── cross_modal_experiments_DVS_SSC.ipynb
│   ├── cross_modal_experiments_refactored.ipynb
│   └── ieee-cross-modal-final (2).ipynb
│
├── configs/                       # Experiment configs (currently empty)
│
├── docs/                          # Obsidian vault — you are here
│   ├── HOME.md                    # This file
│   ├── EXPERIMENTAL_MANIFEST.md   # Full reproducibility spec
│   ├── M7_Architecture_Deep_Dive.md
│   ├── SpiNNaker_Port_Plan.md
│   ├── HARDWARE_ACCESS_CHECKLIST.md
│   ├── 01 - Notes/               # 11 project notes (see below)
│   ├── 00 - Inbox/               # Empty
│   ├── 02 - Daily/               # Empty
│   ├── 03 - References/          # Empty
│   └── 04 - Attachments/         # Empty
│
├── ICONS_M7/                      # Data + checkpoints (gitignored, symlinked)
│   ├── datasets/                  # ~23 GB (NMNIST, SHD, DVSGesture, etc.)
│   ├── results/
│   │   ├── cl_trimodal_results.csv  # EXP-1 results (has duplicate rows)
│   │   └── checkpoints/             # 21 .pt files (~862 MB total)
│   └── figures/                   # Generated plots
│
├── results/                       # EXP-2 ablation outputs
│   ├── ablation_results.csv
│   ├── training_summary.csv
│   ├── dataset_statistics.csv
│   ├── table_cross_modal_complete.csv
│   ├── table_energy_efficiency.csv
│   └── *.pkl (history objects)
│
└── figures/                       # Publication-quality figures
    ├── training_curves_all_models.png
    ├── confusion_matrix_model_{1-5}.png
    ├── tsne_assemblies_model_{1-5}.png
    ├── cross_modal_complete_analysis.png
    ├── energy_efficiency_comprehensive.png
    ├── engram_analysis_complete.png
    ├── spike_pattern_analysis.png
    └── noise_robustness_analysis.png
```

---

## ⚠️ Hard-Won Lessons (don't repeat these)

1. **One agent per directory.** Running kimi + another tool in the same folder caused
   repeated working-directory crashes. Don't overlap.

2. **`results/` and checkpoints are gitignored** → not recoverable from git if deleted.
   The `neuromorphic 2` backup saved the trained model once already. Keep it.

3. **Relative paths bite.** `BASE_DIR='./ICONS_M7'` only works when CWD is `code/`.
   The standalone `m7_trimodal_run.py` handles this; notebook cells assume `code/` as CWD.

4. **Mapping ≠ inference ≠ real hardware.** Three different claims with different
   evidence bars. Keep them distinct, especially in the paper.

5. **Trust but verify kimi's findings.** It read "flat loss" as progress (wrong) and
   diagnoses library bugs confidently — check specifics before they go in the paper.

6. **Buffer API changed.** Old code used `buffer.total_stored` (AttributeError) → now
   `buffer.stats()['total']`. Old code unpacked `buffer.sample()` as 3 values → now 2
   (`embs, lbls = buffer.sample(...)`).

7. **`hist_t3` scope bug.** Was initialized inside the epoch loop instead of before it,
   so only the last epoch's data survived. Fixed by moving initialization outside.

8. **DVS download is fragile.** `tonic.datasets.DVSGesture` can fail silently. The
   notebook wraps it in `try/except` with SSC-35 fallback (contact Blessing for loader).

9. **CSV dedup required.** `cl_trimodal_results.csv` has duplicate rows from reruns.
   Always filter `step.isna()` then `drop_duplicates()` before computing statistics.

10. **SNNs learn slowly on sparse data.** 1 epoch at chance accuracy is expected behavior
    (not a bug) for ablation runs. Verify with 50-sample overfit test before debugging.

---

## ✅ Open Questions / TODO

- [ ] **EXP-3: get an accuracy-on-SpiNNaker number** (virtual board). Feed real N-MNIST
      data, decode predictions, compare with PyTorch baseline.
- [ ] **EXP-3: confirm real hardware access.** Which board? SpiNN-5 needed (~600 cores);
      SpiNN-3 (72 cores) is insufficient without compression.
- [ ] **EXP-3: verify ConvolutionConnector bug** against sPyNNaker source before filing
      or claiming it in the paper.
- [ ] **EXP-3: calibrate `SCALE`/`THRESH`** in SpiNNaker tests — currently arbitrary.
- [ ] **EXP-2: CIFAR10-DVS manual download** for full ablation coverage.
- [ ] **EXP-2: full ablation run** (5 models × 6 datasets × 3 seeds) for paper Table.
- [ ] **Cleanup:** sync/delete stale root `requirements.txt`; add `pytest` to
      `code/requirements.txt`.
- [ ] **Paper:** write honest "what ran on-chip vs host" scope paragraph.
- [ ] **Paper:** generate final figures from `generate_figures_3seed.py` using deduped CSV.
- [ ] **Archival:** commit final checkpoints to Git LFS, HuggingFace, or Zenodo (862 MB,
      too large for standard GitHub).

---

## 📝 Notes Index

All project notes live in `docs/01 - Notes/`:

- [[M7 Project Overview]] — title, core idea, key components, run order
- [[NeurIPS 2026 — M7 Project Plan]] — architecture diagram, baseline results, environment setup
- [[M7 Architecture Diagram]] — ASCII schematic + per-component equations
- [[Tri-Modal CL Sequence]] — T1→T2→T3 protocol, hyperparameters, replay buffer spec
- [[Datasets]] — N-MNIST, SHD, DVS-Gesture preprocessing details + download risks
- [[Experiment Log]] — chronological log of changes and next runs
- [[Experiment Results]] — seed 42 detailed results, gate specialisation, efficiency analysis
- [[Notebook Guide]] — cell-by-cell guide to the 55-cell notebook
- [[DVS-Gesture Smoke Test]] — validation protocol, accuracy guards, fallback
- [[Team Rules & Runbook]] — cell execution order, HF uploads, emergency contacts
- [[Literature Review]] — Hopfield, HGRN, SCL, CL methods, neuromorphic datasets, gap analysis

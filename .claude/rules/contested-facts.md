# Contested facts — do not assert these without checking

Places where the documents disagree with each other, or with the code. **If a task depends on
one of these, resolve it against the code first, and say which source you used.**

Items 1–2 were resolved on **2026-07-14** by reading the source. Items 3–6 remain open.

---

## ✅ 1. RESOLVED — Hopfield pattern count: it was never a contradiction

**M7 uses 100. The legacy bi-modal package uses 256. Both numbers are correct.**

| Codebase | Value | Where |
|---|---|---|
| **M7 tri-modal** (Paper 3) | **100** | `CONFIG['num_patterns'] = 100` → `ModernHopfieldLayer(dim=512, num_patterns=CONFIG['num_patterns'])` in `code/m7_trimodal_run.py` |
| **Legacy bi-modal** (Paper 2) | **256** | `Models/hopfield_layer.py`: `memory_size=256`; constructed as such in `model_3_hopfield.py` and `model_5_hybrid.py` |

`docs/HOME.md` says "256 patterns × 512-d" **about `Models/hopfield_layer.py`** — and it is
**right**. Do not "fix" it.

⚠️ **The trap that makes this look like a contradiction:** M7's `ModernHopfieldLayer.__init__`
has the signature `def __init__(self, dim=512, num_patterns=256, ...)` — a **default of 256**.
But the call site overrides it with `CONFIG['num_patterns']`, which is **100**. Anyone reading
the class signature instead of the call site concludes 256. Read the call site.

## ✅ 2. RESOLVED — Visual backbone: it is 2 conv layers, not 3

`code/m7_trimodal_run.py`, `class SNN_Backbone`:

```
conv1 = Conv2d(2, 32, 5) → MaxPool2d(2) → LIF
conv2 = Conv2d(32, 64, 5) → MaxPool2d(2) → LIF
fc1   = Linear(64*5*5 = 1600, 1024) → LIF
fc2   = Linear(1024, 512) → LIF
fc3   = Linear(512, num_classes)   # unused in full M7 — per-task heads take over
```

- `docs/M7_Architecture_Deep_Dive.md` and `docs/SpiNNaker_Port_Plan.md` were **correct**.
- `docs/EXPERIMENTAL_MANIFEST.md` §6.2 claimed a **3-conv** backbone (`conv3 = Conv2d(64,128,3)`,
  `fc1 = Linear(128*2*2, 256)`, `fc2 = Linear(256,512)`). That matched **neither** M7 **nor**
  the legacy `Models/backbones.py` (which is `conv1(2→64,k3) → conv2(64→128,k3)`). It was
  simply wrong. **Corrected 2026-07-14.**

**Why this mattered:** `fc1 = 1600 × 1024 = 1,638,400 weights` → 6.55 MB → **~205 cores**, which
is exactly the figure the SpiNNaker budget rests on. The 3-conv spec implied a different fan-in
and would have produced a wrong core estimate.

---

## ⚠️ 3. OPEN — Three different result sets circulate

Not contradictory — they are **different experiment stages** — but they get conflated. Always
say which one you mean.

| Result set | Source | Numbers |
|---|---|---|
| Tri-modal, 3-seed, +replay | `EXPERIMENTAL_MANIFEST.md`, `HOME.md` | SHD 80.00 ± 1.54 %, DVS 82.83 ± 1.96 %, FWT 75.11 ± 1.24 pp |
| Two-task, earlier | `M7_Team_Onboarding.docx` | N-MNIST 97.74 ± 0.04 %, SHD 82.21 ± 0.41 %, FWT 72.87 ± 1.56 pp |
| Single-task, standalone | correspondence paper | N-MNIST 97.21 %, SHD 79.64 % |

🚨 **None of these has a CSV in the repo to back it.** There is no results CSV at all. Under
RULE 1 every one of them is **UNSOURCED** until a run produces data. See RULE 1.

## ⚠️ 4. OPEN — The two narratives about cross-modal transfer

Both defensible, trivially conflated into an overclaim.

- **`M7_Team_Onboarding.docx` + `docs/HOME.md`**: forward transfer as a **positive** result
  (FWT ≈ 75 pp) — "visual spikes scaffold auditory learning."
- **`paper/cross_modal_correspondence.tex` + `docs/paper_sections_correspondence.md`**:
  cross-modal *correspondence* is **engineered, not emergent** — held-out retrieval at chance
  (Linear 21.88 ± 19.52 % vs 25 % chance), Procrustes 0 % held-out, RDMs uncorrelated
  (Mantel r = 0.051, p = 0.74). *"A supervised bridge, not a discovered shared manifold."*

**Never let a summary imply an emergent modality-invariant semantic space.** The M7 CL result is
attributed to **architectural isolation (per-task heads) + replay**, not to shared semantics.

## ⚠️ 5. OPEN — Checkpoint count: 18 or 21?

`EXPERIMENTAL_MANIFEST.md` says 18 files (~862 MB); `HOME.md` says 21 (18 `final_*` + 3 `best_*`
for seed 42). Moot today — **there are zero checkpoints in the repo.**

## ⚠️ 6. OPEN — SpiNNaker claims

Nothing has run on a board. `docs/HARDWARE_ACCESS_CHECKLIST.md` is entirely unchecked. `reports/`
holds sPyNNaker provenance from **local** runs, one of which **errored during `setup()`**. Any
energy or hardware-performance number is unsourced — a simulator cannot produce energy figures.
**Mapping ≠ inference ≠ real hardware.**

---

## Fixed on 2026-07-14 (for the record, so they don't creep back)

- **`SEEDS` in `code/m7_trimodal_run.py`** was `[123, 456]` — seed 42 never ran, silently making
  every "3-seed mean ± std" a **2-seed** number. Now `[42, 123, 456]`.
- **Figure 3 (gate trajectory) was fabricated** from hardcoded literals. The training loop *does*
  measure the per-epoch `gate_hop`/`gate_hgrn` split — it printed it to stdout and then discarded
  it. It now persists to `gate_trajectory.csv` (`_save_gate_history`), and the figure is drawn
  from that or **skipped entirely** if the file is absent. It is never fabricated.
- **Figure 4 (memory) was hardcoded** `[80, 150, 4]` MB. Now derived from CONFIG
  (`samples_per_class × classes × 512 × 4 B`) with the arithmetic printed for audit.
- **Figure 1 had hardcoded reference lines** at `y=97.68` / `y=82.16` labelled "single-task" —
  actually M3 and M2 numbers from the **legacy bi-modal paper**, mislabelled and overlaid on a
  tri-modal plot. Removed.
- **`generate_figures_3seed.py` aggregated over CSV rows, not unique seeds**, so a re-run leaving
  a near-duplicate row silently inflated `n` and corrupted the ± std. Now collapses to one row per
  `(seed, condition)` and warns.
- **Stale doc claims** now corrected in `CLAUDE.md` / `AGENTS.md`: there is no notebook (so
  "Cell 11" / "Cell 30" refer to nothing), no `configs/`, `results/`, `figures/`, `experiments/`,
  `ICONS_M7/`, no `code/.venv/`, no `code/test_spinnaker_*.py`, and no `.bib`. `tests/` **does**
  exist with 6 working suites.

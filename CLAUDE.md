# CLAUDE.md — M7 Tri-Modal Continual Learning

Guidance for Claude Code in this repository. **Verified against the actual tree, 2026-07-14.**

Behavioural rules live in `.claude/rules/` and load contextually. This file is the map and
the hard constraints. The harness itself is documented in `.claude/README.md`.

## What this is

**M7 — Modality-Adaptive SNN with Embedding Replay** (NeurIPS 2026). One spiking neural
network learns three neuromorphic datasets *sequentially* — N-MNIST (visual) → SHD (audio)
→ DVS-Gesture (visual-motion) — without catastrophic forgetting, using per-task heads plus
replay of 512-d embeddings instead of raw events. A plan exists to port it to SpiNNaker.

### The team

Blessing Effiong (SLU) — lead: architecture, writing, coordination · Junaid Rehman —
DVS-Gesture loader + T3 · Somshubhra Roy (NC State) — EWC + joint-training baselines ·
Minju Kim (Indiana) — gate analysis + figures · Isaac Nkrumah (SLU) — SynOps energy +
proofreading.

### The three-paper arc

| Paper | Venue | Status | Core claim |
|---|---|---|---|
| 1 | NICE 2026 | Accepted | Memory-augmented SNNs; supervised-contrastive engrams |
| 2 | IEEE Computer | Minor revisions | **Hopfield helps visual, hurts audio** — the modality asymmetry that motivates everything else |
| 3 | **NeurIPS 2026** | **Current** | **M7** — continual learning across three heterogeneous neuromorphic domains |

## Repository map — what is actually here

| Path | Contents |
|---|---|
| `code/` | All executable code. ~42 files, mostly large standalone scripts. `code/archive/` holds superseded ones. |
| `docs/` | 8 flat files. `EXPERIMENTAL_MANIFEST.md` is the repro spec; also `M7_Architecture_Deep_Dive.md`, `SpiNNaker_Port_Plan.md`, `HARDWARE_ACCESS_CHECKLIST.md`, `HOME.md`, `paper_sections_correspondence.md`, `README.md`, `M7_Team_Onboarding.docx`. |
| `Models/`, `Dataloaders/`, `training/`, `utils/` | The **legacy bi-modal** package (Paper 2). Imported by `tests/`. **Not used by `code/m7_trimodal_run.py`.** |
| `tests/` | **6 pytest suites.** Tonic is mocked → fast, safe, needs no datasets. |
| `paper/` | `cross_modal_correspondence.tex` — the *controlled-negative-result* paper. |
| `reports/` | sPyNNaker auto-generated provenance SQLite. Not research output. |

### What does NOT exist — do not re-hallucinate these

`configs/` · `results/` · `figures/` · `notebook/` · `experiments/` · `ICONS_M7/` ·
`code/.venv/` · **any `.ipynb`** · `code/test_spinnaker_*.py` · any `.bib` file ·
**any results CSV** · **any `.pt` checkpoint**.

An earlier version of this file asserted most of these existed. They do not. The M7 notebook
is gone — `code/modify_notebook.py` is the only surviving record of its cell structure, and
its target file is missing, so "Cell 11" and "Cell 30" refer to nothing.

The `SessionStart` hook (`.claude/hooks/session_state.py`) prints the live state at the top of
every session. **When this file and that hook disagree, the hook is right.**

## Commands

There is **no venv**. `python3.11` (3.11.12) is present and is the correct target — the system
default is 3.14, which will not satisfy the pins.

```bash
# One-time setup — or just use the /bootstrap-env skill
python3.11 -m venv code/.venv && source code/.venv/bin/activate
pip install -r code/requirements.txt

# The ONLY fast, safe validation loop. Use this by default.
pytest tests/

# Full tri-modal CL run — 4-5 hrs on MPS. Needs ~23 GB of datasets that are NOT here.
python code/m7_trimodal_run.py

# Figures — requires a results CSV, which does not exist yet
python code/generate_figures_3seed.py

# Automation assistants (see docs/AUTOMATION_TUTORIAL.md)
python code/automation/magazine.py
python code/automation/repo_watch.py
python code/automation/consult.py
python code/automation/improve_paper.py
python code/automation/improve_repo.py
python code/automation/lesson.py "what the assistants should remember"
```

⚠️ `pytest tests/` exercises the **legacy bi-modal** package, not M7. Passing tests do **not**
validate `code/m7_trimodal_run.py`.

## Critical run rules

- ⚠️ **`code/m7_trimodal_run.py` has no `if __name__ == "__main__"` guard and loads datasets at
  import time.** `python -c "import m7_trimodal_run"` does not check the imports — **it starts
  the full 4–5 hour run.** To inspect the module, read the file.
- ⚠️ **Training is silently skipped when a checkpoint exists.** `run_one_seed()` finds a
  matching `final_*.pt`, loads it, and skips training with no warning. Stale checkpoints
  therefore produce stale results that look fresh.
- ⚠️ **Seed 42 used to be silently skipped.** `SEEDS` was `[123, 456]`, so every "3-seed
  mean ± std" was really a 2-seed number. **Fixed 2026-07-14** → `[42, 123, 456]`. But **any
  result generated before that date is a 2-seed result**, however it is labelled.
- The results CSV holds **two schemas** (incremental rows have a `step`; summary rows don't)
  and reruns append duplicates. **Never read it by hand** — use `/analyze-cl-results`. Note the
  documented `drop_duplicates()` ritual is *exact-match*, so near-duplicate rerun rows survive
  and silently inflate `n`.
- Device auto-selects MPS → CUDA → CPU. On macOS **`num_workers=0` is required** (tonic
  multiprocessing). This is not a performance knob.
- Scripts use `BASE_DIR = './ICONS_M7'`, which only resolves as documented when the CWD is
  `code/`.

## Architecture

Event input (T1→T2→T3, sequential) → **separate** per-modality SNN backbones (Conv for visual,
MLP for audio; all emit 512-d) → parallel memory: `ModernHopfieldLayer` ∥ `ImprovedHGRNGate` →
adaptive gate (512→128→2 softmax → `w_hop`, `w_hgrn`) → **per-task heads** (frozen when
inactive).

Loss: `L = CE(x_t) + 0.1·SCL(x_t) + 0.3·CE(replay batch)`

| Task | Dataset | Modality | Classes | Input shape |
|---|---|---|---|---|
| T1 | N-MNIST | Visual | 10 | `[B,25,2,34,34]` |
| T2 | SHD | Audio | 20 | `[B,100,1,1,700]` |
| T3 | DVS-Gesture | Visual-motion | 11 | `[B,T,2,128,128]` → pooled to 34×34 |

Per-task heads isolate gradients; embedding replay prevents forgetting. The gate only learns to
use HGRN *under replay* — without it, `w_hgrn → 0`.

Fixed: LIF **β = 0.9** (shared across every neuron and modality) · `time_steps = 25` · 512-d
embeddings · `replay_weight = 0.3` · SCL τ=0.07, λ=0.1 · val split 10 % with `manual_seed(42)`
(splits are fixed across seeds; only init and shuffling vary).

**Backbone (resolved 2026-07-14 from source):** `conv1(2→32, k5) → MaxPool → conv2(32→64, k5)
→ MaxPool → fc1(1600→1024) → fc2(1024→512)`. **Two** conv layers, not three.
`EXPERIMENTAL_MANIFEST.md` claimed three and was wrong; the deep-dive and the SpiNNaker plan
were right. This matters: `fc1 = 1600 × 1024 = 1,638,400` weights is exactly what the ~205-core
SpiNNaker budget rests on.

**Hopfield patterns:** M7 uses **100** (`CONFIG['num_patterns']`). The class *signature*
defaults to 256, but the call site overrides it — **read the call site.** The legacy
`Models/hopfield_layer.py` genuinely uses 256; that is a different codebase, not a
contradiction. See `.claude/rules/contested-facts.md`.

## Research resources

The project links to a sibling Obsidian vault at `resources/research/` (symlinked from
`/Users/agent1/Documents/neuromorphic/research`). The local RAG indexes literature notes,
concept notes, project notes, and meeting logs so Claude can ground suggestions in the
actual research corpus rather than hallucinate citations.

### Claude skills
Anthropic-style skills live in `.claude/skills/`:
- **research-dream** — grounded ideation, architecture brainstorming, experiment proposals.
- **research-steer** — convention-aware review of plans, PRDs, and diffs.
- **research-analyze-pdf** — analyze a PDF and write a structured literature note to the vault.

Both skills query the RAG via `python code/rag_query.py ... --context`. When a skill is
invoked, ensure the index is fresh (`python code/rag_ingest.py`) before relying on results.

### Automation assistants (`docs/AUTOMATION_TUTORIAL.md`)
A second automation layer is registered as Claude Code slash commands in `.claude/commands/`.
All reports land in the `knowledge/` vault.

| Command | Script | Output |
|---|---|---|
| `/research-magazine` | `python code/automation/magazine.py` | `knowledge/10-Research-Magazine/YYYY-MM-DD.md` |
| `/repo-watch` | `python code/automation/repo_watch.py` | `knowledge/20-Repo-Watch/YYYY-MM-DD.md` |
| `/consult` | `python code/automation/consult.py` | `knowledge/30-Consultant/YYYY-MM-DD_HH-MM-SS.md` |
| `/improve-paper` | `python code/automation/improve_paper.py` | `knowledge/40-Paper/YYYY-MM-DD_HH-MM-SS.md` |
| `/improve-repo` | `python code/automation/improve_repo.py` | `knowledge/60-Patches/YYYY-MM-DD_HH-MM-SS.md` |
| `/lesson "..."` | `python code/automation/lesson.py "..."` | `knowledge/01-Lessons/YYYY-MM-DD_HH-MM-SS.md` |

Daily cron setup: `bash scripts/cron/install-cron.sh` (launchd setup: `bash scripts/cron/install.sh`).

### Ingestion rules
- Literature notes use YAML frontmatter; the parser extracts `title`, `authors`, `year`,
  and `tags` for richer metadata.
- Ingestion is incremental: only changed or new files are re-embedded.
- If `sentence-transformers` is not installed, the system falls back to a deterministic
  hash embedder so the harness still works for testing; install the package for real
  semantic search.

## Known issues

## The five critical rules — non-negotiable

1. **Never put a number anywhere that is not in a CSV.** No estimates. Ever.
2. **Back up to HuggingFace after every seed.** Results and checkpoints are gitignored; git
   will not save you. This team has already lost data once.
3. **Anonymous submission** — self-cite as `\citep{anon2026modality}`.
4. **Report test accuracy, not validation.** SHD's val/test gap is 15–19 pp.
5. **The forgetting claim is ~0 pp — not "100 % reduction."** The honest claim is that replay
   enables gate specialisation.

Also: **hypotheses are not results** (hardware-similarity claims especially) · accuracy always
travels with **spike rate / sparsity and timestep count** · never refactor in a way that
silently changes numerics · never overwrite a file without showing a diff first.

> **Right now there is no results CSV in this repo.** Under RULE 1 that makes every number
> currently quoted in `docs/` and `paper/` unsourced. Treat them as claims to verify, not facts
> to repeat. Use `/verify-claim` or the `results-auditor` agent.

**Boundary rule:** operate only inside this repo. Never write to sibling repos
(`../spinnaker-mapping/`) or the personal vault (`../research/`).

## The two narratives — keep them straight

- `docs/M7_Team_Onboarding.docx` and `docs/HOME.md` present **forward transfer as a positive
  result** (FWT ≈ 75 pp): "visual spikes scaffold auditory learning."
- `paper/cross_modal_correspondence.tex` and `docs/paper_sections_correspondence.md` conclude
  that cross-modal *correspondence* is **engineered, not emergent** — held-out retrieval at
  chance, Procrustes 0 %, Mantel r = 0.051. *"A supervised bridge, not a discovered shared
  manifold."*

They measure different things, so both can hold. But **never let a summary imply an emergent
modality-invariant semantic space** — the team's own data refutes that.

## SpiNNaker port

Phase 1 inference (±1 % of baseline) → Phase 2 host-mediated CL (±2 pp) → Phase 3 on-chip STDP
(stretch). **SpiNNaker runs PyNN, not PyTorch** — no backprop, no autograd; components are
reimplemented, not exported, and parameters do **not** transfer automatically.

Full M7 ≈ **600 cores** → needs SpiNN-5 (~864); SpiNN-3 (72) is ~8× short. `LayerNorm` and
`Softmax` are not native, and HGRN's multiplicative reset `r * h_prev` is **not expressible** —
that is risk R3, and it threatens the paper's central novelty claim.

**Nothing has run on a board.** `docs/HARDWARE_ACCESS_CHECKLIST.md` is entirely unchecked.
Mapping ≠ inference ≠ real hardware. See `.claude/rules/spinnaker.md`.

## The harness

`.claude/` — documented in `.claude/README.md`.

- **Skills:** `/bootstrap-env` · `/check-repo-state` · `/analyze-cl-results` ·
  `/run-m7-experiment` · `/verify-claim` · `/spinnaker-port-check`
- **Agents:** `results-auditor` · `paper-fact-checker` · `snn-code-reviewer` ·
  `spinnaker-port-engineer` · `literature-scout`
- **Workflows:** `results-audit` · `doc-code-reconcile` · `spinnaker-feasibility-audit`
- **Enforced:** you cannot hand-edit a results CSV or a `.pt` checkpoint, and you cannot write
  outside this repo.

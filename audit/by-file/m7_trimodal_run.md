# `code/m7_trimodal_run.py` — deep-dive audit

- **Cluster:** `A-core-entrypoint` · **Shard:** `w1_run` · **Prefix:** `RUN` · **Status:** live
- **Size:** 719 lines, self-contained, imported by nobody.
- **Role:** the reproducibility entry point for M7 tri-modal continual learning (N-MNIST → SHD → DVS-Gesture).
- **Findings:** 35 (`RUN-001`…`RUN-035`) — 3 critical, 10 high, 10 medium, 11 low, 1 info.
- **Sibling clones:** many of the high/critical defects are duplicated verbatim in `code/m7_trimodal_run_provenance.py` (the import-safe copy). Clone line ranges are listed in the JSONL `fix_locations`; fixing agents must patch both files.

> Analysis was static (read-only). This file was **never imported or executed** — importing it *is* the 4–5 h run (`RUN-001`).

---

## The headline: the advertised mechanism barely runs

The paper's story is "embedding replay across three heterogeneous neuromorphic domains, mediated by an adaptive Hopfield∥HGRN gate." Reading the code end-to-end, **each of the three load-bearing pieces of that story is broken or inert**:

1. **Replay is misrouted and mostly destroyed.** The replay buffer is keyed by *class label* but sampled by *task id* (`RUN-002`), and every task is populated with `label_offset=0` so SHD's classes overwrite N-MNIST's in the store (`RUN-003`). Concretely: T2 "replay of all of T1" actually samples only the single class whose integer label is `1`; T3 "replay of T1+T2" samples only classes `1` and `2`, both of which are **SHD** by then (N-MNIST embeddings were overwritten), so **N-MNIST gets zero rehearsal during T3**. And even that trickle is at most 1–2 embeddings per batch, not the requested 32. Whatever replay does for the results, it is not what the paper describes.

2. **The HGRN gate's multiplicative reset — the headline novelty and the central SpiNNaker risk (R3) — does nothing** (`RUN-012`). The gate is only ever called without a hidden state, so `h_prev ≡ 0`, `r * h_prev ≡ 0`, and the reset gate `r` (with `W_r`, `norm_r`) has no effect on the output and receives no gradient. The op that "cannot be expressed on SpiNNaker" is also a no-op in PyTorch.

3. **The "forward transfer ≈ 75 pp" number is mislabeled.** `fwt = acc_t2_after − acc_t2_zero_shot` is a *learning gain* against a random-init head (≈ chance), not forward transfer (`RUN-005`). The provenance sibling actually keeps `learning_gain` and `fwt` as separate quantities, confirming the mislabel.

Layered on top: **near-zero forgetting is structural, not earned** (`RUN-013`). Backbones and heads are per-task and grad-isolated, so only the three shared modules (Hopfield / HGRN / adaptive gate) can drift — forgetting is bounded by architecture regardless of replay. Per RULE 5 this must be reported as "replay enables gate specialisation," never as "eliminates catastrophic forgetting."

And because this is an SNN paper, note that **no accuracy anywhere travels with its spike-rate/sparsity/timestep cost** (`RUN-011`) — a direct RULE violation baked into `evaluate()` and the results dict.

---

## Critical

- **`RUN-001` — no `__main__` guard; the full 6-run loop, dataset download, and dir creation all fire at import.** `for seed in SEEDS: for condition in CONDITIONS: run_one_seed(...)` runs at module scope (699-711); datasets materialize at 109-180; dirs are `mkdir`'d at 41-42. Any import/tool/test-collection starts a 4–5 h run. Fix: wrap in `main()` behind `if __name__ == "__main__"`, as the provenance sibling already does.
- **`RUN-002` — replay `sample()` filters TASK ids against a CLASS-keyed store, and draws ≤1 embedding/class.** `available = [c for c in task_ids if c in self._store]` (383) confuses task numbers with class labels; one `np.random.randint` per class (388-391) caps a "batch" at #available classes. `_task_labels` — the map that would fix this — is never read.
- **`RUN-003` — `populate(..., label_offset=0)` for every task overwrites prior classes.** `c = int(lbl) + label_offset` (369) with offset 0 at both call sites (553, 584) means SHD 0-9 clobber N-MNIST 0-9. Compounds `RUN-002`: T3 replay contains no N-MNIST at all.

## High

- **`RUN-004` — replay routed through the *current* task head** (`forward_from_embeddings(replay_embs, task_name)`, 453). Old-task embeddings are trained against old labels through the new head — does nothing to preserve the old head.
- **`RUN-005` — `fwt` is learning gain, mislabeled as forward transfer** (559, 616). Zero-shot baseline uses a random-init SHD head (≈ chance), so `fwt` ≈ SHD-test-minus-chance.
- **`RUN-006` — `forgetting_T1_T3` is the incremental T2→T3 delta**, not total T1 forgetting after T3 (614 vs 628); inconsistent with its start-to-end siblings (613, 615).
- **`RUN-007` — reported `test_acc` uses last-epoch weights; `best_*.pt` is saved but never reloaded** (485-504). Early stopping selects a "best" that nothing downstream uses; `final_*.pt` is also last-epoch.
- **`RUN-008` — `best_*.pt` filename omits seed+condition** (488), so all 6 runs clobber the same 3 files (`final_*.pt` is namespaced correctly). Also violates the back-up-every-seed rule.
- **`RUN-009` — silent checkpoint-skip loads stale `final_*.pt` with no provenance check** (537-544, 563-572, 590-599), and skips `_save_gate_history`, so `gate_trajectory.csv` (the sole source of the core gate-specialisation figure) silently under-populates on any reuse run.
- **`RUN-010` — results CSV mixes two schemas and appends without dedup.** Summary rows (no `step`, 618-640) and incremental rows (with `step`, 687-697) share a file; read-concat-overwrite (no `drop_duplicates`) means reruns inflate `n`. `gate_trajectory.csv` (650-684) has the same pattern. `step`-presence is the only schema discriminator — the mandated `df[df.step.isna()].drop_duplicates()` ritual is load-bearing.
- **`RUN-011` — accuracy reported with no spike-rate/sparsity/timestep cost** (evaluate 411-419; result dict 618-631). RULE: accuracy never travels alone.
- **`RUN-012` — HGRN reset gate is inert** (`h_prev ≡ 0` ⇒ `r*h_prev ≡ 0`; 226, 230-231; call sites 340, 347). Dead parameters; undercuts the novelty and the SpiNNaker-R3 framing.
- **`RUN-013` — near-zero forgetting is structural** (per-task grad-isolated backbones/heads; only Hopfield/HGRN/gate are shared, 273-300, 421-458). Report as gate specialisation, not as preventing forgetting.

## Medium

- **`RUN-014` — SHD runs 100 timesteps while N-MNIST/DVS run 25**; `CONFIG['time_steps']=25` is never read (139 vs 77-80/94-97).
- **`RUN-015` — collate truncates every sample to the first 25 frames** (78, 95), discarding most of each recording given `time_window=1000/50000 µs` (117, 165). Magic 25, not from config.
- **`RUN-016` — SupervisedContrastiveLoss keeps the self/diagonal term** in both mask and denominator (193, 196-198); deviates from canonical SupCon and biases the loss. (Fixing it moves numbers — flag as deliberate.)
- **`RUN-017` — entire SHD set densified into a Python list at import** (149-150), with a per-event Python loop (145-147). Memory-heavy, slow, import-time.
- **`RUN-018` — `fc1 = Linear(64*5*5, 1024)` hardcodes a 34×34 input** (242), forcing DVS to be avg-pooled 128×128→34×34 in collate (98-102). This 1600×1024 fan-in is also the basis of the SpiNNaker core budget.
- **`RUN-019` — determinism only partially configured**: torch+numpy global seeds set (523-524), but not `use_deterministic_algorithms`, MPS determinism, or per-loader shuffle generators (module-level loaders, 127/155/173). (`random` is unused, so that gap is moot.)
- **`RUN-020` — real schedule is hardcoded 15 epochs / patience 5 / lr 1e-3 / wd 1e-4** (422-424 defaults + 542/567-570/594-597 call sites), overriding and contradicting `CONFIG['max_epochs']=50`/`patience=12`/`weight_decay`.
- **`RUN-021` — 14 of 19 CONFIG keys are never read** (45-65): `learning_rate, weight_decay, max_epochs, patience, dropout, hidden_dim, num_gru_layers, num_workers, time_steps, pin_memory, use_contrastive, contrastive_weight, seed, device`. Looks tunable; mostly isn't.
- **`RUN-022` — dead classifier layers**: visual `fc3/lif5` compute `spk5` that both callers discard (`_, features = bb(x)`, 305/324); audio `fc4/lif4` (288-289) are never called. Wasted params/compute.
- **`RUN-023` — `_get_features` and `forward` duplicate the whole feature-extraction** (302-319 vs 321-342), including the audio unroll verbatim — a drift hazard between the replay path and the training path.

## Low / info

- **`RUN-024`** — `safe_empty_cache` (20-22) never called; CUDA-only despite the "MPS/CPU compatibility" banner.
- **`RUN-025`** — unused imports `os, sys, time` (4-6).
- **`RUN-026`** — dead local import `from torch.nn import Linear, LayerNorm` (280).
- **`RUN-027`** — `import pandas as pd` at line 648 (mid-file), after the functions that reference it; works only because calls come later.
- **`RUN-028`** — Hopfield ctor default `num_patterns=256` is dead (call site passes 100, 293); redundant `temperature`(=1.0) + `/√dim` scaling (206, 209).
- **`RUN-029`** — model `.to(device)` twice (402, 529); device passed as `str` not `torch.device` (530-531).
- **`RUN-030`** — `_task_labels` populated (378) but never read — the exact map needed to fix `RUN-002`/`RUN-004`.
- **`RUN-031`** — stale comment (502) claims the caller renames the checkpoint; no rename exists.
- **`RUN-032`** — `PLOT_DIR` created at import (40-42) but never used.
- **`RUN-033`** — `events_to_dense` (138-148): binary occupancy (no count accumulation), redundant `max_time` guard, and divide-by-zero if all timestamps equal (`max_time==0`).
- **`RUN-034`** *(info)* — train/val split generator hardcoded `manual_seed(42)` independent of the run seed (124-126, 152-154, 169-172). Documented intent; recorded so it isn't "fixed" into a bug.
- **`RUN-035`** — derived metrics (forgetting/fwt) persist only in the end-of-seed summary row (612-640); only raw `T*_after_T*` accuracies are saved incrementally (549/577/604), so an interrupted seed loses them.

---

## Cross-cutting themes

- **Replay subsystem is the top priority.** `RUN-002`, `RUN-003`, `RUN-004`, `RUN-030` are one coupled defect: a class-keyed store that is sampled and routed by task, with no offset. Fix them together (key by `(task_id, class)`, consume `_task_labels`, route per-head) — and mirror into the provenance sibling (446-458 / 427-444 / 409-414 / 1011 / 1079 / 614).
- **Research-integrity surface (RULE 1–5).** `RUN-005`, `RUN-006`, `RUN-011`, `RUN-013` govern how every number this script emits may be described. None are safe to repeat as-is; the labels overstate transfer, mislabel forgetting, and omit spike cost. There is no results CSV in the repo, so all downstream numbers are unsourced regardless.
- **Config is decorative.** `RUN-014`, `RUN-020`, `RUN-021` — the real hyperparameters are hardcoded literals scattered across collate/optimizer/call-sites; CONFIG mostly lies. Single-source it before anyone trusts a documented setting.
- **Import-time + checkpoint-skip are the two operational traps.** `RUN-001` (importing = running) and `RUN-009` (stale checkpoint = fresh-looking stale results) are exactly the hazards CLAUDE.md and `.claude/rules/python-numerics.md` warn about; both are still live here.

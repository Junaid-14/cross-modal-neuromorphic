# Audit — `code/m7_trimodal_run_provenance.py`

- **Prefix:** `PROV` · **Shard:** `w2_prov` · **Cluster:** `B-shared-library` · **Status:** live
- **Size:** 1379 lines. **Role:** the REAL shared hub — referenced by **19** files (`grep` count, not 17): every `code/run_*.py`, `code/eval_transfer_fixed.py`, `code/phase1_sanity_check.py`, `code/test_*` , `tests/test_cross_modal_transfer.py`, two `code/archive/*` scripts, and `.claude/hooks/guard_long_runs.py`. Bugs here propagate to all of them, so severity is weighted up.
- **Findings:** 27 (1 critical, 7 high, 11 medium, 6 low, 2 info).
- **Method:** static read only; the file was never imported or executed. `code/m7_trimodal_run.py` (the orphan copy) was read as text for divergence comparison.

## One-paragraph orientation

This file is a superset re-implementation of the tri-modal CL experiment: `@dataclass ExperimentConfig` (33-74), the model stack (`SNN_Backbone`, `M7_ContinualAdaptiveModel`, `ModernHopfieldLayer`, `ImprovedHGRNGate`), two replay buffers, a `ContinualLearner` with task-freezing and a cross-modal alignment phase, a random-init FWT baseline, a results aggregator with Wilcoxon tests, and `main()` behind a proper `__main__` guard (1378). `run_one_seed(cfg, data_loaders, results_dir, ckpt_dir)` (962) takes **injected** loaders — there is no import-time global dataset load, so importing the module is safe (PROV-026), which is the single most important property given 19 dependents. The good news stops there: the embedding-replay mechanism is broken end-to-end (PROV-001), the default forward pass silently runs a different model than both the docs and the orphan (PROV-002/003), the "best" model is never restored (PROV-004), one forgetting metric is defined inconsistently (PROV-005), the results CSV is aggregated without dedup and in a schema that diverges from the orphan and the canonical loader (PROV-006/007/008), and a large tail of config fields is dead or misleading.

---

## CRITICAL

### PROV-001 — Embedding replay is fundamentally broken (correctness)
The pre-verified defect, formalized and expanded. Three coupled failures make `EmbeddingReplayBuffer` unable to rehearse prior tasks:

1. **`sample()` treats `task_ids` as store keys, not task ids.** `available = [c for c in task_ids if c in self._store]` (449). Call sites pass `replay_task_ids=[1]` (1050) and `[1, 2]` (1102), so the buffer returns only the embeddings stored under integer keys `1` (and `2`) — i.e. class **indices** 1 and 2 — never "task 1" / "task 2".
2. **`label_offset=0` at every populate call (1011, 1079) collides tasks.** `populate` keys the store by `c = int(lbl.item()) + label_offset` (435) and writes `self._store[c] = torch.stack(samples)` (443). With offset 0, SHD's classes 0-9 **overwrite** N-MNIST's classes 0-9. After T2, N-MNIST embeddings for digits 0-9 are gone; keys 0-19 are all SHD.
3. **Replayed samples go through the wrong head with class-index labels.** The sampled labels are the store keys (457), and they are fed to `forward_from_embeddings(replay_embs, task_name)` (612-614) using the **current** task's head. So during T3, replay pushes a couple of SHD-derived embeddings, labeled `1`/`2`, through the **DVS** head. `_task_labels[c] = task_id` (444) records the correct task but `sample()` never reads it.

Net effect: "embedding replay" rehearses at most one or two wrong-task embeddings through the current head each step. It cannot prevent forgetting, which directly undermines the paper's central claim that replay enables gate specialisation.

**The correct pattern is present in the same file:** `RawInputReplayBuffer.sample` (498-521) filters by `task_names` against a task-keyed store and returns per-task batches. The embedding buffer simply ignores that pattern.

**Fix locations** (this file **and** the clone): `m7_trimodal_run_provenance.py:435,444,446-458,611-615,1011,1050,1079,1102` and `m7_trimodal_run.py:380-392,553,569,584,596`.

---

## HIGH

### PROV-002 — The default forward pass secretly reparametrises features (architecture / numerics / divergence)
`forward` (401) → `forward_with_shared` (372) → `_split_shared_private` (394) on **every** call. Features are projected `proj_shared` (512→128) and `proj_private` (512→384), concatenated back to `full` (369), and it is `full`, not the raw backbone features, that feeds the gate/Hopfield/HGRN (395-398). Those two projections are independent trainable `nn.Linear` layers, so `full = W·features + b` is an extra learned 512→512 layer inserted before the memory stack. The docstring says this path is "used during cross-modal alignment training" (373-374), but it runs even with `align_cross_modal=False` (the default, 1321). The orphan feeds **raw** features to the same modules (`m7_trimodal_run.py:338-340`), and CLAUDE.md documents no shared/private split. The shared hub therefore runs a higher-capacity, differently-parametrised model than the docs and the nominal entry point — a silent numeric divergence.

### PROV-003 — Replay path skips the projection that the live path applies (correctness)
Consequence of PROV-002. `forward_from_embeddings` (409-414) applies gate/Hopfield/HGRN **directly** to the stored embeddings, with no `_split_shared_private`. But stored embeddings are raw backbone features (`populate` → `_get_features`, 433/359). So replay trains the memory stack on the raw-feature distribution while inference uses the projected `full` distribution — a train/inference mismatch that corrupts the replay gradient independently of PROV-001. In the orphan both paths use raw features, so they are consistent there; this inconsistency is provenance-specific.

### PROV-004 — Best-validation weights are never restored (research-integrity)
`train_task` tracks `best_val` as a scalar (662-663) but never snapshots the weights. Early stopping breaks after `patience` non-improving epochs (665-669), leaving the model in a degraded post-peak state. `test_acc` is computed on that final state (671), and the checkpoint saved by `run_one_seed` is the final-epoch `model.state_dict()` (1003/1054/1106). So the reported `best_val`, the `test_acc`, and the persisted/reloaded weights are three different things. The orphan at least writes a `best_*.pt` (`m7_trimodal_run.py:488-495`) — but also never reloads it before test — whereas provenance dropped even that save.

### PROV-005 — `forgetting_T1_T3` uses an inconsistent baseline (research-integrity)
`forgetting_T1_T2 = acc_t1_after − acc_t1_after_t2` (1133) and `forgetting_T2_T3 = acc_t2_after − acc_t2_after_t3` (1135) both measure drop from a task's own post-training accuracy. But `forgetting_T1_T3 = acc_t1_after_t2 − acc_t1_after_t3` (1134) measures the **incremental** T2→T3 drop, not cumulative T1 forgetting after T3 (which would be `acc_t1_after − acc_t1_after_t3`). The three metrics do not share a definition, so any total-forgetting summary is wrong. Identical in the orphan (`m7_trimodal_run.py:613-615`). Forgetting is RULE-5-sensitive.

### PROV-006 — CSV appended and aggregated without dedup; schema diverges (research-integrity)
`save_seed_result` appends each row via read+concat+write (853-856), no dedup, so reruns accumulate duplicates. `aggregate_results` reads the master (883), keeps only `df["T1_after_T1"].notna()` (885), and computes mean/std/CI/Wilcoxon over the **raw** rows (888-936) — duplicates inflate `n` and corrupt the spread, exactly the hazard the repo's own `load_results.py` warns about. Schema also diverges: this file writes summary rows with **no `step` column** and columns `T2_fwt`/`T3_fwt` (1154-1155), while the orphan writes `step`-bearing incremental rows plus a single `fwt` (`m7_trimodal_run.py:687-697,630`). The documented ritual `df[df["step"].isna()]` KeyErrors on this file's CSV (no `step` column), and the canonical loader reads a `fwt` column (`load_results.py:206`) this file never writes. Both hubs write the **same** filename, so mixing runs yields inconsistent columns and the `notna` filter would even retain the orphan's `step` rows.

### PROV-007 — Two divergent copies of the model+trainer (duplication)
The hub and the orphan share the same public names (`M7_ContinualAdaptiveModel`, `ContinualLearner`, `EmbeddingReplayBuffer`, `SNN_Backbone`, `run_one_seed`, `TASKS`, collate fns) but have diverged in: the forward math (PROV-002), task freezing (`_freeze_completed_tasks` 550-568 exists only in provenance; the orphan trains all params every task), the `fwt` definition (PROV-008), the CSV schema (PROV-006), and import-safety (guard vs top-level run). They also write **identical** output filenames (`cl_trimodal_results.csv`, `final_{task}_{task}_seed{seed}_{condition}.pt`), so their artifacts are mutually confusable while their numbers differ. Edits to one will not reach the other.

| Aspect | provenance hub | orphan `m7_trimodal_run.py` |
|---|---|---|
| Forward into memory stack | `full` after proj_shared/private (394-398) | raw `features` (338-340) |
| Task freezing | yes, `_freeze_completed_tasks` (550) | none |
| `fwt` | `zero_shot − random_init` (1066) | `after − zero_shot` (616) = learning gain |
| Replay/live consistency | inconsistent (PROV-003) | consistent |
| CSV | summary-only, no `step`, `T2_fwt`/`T3_fwt` | dual-schema `step` rows + `fwt` |
| Config | `ExperimentConfig` dataclass | `CONFIG` dict globals |
| Seeds default | `[42,123,456,789,101112]` (5) | `[42,123,456]` (3) |
| Import-time | safe (`__main__` guard) | runs 4-5h at import |
| Extra machinery | alignment, raw replay, random-init baseline, aggregator | none |

### PROV-008 — `fwt` is a learning-gain mislabel in the orphan; only that mislabel yields a large positive FWT (research-integrity)
This file's FWT (`zero_shot − random_init`, 1066) is the honest one: both terms use an **untrained** T2 head, so FWT ≈ 0 by construction, and `print_task_sanity` warns if zero-shot strays >5 pp from chance (840). The orphan's `fwt = after − zero_shot` (`m7_trimodal_run.py:616`) is the **training gain**, stored under the name `fwt`. The docs' unsourced "FWT ≈ 75 pp" positive-transfer number can only come from the orphan's mislabeled formula, not from a real forward-transfer measurement — directly relevant to contested-fact #4. Do not repeat the 75 pp claim without a CSV (RULE 1).

---

## MEDIUM

- **PROV-009 — "95% CI" uses `1.96` for tiny n (896).** Correct multiplier is Student-t (`t.975`= 4.30 at n=3, 2.78 at n=5); the interval is too narrow and mislabeled. `n==1` emits `ci95=0.0` → a fake zero-width CI; `std(ddof=1)` is NaN at n=1. Fix with `scipy.stats.t.ppf` (scipy already imported, 23).
- **PROV-010 — Wilcoxon on 3-5 pairs (920).** Underpowered — at n=3 the smallest two-sided p is ~0.25, so significance is unreachable; reporting the p-value misleads. Dead `seeds = df["seed"].unique()` (911).
- **PROV-011 — `random_init_baseline` dirties the global RNG (812, called 982-984).** It reseeds `torch.manual_seed(cfg.seed+1000+s)` and leaves the global stream altered before T1 training (997), so training shuffling depends on the baseline procedure; changing `n_seeds` (hardcoded 3 at 983) silently moves results. Isolate with a local `Generator` or save/restore RNG state, then reseed to `cfg.seed`.
- **PROV-012 — Default is 5 seeds (63, 1320)**, deviating from the documented `[42,123,456]` policy and the orphan; the canonical loader's seed gate is calibrated to three.
- **PROV-013 — The "per-seed" CSV holds the whole master (858-860).** `df` was reassigned to the concatenation at 855, then written again to `cl_results_seed{seed}_{condition}.csv` (860); the file labeled per-seed accumulates every seed. Write a fresh single-row frame.
- **PROV-014 — SupCon counts the anchor as its own positive (159/164; 192/198/202).** Deviates from Khosla et al.; `SupervisedContrastiveLoss` is active in the default run (609). `CrossModalSupConLoss`'s comment (201) — "self gives 0 contribution … log(1)=0" — is mathematically wrong (the diagonal is a log-softmax term ≤ 0, not 0). Mask the diagonal out of the positive set; fix/delete the comment. Numerics-affecting.
- **PROV-015 — Dead / misleading config fields (33-74).** Zero `cfg.`-reads for `optimizer, max_epochs, patience, dropout, hidden_dim, num_gru_layers, num_workers, pin_memory, time_steps`; `align_joint_training, align_mixed_batches, nmnist_train_per_class` referenced nowhere. Actively misleading: `max_epochs=50`/`patience=12` are ignored (train_task called with `num_epochs=15, patience=5` at 999/1048/1100); `dropout=0.2` but the model has no dropout layer; `optimizer` hardcoded AdamW (580).
- **PROV-016 — Unconditional legacy import + always-built paired loader (1251, 1262-1271).** `from Dataloaders.paired_digit_loader import get_paired_digit_loaders` runs and the paired loader is materialised even when `align_cross_modal=False` (only used at 1017). The hub has no `sys.path` insert, so it relies on each of the 19 callers having set it. Guard both behind the flag and/or vendor the loader into `code/`.
- **PROV-017 — SHD fully densified into RAM (1212-1213).** Every SHD sample becomes a dense `(100,1,1,700)` tensor in a Python list before training — gigabytes resident. Use a lazy Dataset / disk cache.
- **PROV-018 — Timestep count inconsistent.** `cfg.time_steps=25` is dead (PROV-015); N-MNIST/DVS are cropped to 25 frames (113-116/133-140) but SHD runs at 100 bins (`_events_to_dense` time_bins=100, 1201; `shd_collate_fn` does no crop, 121-123). Backbones loop `T=x.size(1)` (275/353), so audio runs 4× deeper than visual. Complicates the mandatory "accuracy travels with its timestep count" reporting.
- **PROV-022 — CWD-relative paths (1169, 1285).** `./ICONS_M7` resolves as documented only from `code/`; a hub used by 19 callers from varying CWDs will read/write in the wrong place, including loading a stale `final_*.pt` (the checkpoint-skip trap). `run_one_seed` already takes explicit dirs — `main`/`build_data_loaders` should too.

---

## LOW

- **PROV-019 — Unused imports (7-14):** `os, sys, time, copy`, and typing `Optional, Tuple, Dict`. (`copy` would be needed if PROV-004 is fixed with `copy.deepcopy`.)
- **PROV-020 — `safe_empty_cache` (89-91) never called.**
- **PROV-021 — `EmbeddingReplayBuffer` default device is mps/cpu only (420)**, ignoring CUDA; a CUDA-host caller omitting `device` would hit a device mismatch in `forward_from_embeddings`. (`run_one_seed` passes device explicitly at 975.)
- **PROV-023 — N-MNIST/DVS collate crops to the first 25 frames (113-114/133-135)**, silently dropping most of a ~300-window recording; surface this where timestep cost is reported.
- **PROV-024 — `_events_to_dense` divides by `max_time` which can be 0 (1204-1205)** → NaN indices when all events share `t=0`; the `len>0` guard doesn't cover it. Use `max(events['t'].max(), 1)`.
- **PROV-025 — Redundant duplicate DataLoaders per split (1186-1248):** a `train` loader and a `train_val_test[0]` loader over the same subset, iterating in different (unseeded shuffle) orders.

---

## INFO

- **PROV-026 — Positive: the hub is import-safe.** `__main__` guard (1378), lazy `build_data_loaders` (1168), and only cheap top-level side effects (`spike_grad` 148, `TASKS` 955). Unlike the orphan, importing this module does **not** start the 4-5h run — the property that makes 19 dependents viable. Preserve it.
- **PROV-027 — SpiNNaker-non-portable ops live here:** `LayerNorm` in Hopfield (228), HGRN (243-245) and heads (332); `Softmax` gate (327); HGRN multiplicative reset `r * h_prev` (254, risk R3). Not a defect — recorded so the port references the hub source. Mapping ≠ inference ≠ hardware.

---

## Cross-file / clone fix map (for downstream fixers)

| Finding | This file | Orphan clone `code/m7_trimodal_run.py` |
|---|---|---|
| PROV-001 replay | 435, 444, 446-458, 611-615, 1011, 1050, 1079, 1102 | 380-392, 553, 569, 584, 596 |
| PROV-004 best-weights | 662-673, 1003, 1054, 1106 | 484-505 |
| PROV-005 forgetting | 1132-1135 | 613-615 |
| PROV-006 CSV schema | 849-905, 1137-1159 | 618-697 |
| PROV-008 fwt mislabel | 1064-1066, 1120 (correct here) | 616, 630 (mislabeled) |
| PROV-014 SupCon self-positive | 156-165, 181-203 | 186-199 |

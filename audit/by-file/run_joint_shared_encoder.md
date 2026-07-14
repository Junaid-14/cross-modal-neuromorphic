# Audit — `code/run_joint_shared_encoder.py` (1004 lines)

**Shard:** `w4a_jnt` · **Cluster:** `E-joint` · **Prefix:** `JNT` · **Findings:** 17 (JNT-016…JNT-032)
**Method:** static read only. Nothing imported or executed.

## What the script does

Tests whether a **shared encoder trained jointly on digits 0–5 only** learns a cross-modal geometry
that **generalizes to held-out digits 6–9**. Unlike the magnitude ablation, this protocol is
genuinely clean on the training side: stems + `shared_core` + the 6-way `classifier` are trained
**exclusively on 0–5** (`build_paired_train_val_loaders` filters to `TRAIN_CLASSES`), and a correct
leakage assert enforces it (782-787). Held-out 6-9 are only ever seen at test time, so the retrieval
metric is a real generalization test. A `shuffled_labels` control **is** implemented here (512-514).

Architecture: own `VisualStem`/`AudioStem` (5 LIFs, ending in `fc_proj`→256) → `SharedCore`
(GELU, 128-d, L2-normalised) → 6-way `classifier`. Metrics (`compute_held_out_metrics`,
`run_procrustes`) are copied "faithfully from v2" (610). Imports only `CrossModalSupConLoss` and
`spike_grad` from `m7_trimodal_run_provenance` (47-50) and re-implements everything else, which is
the root of most of the duplication findings below.

The headline weaknesses are not in the training protocol but in **reporting** (hardcoded baselines),
**data-plumbing correctness** (SHD label convention), a **meaningless metric**, and **duplication /
config-drift** against the provenance library.

---

## HIGH

### JNT-016 — Hardcoded baseline numbers written into the report as if measured (RULE 1)
The generated `report.txt` embeds a head-to-head baseline as **literals** (990-991):
```
Frozen-separate held-out V2A:  ~0.22 ± 0.20 (chance = 0.25)
Frozen-separate Procrustes:     0.00
```
These are dressed with `~` and `± 0.20` as if they came from a run, but they are hardcoded and there
is **no results CSV in the repo** to back them. This is a direct RULE 1 violation and it lands in a
file a teammate would quote. (The out-of-scope `run_correspondence_anchor_v2.py:716-718` has the same
anti-pattern.) Read baselines from their source CSV at runtime, or drop them.

---

## MEDIUM

### JNT-017 — Parallel `Config` dataclass (config-drift hazard)
`Config` (72-88) re-declares `lr=1e-3`, `weight_decay=1e-4`, `beta=0.9`, `batch_size`, `num_workers`,
`seeds` with its **own** defaults, parallel to `ExperimentConfig`
(`m7_trimodal_run_provenance.py:33-74`). Two independent sources of truth for the same knobs means a
change to β or lr in one place silently fails to propagate. Compose/derive from `ExperimentConfig`
for the shared hyperparameters; keep only the experiment-unique fields (`d_stem`, `d_embed`,
`supcon_temperature`, `samples_per_class`, `epochs`) local.

### JNT-018 — Own `VisualStem`/`AudioStem` — third divergent copy
280-320 / 323-352 duplicate the provenance backbone (`258-297`, `307-317`) and differ from the
magnitude ablation's stems (this one adds `fc2` 1024→512 and `fc_proj` 512→`d_stem`, 5 LIFs vs the
ablation's 4). See JNT-008; one shared parameterized stem module resolves both.

### JNT-019 — Import-time filesystem side effect
`RESULTS_DIR.mkdir(parents=True, exist_ok=True)` runs at **module top level** (56-57), so merely
importing this module creates `results/joint_shared_encoder/` on disk — the exact import-time side
effect the SPEC flags. The magnitude ablation correctly defers its `mkdir` into `main()` (866). Move
it into `main()`/`run_one_seed`.

### JNT-020 — SHD train cache uses `%10` labels, test cache stores raw 0–19
Train cache: `int(l) % 10` (414). Test cache: `int(l)` **without** `%10` (472). The test side is only
reconciled **post-hoc** by `aud_lbl_all = aud_lbl_all % 10` in `run_one_seed` (848). That `%10` is
load-bearing: without it, the `HELD_OUT_CLASSES=(6,7,8,9)` mask would drop German 6-9 (labels 16-19)
and the `TRAIN_CLASSES` mask would drop German 0-5. Two label conventions under the same
`shd_*_dense.pkl` filename convention is a latent trap for any reader or reuse. Apply `%10` at
cache-build time for **both** train and test (472), then the 848 fix becomes a removable no-op.

### JNT-021 — Meaningless "held-out classifier accuracy"
The classifier has `NUM_TRAIN_CLASSES=6` outputs (373), trained on 0-5. For held-out 6-9 the code
computes `argmax == (lbl - 6).clamp(0,5)` (879-880) — relabeling 6→0,7→1,8→2,9→3 — and stores the
match rate as `held_out_classifier_acc_vis/aud` (891-892), which then flows into the per-seed CSV and
`summary_normal.csv`. The comment itself calls it "random guesses". The number has no defensible
interpretation; remove it or replace with the embedding-retrieval metric already computed.

### JNT-022 — Accuracy without spike-rate / timestep cost
`acc_vis`/`acc_aud` (578-579), logged each epoch (823-825) and the retrieval metrics come from
spiking stems (sum over T, 320/352), with no spike rate/sparsity or timestep count reported. Same
rule as JNT-007; log sparsity and T (25 visual / 100 audio) beside every accuracy.

### JNT-023 — Eager full-dataset materialization
`ClassFilteredDataset.__init__` loops `base_dataset[i]` over the **entire** dataset just to read
labels (104-107) — for `cached_nmnist` (DiskCachedDataset over `ToFrame`) that decodes every N-MNIST
sample. `PairedDigitDatasetFiltered.__init__` then re-indexes the full visual + audio datasets again
(143-155). N-MNIST train is decoded several times per seed before any training. Expose a labels-only
accessor / cache the label array once.

### JNT-024 — `PairedDigitDatasetFiltered` encodes by class VALUE, decodes by positional INDEX
Encode: `int(self.classes[c] + s*self.num_classes)` (186, uses the class **value**). Decode:
`c = flat_idx % self.num_classes; s = flat_idx // self.num_classes` (193-196, positional **index**),
then `self.visual_indices[c]` keyed by value (203). This is correct **only** when
`self.classes == (0,1,…,N-1)`. It is always called with `TRAIN_CLASSES=(0,1,2,3,4,5)` so it works
today, but any non-zero-based/non-contiguous class list silently mis-decodes. Make encode/decode
symmetric (`c_index + s*num_classes`, map to value via `self.classes[c_index]`), or assert the
class list is `range(num_classes)`.

---

## LOW

- **JNT-025** — `run_one_seed(seed, cfg, shuffled_labels)` (764) is a fourth same-named per-seed
  runner with a distinct signature (cf. `m7_trimodal_run_provenance.py:962`,
  `run_correspondence_anchor_v2.py:524`, and the ablation's `run_one_arm` 756). Rename for clarity.
- **JNT-026** — own `nmnist_collate_fn`/`shd_collate_fn` (241-273) duplicate
  `m7_trimodal_run_provenance.py:107-123`; the ablation imports the provenance ones instead
  (inconsistent reuse). Import them.
- **JNT-027** — `_events_to_dense` (258-268) is another SHD-densifier copy (JNT-011); its
  `(100,1,1,700)` output is then squeezed at `shd_collate_fn` (273) and again in
  `PairedDigitDatasetFiltered.__getitem__` (226-227) — three shape-normalization sites for one tensor.
- **JNT-028** — unused imports `os` (24), `snntorch.surrogate` (42), `typing.Optional` (31);
  `NUM_HELD_OUT_CLASSES` (64) defined but never used; `SharedCore` takes `beta` (358) but has no LIF
  and ignores it.
- **JNT-029** — module-global `DEVICE` (66) is MPS-or-CPU only (no CUDA) and fixed at import, so a
  CUDA-only host silently runs on CPU; diverges from provenance `get_device`
  (`m7_trimodal_run_provenance.py:94-101`) which the ablation uses (903).
- **JNT-030** — `best_state` can be undefined → `NameError` at `model.load_state_dict(best_state)`
  (839) if the first epoch's val loss is NaN (`NaN < inf` is False) or `cfg.epochs==0`. Initialise
  `best_state = copy.deepcopy(model.state_dict())` before the loop.
- **JNT-031** — the shuffled-control permutation `torch.randperm(...)` (513, 560) is built on CPU
  while `all_labels` is on the MPS `DEVICE`; cross-device indexing `all_labels[perm]` can raise on
  MPS/CUDA (version-dependent; torch pinned 2.11.0), and the shuffled-control call is unguarded in
  `main` (930). Build the permutation with `device=all_labels.device`. *(confidence: low)*
- **JNT-032** — the (correct) leakage assert iterates the full train and val loaders with a per-label
  Python loop before training (782-787); vectorize with `torch.isin`.

## Notes for the fixing agent
- The **training/held-out protocol here is sound** — 6-9 are truly held out and there is a real
  shuffle control. The problems are reporting (JNT-016), a bogus metric (JNT-021), data-plumbing
  fragility (JNT-020, JNT-024), and heavy duplication/config-drift against the provenance library
  (JNT-017, JNT-018, JNT-026, JNT-027).
- Contrast with the magnitude ablation: that file's held-out set **is** confounded (JNT-002) and its
  shuffle control **is** fake (JNT-001); this file gets both of those right. Do not "harmonize" them
  by copying the ablation's approach.
- **RULE 1:** aside from the hardcoded baselines in JNT-016 (which are literals in the source, not
  measured), no number in this audit is asserted as a result; there is no results CSV in-repo.

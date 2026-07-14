# Audit — `code/run_joint_diagnostic.py` (700 lines)

Shard `w4b_cmp` · cluster `E-joint` · prefix `CMP` · status `live`
Static/text-only audit. The file was **not** imported or executed (importing it pulls the
whole magnitude-ablation + provenance stack; see the cross-file notes).

## What the script does

A 2×2 diagnostic — classifier **SHARED vs SEPARATE** × cross-modal SupCon **ON vs OFF** — to
decide whether the magnitude-ablation OFF arm's held-out V2A≈50% (seed 42) is a real
cross-modal effect or an artifact of a shared-classifier leak + a degenerate 4-class metric.
It reuses the magnitude experiment's numerics (model, stems, loaders, cache, SupCon) without
editing that file, adds a `SharedHeadJointModel` positive control, and **replaces** the
quantized 4-class prototype-argmax metric with a per-sample train-prototype retrieval metric
plus a permutation null and a binomial significance test. Verdict text is auto-generated.

The core experimental logic is **sound** (see CMP-027): retrieval is on the encoder embedding
`z`, and the SHARED cell's phase-2 within-modality CE backprops through one shared head for
both modalities, which is exactly the leak mechanism being isolated. The issues are in the
significance test, some dead/wasted `legacy_*` machinery, the import-time pickle hack, and
duplication.

---

## MEDIUM

### CMP-018 (research-integrity) — "Above chance" rests on a binomial z that assumes independent trials
`_binom_z` (451-460) and `_above_chance` (463-466):

```python
se = (p0 * (1.0 - p0) / max(n, 1)) ** 0.5      # n = n_vis_heldout
return (p_hat - p0) / (se + 1e-12)             # thresholded at z > 6.0
```

`n = n_vis_heldout` is the count of held-out (6-9) N-MNIST test samples — on the order of
thousands. But every query is scored against the **same** prototypes with the same argmax
structure, so the trials are strongly correlated (pseudo-replication). The binomial SE is
understated and z is inflated: with n in the thousands, an accuracy only ~13% (barely above
the 10% chance) already clears z>6. This statistic drives the written verdicts — "CLASSIFIER-
LEAK mechanism", "no emergent cross-modal generalization" (546-580). Use a test that accounts
for the correlation (per-class / per-prototype resampling, or the permutation null with a
robust spread), and stop equating sample count with independent trials. The script's own note
(456-458) already distrusts the perm-null *std*; the fix is a better test, not a bigger n.

### CMP-019 (research-integrity) — `legacy_untrained_v2a` computed but never analyzed; `legacy_proto_a2v` written but never read
`run_one_diagnostic_cell` computes `legacy_untrained = compute_held_out_metrics(untrained, …)`
(line 374) — a **full two-modality embedding pass over the test sets** — only to populate the
`legacy_untrained_v2a` column (416), which no summary or report consumes. `legacy_proto_a2v`
(415) is likewise written to every CSV row but never read. Only `legacy_proto_v2a` is actually
used (the artifact-collapse cross-check, 601-604). So part of the `legacy_*` surface is dead
code plus a wasted expensive forward pass per cell. This is the "dead/confusing `legacy_*`
fields" concern from the assignment: the `legacy_proto_v2a` cross-check is *legitimate and
should stay* (with a clarifying comment), but `legacy_untrained_v2a` + `legacy_proto_a2v` are
dead. Drop them (and the untrained `compute_held_out_metrics` call) or actually use them.

### CMP-020 (safety) — Module-level pickle-alias hack mutates `sys.modules["__main__"]` at import
Lines **67-74** inject `CachedNMNIST`/`CachedSHD`/`SubsampledByClass` into `__main__` so a
`torch.load(weights_only=False)` of a cache pickled under `__main__` (from the magnitude probe
run as main) resolves without a 23 GB re-render. Import-time side effect / `__main__` namespace
pollution, duplicated in `run_compositional_crossmodal.py:87-95`, enabling an arbitrary-pickle
`torch.load`. **Why it's needed:** `_build_modality_loaders` calls
`torch.load(<cache>, weights_only=False)`, and pickle records the class's module as `__main__`
when the producing script ran as `__main__`; a *different* `__main__` (this script) then can't
resolve the class. **How to remove it:** rebuild the caches so the classes are pickled under
the importable `run_joint_magnitude_ablation` module, or load them through a custom `Unpickler`
that maps `__main__.<cls>` → the module class — then delete the `__main__` mutation and share
one helper across both scripts.

### CMP-021 (safety) — With no cache, running triggers a ~23 GB tonic render
`main` (659-668) → `_build_modality_loaders`, which renders from tonic when the cache is
absent. No `results/` cache exists in-repo, yet the comment (66-70) says the alias lets "the
existing cache load without a 23 GB re-render". The `__main__` guard (699-700) keeps *import*
safe, but explicit execution silently starts the full render/download. Fail fast when the
cache is missing.

### CMP-022 (research-integrity) — Timestamped rerun CSVs double-count seeds in aggregation
`load_rows_from_disk` (469-484) globs `cell_*_seed*.csv`, matching both the base and the
`__<ts>.csv` variant `_cell_csv_path` (357-362) writes when the base exists.
`write_summary_and_report` (487-505) means over all gate-passing rows per `(kind, use_supcon)`
with no collapse to unique seed, so a rerun inflates `n_seeds` and corrupts mean/std — the
same near-duplicate inflation documented for `generate_figures_3seed`. Collapse to one row per
`(model_kind, use_supcon, seed)` before aggregating.

### CMP-024 (architecture) — Structural clone of the sibling / magnitude harness
`set_seed` (130), `_build_cfg` (624), `_mean`/`_std` (441-448), `_binom_z` (451),
`load_rows_from_disk` (469), `write_summary_and_report` (487), `_report_spiking_stats` (336),
and the main/aggregate scaffolding are near-duplicates of `run_compositional_crossmodal.py`
and `run_joint_magnitude_ablation.py`. Extract a shared importable harness.

---

## LOW

- **CMP-023 (dead-code)** — Line 65: `from m7_trimodal_run_provenance import ExperimentConfig
  as _EC  # … (ensures import path)` is unused; the import path is already established by
  importing `run_joint_magnitude_ablation`. Remove it.
- **CMP-025 (quality)** — `_build_cfg` (624-636) sets ~30 fields; only device,
  `nmnist_train_per_class`, `batch_size`, `lr`, `weight_decay`, `max_epochs`, `patience` are
  consumed. The Hopfield/replay/align/tri-modal fields are inert noise.
- **CMP-026 (docs)** — Docstring (9-16) cites unsourced result numbers ("V2A=50%", "an
  UNTRAINED model also scores 50% (:769)") — under RULE 1 these are unsourced in-repo (no CSV)
  — and hard cross-file line references (`:378-379`, `:572-632`, `:769`) that will rot. The
  refs are accurate today (verified: separate classifiers at 378-379; `compute_held_out_metrics`
  572-632; the untrained control near 768-769). Mark the 50% as an unsourced prior observation
  and replace line numbers with symbol names.

## INFO

- **CMP-027 (correctness, positive)** — The 2×2 design is correct. Retrieval is measured on
  encoder `z` (`per_sample_retrieval` and `compute_held_out_metrics` both call `model.embed`),
  not on classifier logits; the SHARED cell's phase-2 within-modality CE (204-213) backprops
  through the single `shared_classifier` for both modalities, inducing cross-modal alignment of
  the 6-9 embeddings with **no** cross-modal supervision — a valid positive control for the
  leak, absent in the SEPARATE cell. The per-sample retrieval with **TRAIN** prototypes
  (282-333, prototypes from the full train set, queries from the test set) is a genuine
  improvement over the transductive quantized 4-class metric it replaces, and its stated
  chances (10% all-10 / 25% held-4) are correct. `_report_spiking_stats` (336-349) here is a
  **genuine** spike rate — the `VisualStem`/`AudioStem` it probes both sum LIF spikes over T
  (contrast the compositional file's visual stem, CMP-003).

---

## Cross-file notes
`JointMagnitudeModel`, the stems, `_build_modality_loaders`, `_build_paired_0to5_loaders`,
`_within_modality_probe`, `_extract_embeddings`, `compute_held_out_metrics`, `_accuracy_on_all`
are all defined in `run_joint_magnitude_ablation.py` (import-safe, `__main__` guard at 1006,
no import-time data load). `CrossModalSupConLoss`/`ExperimentConfig`/`get_device` originate in
`m7_trimodal_run_provenance.py` (guard at 1378). The diagnostic's STEP-0 docstring claims
(separate classifiers → no shared-leak possible in the magnitude experiment; the 4-class metric
is quantized to 25% steps) are **verified**: `JointMagnitudeModel` has separate
`visual_classifier`/`audio_classifier` (378-379), and `compute_held_out_metrics` (572-632)
returns `v2a_correct / len(held_out_classes)` over 4 classes → values in {0, .25, .5, .75, 1}.

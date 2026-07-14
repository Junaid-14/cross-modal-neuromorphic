# Audit — `code/run_joint_magnitude_ablation.py` (1007 lines)

**Shard:** `w4a_jnt` · **Cluster:** `E-joint` · **Prefix:** `JNT` · **Findings:** 15 (JNT-001…JNT-015)
**Method:** static read only. Nothing imported or executed (this file has a `__main__` guard, but its
dependency `m7_trimodal_run.py` does not, and the SPEC forbids import either way).

## What the script does

Trains a **single shared encoder from scratch** on N-MNIST (visual) + SHD (audio) and ablates one
variable: a **shared magnitude head** (ON vs OFF) that regresses digit value on the L2-normalised
embedding `z`. Cross-modal supervised-contrastive loss is applied to digits **0–5 only**; digits
**6–9 are the held-out set** whose cross-modal retrieval is the headline measurement. Per modality
there are **separate** classifiers (`visual_classifier`, `audio_classifier`, 378-379) — so a
shared-classifier leak is genuinely impossible (this is independently confirmed by
`run_joint_diagnostic.py:11`). The magnitude head, however, **is shared** (381).

Architecture: `VisualStem`/`AudioStem` (own re-implementations) → `shared_core` (GELU MLP, 371-376)
→ `z` (128-d, L2-normalised). Loss on the paired 0–5 batch = `CE(vis)+CE(aud)+SupCon`, plus a
separate full per-modality classification pass over **all** classes each epoch (518-530).

It imports the real shared library `m7_trimodal_run_provenance` (config, device, collates, loss) and
`Dataloaders.paired_digit_loader` — good — but also imports two symbols from
`run_correspondence_anchor_v2` it never uses (JNT-009).

---

## HIGH

### JNT-001 — The "shuffled-label control" does not shuffle labels
`run_one_arm` (809-817) builds `model_s` and calls `train_model(model_s, ...)` for 40 epochs. The
comment (815) says *"Permute audio labels within each batch during training"* — but `train_model`
(457-551) has **no `shuffled_labels` parameter at all**, and its loss uses the true labels
throughout (494-506). So `shuffle_ho` / `result["shuffle_v2a"]` (817, 842) is just a **second normal
run with fewer epochs**, reported as a null control. This is the one control that is supposed to
prove the effect is not an artifact, and it is inert. The sibling scripts implement it correctly
(`run_correspondence_anchor_v2.py:305-306`, `run_joint_shared_encoder.py:512-514`) — copy that.

### JNT-002 — "Held-out 6-9 never receive a cross-modal signal in either arm" is false
The header docstring (11-12) is the experiment's core claim of a clean held-out protocol. It is
contradicted by the code:
- **ON arm:** the *shared* `magnitude_head` (381) is trained to regress `y/9.0` on **all** classes,
  including 6-9, for **both** modalities in the within-modality pass (524-526). That is a shared
  supervised axis directly aligning held-out digits across modalities — exactly the "engineered, not
  emergent" bridge `CLAUDE.md` warns against.
- **Both arms:** the *shared* `shared_core` (371-376) receives within-modality CE gradients from 6-9
  samples of both modalities (518-523), co-shaping the very `z`-space that held-out retrieval
  measures (794-797).

The report's own verdict text concedes the ON side-channel (970-971, 996-997) — so the docstring and
the report disagree with each other. A reader who trusts the docstring would wrongly conclude
emergent modality-invariant structure. Fix the docstring; consider an arm that excludes 6-9 from the
within-modality pass to actually isolate emergence.

---

## MEDIUM

### JNT-003 — Silent, outcome-dependent seed exclusion
`run_one_arm` asserts `probe["visual"] > 40` and `probe["audio"] > 40` (800-801). `main` wraps each
arm/seed in a `try/except` that prints a traceback and continues (917-928). A seed whose probe dips
below 40 % is therefore **dropped from `all_results`**, and the aggregate is computed over the
survivors with `n=len(sub)` (940, 948-951). Filtering seeds on a performance threshold is
outcome-dependent selection that inflates the mean and shrinks the reported spread. Record the probe
and report it, or fail the whole sweep — do not silently discard.

### JNT-004 — Leakage assert uses a non-element-wise idiom
`assert not ((labels >= 6).any() and (labels <= 9).any())` (488, duplicated 776, 778). The two
`.any()` calls are independent, so this reads as *"no element ≥6 anywhere AND no element ≤9
anywhere"* — **not** the intended *"no element in [6,9]"*. For the 0–9 label range it happens to
catch a leak (a stray 8 is both ≥6 and ≤9), so it is not live-broken today, but it is the wrong
idiom and will mislead. Replace with `assert not ((labels >= 6) & (labels <= 9)).any()`.

### JNT-005 — `CachedNMNIST`/`CachedSHD` pickled as `__main__` (confirmed downstream damage)
The dataset classes are defined in this run-as-`__main__` script and saved **whole** with
`torch.save` (184-185, 197-198), reloaded with `torch.load(..., weights_only=False)` (172-173,
190-191). Pickled under `python run_joint_magnitude_ablation.py`, their class path is
`__main__.CachedNMNIST`, so **any other process must recreate those symbols under `__main__`** to
load the cache. This has already forced identical alias hacks in two importers:

- `run_compositional_crossmodal.py:92-95`
- `run_joint_diagnostic.py:71-74`

both do `import run_joint_magnitude_ablation as _mag_module; for _cn in ("CachedNMNIST","CachedSHD",
"SubsampledByClass"): setattr(sys.modules["__main__"], _cn, ...)`, with comments explicitly blaming
this pickling. `weights_only=False` is also an arbitrary-unpickle code-execution surface. **Fix:**
move `CachedNMNIST`/`CachedSHD`/`SubsampledByClass` into an importable module (e.g.
`code/joint_datasets.py`) and cache **tensor state**, not Dataset objects.

### JNT-006 — `_magnitude_sanity` reports a mislabeled, tie-broken Spearman
Docstring (724) promises *"Spearman rho between |z| and digit value, and MSE of magnitude head."*
The code (723-753) instead: computes `mag` (733) then **discards it**; PCA-projects `z` onto its
first component (743-746); and applies the **no-ties** Spearman closed form
`rho = 1 - 6·d²/(n(n²−1))` (751-752) to data with ~n/10 ties per digit value. So the reported
`mag_rho_visual/audio` (845-846, summarised 986-987) is a PC1-vs-value correlation with a biased
formula, labelled as a magnitude-head sanity check. Use `scipy.stats.spearmanr` on the actual head
output, or rename the metric.

### JNT-007 — Accuracy without spike-rate / timestep cost
`acc_visual_all`/`acc_audio_all` (786-787), the 6-9 probes (788, 829-830) and the printed/CSV/report
figures (819-843) come from spiking stems that time-sum spikes (324, 352), but no spike
rate/sparsity or timestep count travels with them. Repo rule: accuracy never travels alone. Log
per-stem sparsity and T (25 visual / 100 audio) beside every accuracy.

### JNT-008 — `VisualStem`/`AudioStem` duplicated from the provenance backbone
288-324 and 327-352 re-implement the canonical stems
(`m7_trimodal_run_provenance.py:258-297` visual, `307-317` audio) with a different tail (256-d,
4 LIFs). `run_joint_shared_encoder.py:280-352` is yet a **third** variant. Extract one parameterized
stem module and consume it everywhere; if the joint experiments intend a smaller stem, document that
in one place.

### JNT-009 — Dead imports from `run_correspondence_anchor_v2` drag in a heavy transitive dep
`compute_held_out_metrics as _compute_held_out_metrics_v2` and `run_procrustes` (56-59) are **never
used** — the file has its own `compute_held_out_metrics` (572) and `run_procrustes_z` (648, called at
795). But importing `run_correspondence_anchor_v2` transitively runs
`from utils.cross_modal_transfer import evaluate_cross_modal_transfer`
(`run_correspondence_anchor_v2.py:45`), so an unrelated break in v2 or `utils` fails this script at
import for functionality it does not use. Delete the block.

---

## LOW / INFO

- **JNT-010 (low)** — unused imports: `json` (23), `math` (24), `TensorDataset` (34),
  `SHDDigitIdentityDataset` (44). grep-confirmed no usages.
- **JNT-011 (low)** — `_events_to_dense_shd` (72-82) is one of four copies of the SHD densifier
  (`run_joint_shared_encoder.py:258-268`, `m7_trimodal_run_provenance.py:1200-1210`,
  `run_correspondence_anchor_v2.py:93-103`), with differing output shapes.
- **JNT-012 (low)** — paired loaders are fully iterated once just for the (broken) leakage assert
  (775-778) before `train_model` iterates them again.
- **JNT-013 (low)** — `ExperimentConfig` is instantiated with ~30 M7/replay/hopfield/align fields
  this experiment never uses, incl. `task_order=[...,"dvs"]` though there is no DVS or CL here
  (868-902). Only `lr, weight_decay, batch_size, max_epochs, patience, nmnist_train_per_class,
  device` are consumed.
- **JNT-014 (low)** — aggregates use population `np.std` (ddof=0) labelled "± std" over n≤5
  (948-951, 959-962, 978-991), understating spread; the set also mixes in the non-control shuffle.
- **JNT-015 (info)** — `SEEDS=[42,123,456,789,1011]` (62): 5th seed `1011` differs from the
  provenance default `101112` (`m7_trimodal_run_provenance.py:63`), and none of this matches the
  `[42,123,456]` headline policy. Cross-file seed-set inconsistency (matches
  `run_joint_shared_encoder.py:85`).

## Notes for the fixing agent
- `main()` has a proper `__main__` guard (1006-1007) and defers its `mkdir` into `main()` (866) —
  both good, and in contrast to `run_joint_shared_encoder.py` (JNT-019).
- The most consequential fixes are **JNT-001** (the control is fake) and **JNT-002** (the held-out
  claim is confounded); both change how the *results* should be described, not just the code.
- **RULE 1:** every accuracy/retrieval number this script prints is, in-repo, unsourced (no CSV
  exists). Nothing in this audit asserts any measured value; all cited numbers are literals in the
  source (thresholds, chance, `y/9.0`, dims).

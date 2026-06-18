# Cross-Modal Correspondence: Methods and Controlled Negative Results

This document drafts the Methods and Controlled Negative Results sections for the
N-MNIST ↔ SHD cross-modal correspondence study. The study asks whether visual
and auditory digit representations share a *generalizable* geometry, or whether
successful cross-modal transfer is merely an engineered artefact of the
training objective.

---

## 1. Methods

### 1.1 Datasets and pre-processing

We use two neuromorphic digit-classification datasets:

* **N-MNIST** (visual): event-based recordings of MNIST digits, converted to
  dense frames with Tonic `ToFrame(sensor_size=(34,34,2), time_window=1000)` and
  `Denoise(filter_time=10000)`. For the joint shared-encoder experiments, frames
  are truncated or zero-padded to a fixed length of 25 time steps.
* **SHD** (audio): spiking-heidelberg-digits recordings, converted to a dense
  binary tensor of shape (100, 700) by binning events into 100 time bins across
  700 frequency channels. Labels are reduced modulo 10 to align with N-MNIST
  digit identities 0–9.

All experiments split the ten digit classes into **train classes** {0,1,2,3,4,5}
and **held-out classes** {6,7,8,9}. Models are trained only on digits 0–5;
digits 6–9 are reserved to test whether any learned cross-modal structure
generalizes to *unseen* identities.

### 1.2 Frozen-backbone correspondence v2

The first protocol tests whether separately trained, frozen backbones already
contain a cross-modal digit geometry that can be exposed by a lightweight
post-hoc alignment head.

**Base model.** We take the M7 continual-learning model
(`M7_ContinualAdaptiveModel`) trained on N-MNIST → SHD without any explicit
cross-modal alignment loss (checkpoint
`results/cl_confirmation/checkpoints/A_no_align_seed42/final_shd_shd.pt`). Its
N-MNIST and SHD backbones are kept frozen throughout. We verify non-drift by
MD5 hashing backbone parameters before and after correspondence training, and by
confirming single-task accuracy does not change.

**Correspondence heads.** On top of the frozen 512-dimensional backbone
features we train two head types:

* **Linear head**: `Linear(512, 64)`.
* **MLP head**: `Linear(512, 256) → GELU → Linear(256, 64)`.

Each head projects backbone features into a 64-dimensional anchor space. We use
10 learned anchor vectors (one per digit 0–9) and train the heads with a
cross-entropy objective that maps each modality's projection to the correct
digit anchor.

**Held-out variant.** To test generalization, we re-train heads on digits 0–5
only (6 anchors) while masking out digits 6–9, and then evaluate retrieval on
digits 6–9.

**Procrustes corroboration.** Independently of the learned heads, we compute
per-digit prototypes in the frozen 512-d backbone space and fit an orthogonal
rotation `R` from visual prototypes to audio prototypes using only the train
classes {0,…,5}. We then test whether `R` correctly retrieves held-out audio
prototypes from held-out visual prototypes.

All v2 experiments were run with 8 seeds
{42, 123, 456, 789, 1011, 1213, 1415, 1617} for both head types. Code:
`code/run_correspondence_anchor_v2.py`; outputs:
`results/hypergraph_anchor_v2/`.

### 1.3 Joint shared encoder

The second protocol removes the "separate backbones" assumption entirely: a
single encoder is trained from scratch on paired N-MNIST/SHD examples of digits
0–5. If co-development is sufficient to learn a generalizable cross-modal
geometry, the shared space should transfer to digits 6–9 even though those
identities were never seen during training.

**Architecture.**

* **Visual stem**: Conv2d → MaxPool → LIF → Conv2d → MaxPool → LIF →
  Linear(64·5·5, 1024) → LIF → Linear(1024, 512) → LIF → Linear(512, 256),
  with spike responses summed over time.
* **Audio stem**: Linear(700, 1024) → LIF → Linear(1024, 1024) → LIF →
  Linear(1024, 512) → LIF → Linear(512, 256), with spike responses summed over
  time.
* **Shared core** (shared weights across modalities): `Linear(256, 128) → GELU`
  followed by L2 normalization. We initially used a spiking LIF non-linearity
  but found it produced zero output and blocked learning; replacing it with GELU
  restored training.
* **Shared classifier**: `Linear(128, 6)` for digits 0–5, also shared across
  modalities.

**Training.** Pairs are formed by yoking one N-MNIST sample and one SHD sample
with the same digit label. We use 500 samples per class (3,000 training pairs),
with a 90/10 train/validation split. The model is optimized with AdamW
(lr=1e-3, weight_decay=1e-4) for up to 100 epochs with early stopping
(patience=10). The loss is the sum of cross-entropy on both modalities plus a
supervised contrastive (`CrossModalSupConLoss`, temperature=0.1) loss on the
concatenated L2-normalized embeddings.

**Controls.** We include:

* **Label-shuffled control**: the same architecture trained with randomised
  labels within each batch, to verify chance-level held-out performance.
* **Untrained encoder baseline**: held-out retrieval before any training.
* **Leakage asserts**: every training batch is checked to contain only labels
  0–5.

Experiments were run with 5 seeds {42, 123, 456, 789, 1011}. Code:
`code/run_joint_shared_encoder.py`; outputs: `results/joint_shared_encoder/`.

### 1.5 Ordinal anchor diagnostic (B5)

Because digits are ordered by magnitude, one possible escape from the
zero-shot ceiling is that the backbones might encode digit *ordinality*
geometrically, allowing classes 6–9 to extrapolate from 0–5. We test this in
two stages.

**Stage 0 — RDM diagnostic.** We compute 10×10 representational dissimilarity
matrices (RDMs) from the frozen backbone prototypes for all digits 0–9. We
report (i) the Mantel correlation between the vision and audio RDMs, and (ii)
the Spearman correlation of each RDM with the absolute digit-difference
|i−j| (magnitude monotonicity). This diagnostic is cheap and decisive: if
neither RDM is ordinal and the RDMs are uncorrelated, no anchor structure can
rescue generalization.

**Stage 1 — Ordinal anchors.** We replace the v2 orthonormal anchors with a
shared circular/ordinal anchor structure: anchor(d) is a unit vector at angle
π·d/9 in a 64-dimensional space. Similarity between anchors decreases smoothly
with digit distance. We then repeat the v2 protocol: train linear and MLP
projection heads on digits 0–5 only, test held-out retrieval on 6–9, and
record the confusion structure (ordinal-adjacent vs non-adjacent errors).

Code: `code/run_ordinal_anchor.py`; outputs: `results/ordinal_anchor/` and
`results/structural_diagnostic/`.

### 1.4 Evaluation metrics

For a set of classes `C`, we compute per-class prototypes by averaging and
L2-normalizing embeddings. Given visual prototypes `{p_v(c)}` and audio
prototypes `{p_a(c)}` for `c ∈ C`:

* **Prototype alignment**: mean cosine similarity between same-class prototypes,
  `(1/|C|) Σ_c p_v(c)ᵀ p_a(c)`.
* **Null alignment**: mean cosine similarity across all off-diagonal pairs.
* **Alignment gap**: same-alignment minus null-alignment.
* **V2A retrieval**: fraction of visual prototypes whose nearest audio
  prototype (by cosine similarity) has the same class label.
* **A2V retrieval**: the symmetric audio→visual fraction.

For the frozen-backbone v2 protocol we also report the main 0–9 cross-modal
transfer accuracy (`evaluate_cross_modal_transfer`) as a sanity check.

Chance levels for 4-way held-out retrieval are 25%. The joint encoder's
held-out classifier is a 6-way output trained on 0–5; evaluated on 6–9 it serves
only as a diagnostic, not a primary metric.

---

## 2. Controlled Negative Results

### 2.1 Post-hoc alignment is an engineered bridge

Training lightweight correspondence heads on frozen, separately trained
backbones yields excellent *in-domain* cross-modal retrieval on digits 0–9
(MLP 84.4% V2A; Linear 80.0% V2A), but this is expected: the heads and anchors
are explicitly optimised to map the two modalities onto a shared digit code.
The decisive test is held-out performance on digits 6–9 after training only on
0–5.

| Head type | Main 0–9 V2A | Held-out 6–9 V2A | Held-out gap | Verdict |
|-----------|-------------:|-----------------:|-------------:|:--------|
| Linear    | 79.98 ± 0.99 | 21.88 ± 19.52    | 0.039 ± 0.084 | Engineered bridge |
| MLP       | 84.43 ± 1.17 | 18.75 ± 16.54    | −0.044 ± 0.147 | Engineered bridge |
| Chance    | —            | 25.0             | 0.0          | — |

Both head types perform at chance on held-out classes; the MLP actually scores
below chance on average. Procrustes alignment in the frozen backbone space
produces **0% held-out retrieval** (same-alignment 0.054, null-alignment
−0.002, gap 0.056). This parameter-free orthogonal map, fit on the train
classes, fails completely on the held-out classes.

**Conclusion.** The strong 0–9 transfer is produced by the head, not by an
underlying cross-modal geometry in the frozen backbones. The held-out test and
Procrustes corroboration jointly falsify the claim that separate training on
N-MNIST and SHD induces a digit-identity manifold shared across modalities.

### 2.2 Joint encoder test is confounded, not a fair diagnostic

We also ran an exploratory joint shared-encoder experiment in which a single
encoder is trained from scratch on paired N-MNIST/SHD digits 0–5 only. In-domain
alignment is perfect (V2A/A2V retrieval 100%, prototype alignment 0.954 ±
0.009), confirming the architecture can align seen classes. Held-out retrieval
on digits 6–9, however, is at or near chance, with high seed-to-seed variance:

| Metric | Joint shared encoder (n=5) | Shuffled-label control |
|:-------|---------------------------:|-----------------------:|
| In-domain V2A retrieval | 1.000 ± 0.000 | 1.000 |
| In-domain prototype alignment | 0.954 ± 0.009 | 0.961 |
| Held-out V2A retrieval | 0.400 ± 0.123 | 0.250 |
| Held-out A2V retrieval | 0.500 ± 0.158 | 0.500 |
| Held-out alignment gap | 0.102 ± 0.039 | 0.003 |
| Procrustes held-out retrieval | 0.100 ± 0.123 | 0.000 |
| Held-out within-modality classifier acc (visual) | 0.124 ± 0.013 | 0.113 |
| Held-out within-modality classifier acc (audio) | 0.259 ± 0.036 | 0.221 |

**This test is confounded.** Because the encoder was trained only on digits
0–5, it never learned usable features for digits 6–9 *within either modality*.
Visual held-out classifier accuracy is 12.4% (below the 25% chance level for a
4-class problem) and audio held-out accuracy is only 25.9% (near chance). When
within-modality features for the held-out classes are absent or unusable, the
cross-modal retrieval signal is swallowed by the feature failure. The held-out
result therefore cannot be interpreted as evidence that joint co-training fails
to generalize; it shows only that the test was ill-designed.

A valid diagnostic would train within-modality features on all digits 0–9
while applying the cross-modal loss only on 0–5, thereby isolating alignment
generalization from feature generalization. We leave that corrected test for
future work.

### 2.3 Ordinal anchors do not rescue held-out generalization

Digits 0–9 carry a natural magnitude ordering, so we explicitly tested whether
ordinal-structured anchors could extrapolate from seen (0–5) to unseen (6–9)
classes.

**Stage 0 — RDM diagnostic.** The frozen backbone RDMs are uncorrelated across
modalities (Mantel r=0.051, p=0.74) and neither shows significant magnitude
structure (vision ρ=0.11, p=0.46; audio ρ=−0.04, p=0.80). The prerequisite for
ordinal extrapolation is absent.

**Stage 1 — Ordinal anchor training.** With ordinal anchors, the MLP head again
learns strong in-domain alignment (67.9% V2A), but held-out retrieval on 6–9 is
at chance (20.0% ± 18.7%), identical to the shuffled-label control (25.0%).

| Head type | In-domain V2A | Held-out 6–9 V2A | Shuffle control | Adjacent-error fraction | Verdict |
|:----------|--------------:|-----------------:|----------------:|------------------------:|:--------|
| MLP       | 67.88 ± 1.86  | 20.00 ± 18.71    | 25.00 ± 0.00    | 0.23 ± 0.29             | Engineered bridge |
| Linear    | 33.62 ± 1.73  | 45.00 ± 18.71    | 30.00 ± 10.00   | 0.83 ± 0.21             | Noisy / not reliable |
| Chance    | —             | 25.0             | 25.0            | —                       | — |

The linear head appears to show above-chance held-out retrieval (45.0%) and a
high fraction of ordinal-adjacent errors, but this head fails to fit even the
in-domain task (33.6% V2A vs 67.9% for the MLP). With only four held-out
classes, retrieval is noisy, and the shuffled-label control is also elevated
(30.0% ± 10.0%). We therefore do not treat the linear-head result as
convincing evidence of ordinal generalization. The more reliable MLP result
remains at chance.

**Conclusion.** The Stage 0 diagnostic already predicted failure, and the
follow-up experiment confirms it: even when the anchor structure explicitly
encodes digit ordinality, the frozen backbones do not provide a geometry that
generalizes to unseen classes.

### 2.4 Synthesis: the clean negative result is v2; the joint test is inconclusive; ordinal anchors do not help

The decisive controlled negative comes from the frozen-backbone v2 protocol.
The joint-encoder held-out result is at chance, but because the encoder never
saw digits 6–9 in any modality, that result cannot distinguish alignment
failure from feature failure.

| Protocol | Held-out V2A retrieval | Procrustes held-out retrieval | Interpretation |
|:---------|-----------------------:|------------------------------:|:---------------|
| Frozen backbones + Linear head | 0.219 ± 0.195 | 0.000 | Engineered bridge |
| Frozen backbones + MLP head    | 0.188 ± 0.165 | 0.000 | Engineered bridge |
| Frozen backbones + Ordinal anchors (MLP) | 0.200 ± 0.187 | 0.000 | Engineered bridge |
| Joint shared encoder (0–5 only) | 0.400 ± 0.123 | 0.100 ± 0.123 | Confounded |
| Chance                         | 0.250         | 0.250 | — |

For the frozen-backbone conditions, four independent map families — nonlinear
MLP, linear, parameter-free orthogonal Procrustes, and ordinal-structured
anchors — all fail on held-out classes. The failure is in the encoders:
separately trained visual and auditory backbones do not share a generalizable
digit-identity geometry. No post-hoc head or anchor structure can manufacture
structure that the representations do not contain.

The joint-encoder condition is not a fair comparison to v2. Until within-modality
features for 6–9 are trained, the held-out test only measures feature
generalization, not cross-modal alignment generalization. We therefore do **not**
cite it as evidence either for or against joint co-training.

### 2.5 Theoretical framing: why held-out failure is expected

The held-out failure is not a methodological artefact; it is the theoretically
predicted outcome for zero-shot generalization without shared semantic side
information. Palatucci et al. (2009) and the zero-shot learning literature
(e.g., Xian et al.) establish that generalization to unseen classes requires a
shared semantic basis linking seen and unseen classes. Our fixed orthonormal
anchors provide such a basis for digits 0–5, but the anchors for 6–9 are
arbitrary and orthogonal to those for 0–5. Nothing in the training signal
tells the model that digit 6 is "one more than 5"; hence chance-level
held-out retrieval is the expected result.

This distinguishes our setting from CLIP/DeViSE-style systems, where a
language-embedding space supplies shared semantic structure between seen and
unseen labels, and from the Platonic Representation Hypothesis, whose
convergence claims are conditioned on scale. Small separately-trained SNN
backbones are precisely the regime where convergence is not expected.

### 2.6 Implications and future work

The correspondence architecture is a supervised bridge, not a discovered shared
manifold. It is valuable precisely because it provides reliable, zero-forgetting
cross-modal binding for trained classes. The controlled negative result — that
the bridge does not generalize to unseen digit identities — is the paper's most
important citable contribution, because it is backed by a clean experimental
design, a clear theoretical explanation, and now an explicit test of the most
plausible escape route (digit ordinality).

The M7 continual-learning result should likewise be understood as arising from
architectural isolation (separate backbones and task heads) and replay, not
from an emergent modality-invariant semantic space.

For future work, the corrected joint encoder remains the decisive diagnostic:
train within-modality features on all digits 0–9 while applying cross-modal
alignment loss only on 0–5. This isolates alignment generalization from feature
generalization. Our Stage-0 RDM diagnostic suggests that, for the current
frozen backbones, no post-hoc anchor structure can rescue held-out
generalization; the corrected joint encoder test would reveal whether
co-training can do better.

---

## Reproducibility

| Experiment | Script | Output directory |
|:-----------|:-------|:-----------------|
| Frozen-backbone correspondence v2 | `code/run_correspondence_anchor_v2.py` | `results/hypergraph_anchor_v2/` |
| Joint shared encoder | `code/run_joint_shared_encoder.py` | `results/joint_shared_encoder/` |
| Ordinal anchor diagnostic (B5) | `code/run_ordinal_anchor.py` | `results/ordinal_anchor/` |
| RDM / structural diagnostic | `code/run_ordinal_anchor.py` | `results/structural_diagnostic/` |

All numbers reported above are taken from `summary_linear.csv`,
`summary_mlp.csv`, `summary_normal.csv`, `summary_shuffled.csv`,
`summary_mlp.csv`, `summary_linear.csv`, and `rdm_analysis.txt` in those
output directories.

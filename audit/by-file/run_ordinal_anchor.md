# Audit — `code/run_ordinal_anchor.py` (774 lines)

Cluster **E-alignment** · shard `w3_algn` · IDs **ALGN-015 … ALGN-020**

## What this file is

"Ordinal Anchor Alignment (B5) + Stage 0 RDM Diagnostic." Two stages:

- **Stage 0 (diagnostic):** build 10×10 representational dissimilarity matrices (RDMs)
  from the frozen N-MNIST and SHD backbone prototypes, test whether the two RDMs are
  correlated (Mantel) and whether either is monotonic in digit magnitude (Spearman).
- **Stage 1 (experiment):** replace v2's QR-orthonormal anchors with a shared
  **circular/ordinal** anchor geometry (digit `d` at angle `π·d/(N-1)`), train projection
  heads on 0-5, and test held-out 6-9 cross-modal retrieval, including an
  ordinal-adjacency confusion analysis.

It **reuses v2's helpers** — imports `_hash_params, _load_cached_features,
build_paired_feature_dataset, compute_held_out_metrics, run_procrustes,
train_correspondence` from `run_correspondence_anchor_v2` (51-58). Output:
`results/ordinal_anchor/` + `results/structural_diagnostic/`.

## Safety posture

- Has a `__main__` guard (773). No import-time side effects; imports the guarded
  provenance module. Same clean posture as v2.

## Critical

### ALGN-015 — Stage 1 inherits v2's broken held-out training
Because it imports `train_correspondence` and drives the identical held-out path —
`corr_ho = OrdinalCorrespondenceModel(..., num_anchors=6)` (483-485) trained with
`held_out_classes=(6,7,8,9)` (487-497) on labels 0-9 — it hits **exactly** the
`CrossEntropyLoss` target-out-of-bounds crash + no-op mask documented in **ALGN-001**.
`main()` swallows it via `try/except` (680-684). Stage 1 (the ordinal held-out retrieval
that is the whole point of the file, plus the adjacency-error analysis at 502-507) never
completes. Fixing `train_correspondence` per ALGN-001 fixes this file too.

## High

### ALGN-016 — invalid Mantel permutation null
`_mantel_test` (147-168) computes the observed `r` correctly (correlation of the two
RDMs' upper triangles), but its permutation null **permutes the vectorized 45-element
upper triangle directly** (line 161, `perm = rng.permutation(len(yv))`). A valid Mantel
test permutes the **n=10 underlying objects** — apply one permutation symmetrically to the
rows *and* columns of a RDM (`rdm[np.ix_(perm, perm)]`), then re-extract the triangle.
Shuffling the 45 distances independently destroys the distance-matrix structure and
produces an invalid p-value. This feeds the reported `vision_audio_rdm_p` and the
"RDMs uncorrelated (Mantel r, p)" diagnostic used elsewhere in the project.

### ALGN-017 — `CKPT_SOURCE` and feature cache both missing
`CKPT_SOURCE` (63-70) is the same absent `results/cl_confirmation/.../final_shd_shd.pt` as
in v2; `torch.load(CKPT_SOURCE)` at 410 fails. `main()` also calls the imported
`_load_cached_features` (615) which needs the missing `results/hypergraph_anchor/cache`
(the comment says "from v2" but that cache is actually a **v1** artifact). The script is
un-runnable in the current tree regardless of the Stage 1 bug.

## Medium

- **ALGN-018 (duplication):** `OrdinalCorrespondenceModel` (247-326) is a near-verbatim
  copy of `CorrespondenceModelV2` (v2:164-234) — same heads, `_get_backbone_features`
  (282-299), `project`, `forward_with_shared`, `task_accuracy`. The only substantive
  difference is `build_ordinal_anchors` vs the QR construction. Should subclass and
  override just the anchor builder. Carries a second copy of the audio-forward
  reimplementation (see ALGN-007).

## Low / info

- **ALGN-019 (quality):** `hashlib` (21) is imported but never used here (`_hash_params`
  is imported from v2, where hashlib actually lives); `List` (26) also unused.
- **ALGN-020 (numerics):** `_spearman` (130-144) uses the **no-ties** shortcut
  `1 - 6Σd²/(n(n²-1))` on tie-averaged ranks, and a normal approximation for the p-value.
  Fine-ish for n=45 pairs but biased under ties; if it enters the paper, label it
  approximate or use `scipy.stats.spearmanr`.

## Metric-correctness notes

- RDMs are cosine-distance matrices of per-class prototypes computed on `*_test`
  features — **test**, no val/test confusion.
- `build_ordinal_anchors` (227-244) is correct: unit-norm by construction, dot product
  `cos(π|i-j|/(N-1))` decreasing with digit distance.
- `compute_confusion_matrix` (334-375) and `ordinal_error_score` (378-392) are correct as
  written — but they only ever run on the output of the broken Stage 1 (ALGN-015), so
  they never produce numbers today.
- Retrieval here is again 4-way (chance 0.25) — same coarse/high-variance caveat as
  ALGN-013.

# Audit — `code/run_compositional_crossmodal.py` (1227 lines)

Shard `w4b_cmp` · cluster `E-joint` · prefix `CMP` · status `live`
Static/text-only audit. The file was **not** imported or executed (it pulls the whole
magnitude-ablation + provenance stack transitively; see CMP-017).

## What the script does

Tests whether a jointly-trained SNN encoder **factorizes position** so that an unseen
2-digit number (e.g. "47") aligns across N-MNIST (visual) and SHD (audio) because the
primitives 4 and 7 do. It synthesizes 1- and 2-digit visual canvases (74-wide) and audio
clips (230 steps) from the cached single-digit datasets, trains four models
(`compositional`, `holistic`, `shuffled`, `untrained`), and reports per-sample cross-modal
SET / ORDER / whole-number retrieval on a held-out set of 30 numbers, each against a
permutation null. Verdict text is auto-generated.

The design intent is sound and the integrity instincts are mostly good (permutation nulls,
a within-modality gate, a holistic memorization control, timesteps recorded, CSVs never
overwritten, a git-SHA + split sidecar). The problems are in the **protocol details**, where
several of them quietly undermine the headline "held-out" claim.

---

## CRITICAL / HIGH

### CMP-001 (high, correctness) — Near-miss augmentation leaks held-out numbers into training
`train_compositional` phase (b), lines **534-540**:

```python
for num in train_numbers:
    d1, d2 = num // 10, num % 10
    for _ in range(PHASE_B_COMPOSITES):
        specs.append((d1, d2)); labels.append(num)
    if NEAR_MISS_FLIP:
        d2p = int(rng.integers(0, 10))
        specs.append((d1, d2p)); labels.append(d1 * 10 + d2p)   # shares pos1
```

`d2p` is drawn uniformly from 0-9 with **no check that `d1*10+d2p` is not a held-out
number**. `assert_split_coverage` (265-281) guarantees *every* pos1 digit appears in the
held set, so for any train number's `d1` there is at least one held number `d1*10+d2p`
reachable. With `shuffle_pairs=False` the visual and audio composites at the same batch
index carry that same label (`lbl = cat([lab, lab_a])`, lab_a==lab), so the held-out number
is trained as a **labeled cross-modal SupCon positive pair**. Over 30 phase-(b) epochs a
large fraction of held numbers get direct cross-modal training.

This defeats the experiment's own premise (docstring 16-18: "if it merely memorizes seen
combinations, held-out numbers stay at chance"). A positive held-30 result can no longer be
attributed to compositional generalization vs. memorization. If any held-30 number from this
script is ever quoted as evidence of generalization, treat it as **critical**.

**Fix:** reject-sample `d2p` until `d1*10+d2p not in held_numbers`, or give near-miss items a
label outside the evaluated space; assert near-miss labels are disjoint from `held`.

---

## MEDIUM

### CMP-002 (research-integrity) — held-30 whole-number chance is 1%, not the documented 3.33%
Docstring line **40**: "Whole-number retrieval (chance 1.0% all-100 / 3.33% held-30)". But
`run_one_model` (895-916) builds prototypes over **all present numbers**
(`Pa, ua = number_protos(aZ, a_lab)`) and retrieves the held-mask queries against that
100-number pool. So `wholenum_v2a_held30` has chance **1/100 = 1%**, not 1/30. The verdict
uses `wn_held > 2.0` (line 1017) as if 3.33% were the baseline, and the empirical perm null
(~1%) will contradict the documented 3.33%. Either restrict the held pool to the 30 held
prototypes (then 3.33% is correct) or fix the doc/threshold.

### CMP-003 (research-integrity) — Visual "spike_rate"/"sparsity" is a post-ReLU activation rate
`_report_spiking_stats` (794-809) treats the stem output as summed spikes. But
`VisualMapStem.forward` (304-315) returns `F.relu(self.conv2(acc))` — `conv2` is a
**non-spiking real-valued refiner**, so the visual tokens are ReLU activations, not spikes.
`AudioMapStem` (333-346) returns genuine LIF spikes (`s3`). So the reported visual
`spike_rate`/`sparsity` is mislabeled (audio is fine). CLAUDE.md's "accuracy never travels
alone" requires a *real* spike-rate/sparsity + timestep cost; a mislabeled cost metric is an
integrity violation. Also note the stat is taken from a single hardcoded composite
`visual_2digit(1,2)` (n=1). Measure at the LIF output, or rename the column.

### CMP-004 (correctness) — Per-digit jitter is dead for the right digit, boundary-violating for the left
`visual_2digit` (206-215): `xr = max(40, min(40+jr, CANVAS_W-34=40)) == 40` for **every**
`jr in [-3,3]` → the right digit never jitters. The left digit uses
`xl = max(0, min(jl, 33))` → only non-negative jitter applies, and `xl` up to 33 pushes the
left digit into cols [33,67), across the buffer into the right region, contradicting the
"keep within cols [0,34)" comment. Position augmentation is much weaker than intended (which
*helps* memorization, compounding CMP-001).

### CMP-005 (correctness) — "shuffled" control is inconsistent across phases
Phase (a) (511-519) applies **one** permutation to a single label vector reused for both
modalities (cross-modal pairing preserved, label↔data broken). Phase (b) (547-556) shuffles
the **audio** labels with an *independent* permutation (`lab_a`) while visual keeps `lab`
(cross-modal pairing itself broken). The negative control means two different things across
the curriculum. Pick one semantics and apply it identically.

### CMP-006 (research-integrity) — Timestamped rerun CSVs double-count seeds in aggregation
`load_rows_from_disk` (953-966) globs `model_*_seed*.csv`, which matches both
`model_X_seedN.csv` and the `model_X_seedN__<ts>.csv` variant `_model_csv_path` (817-822)
writes when the base exists. `write_summary_and_report` (984-997) means over **all** rows
with no collapse to unique `(model, seed)`, so a rerun inflates `n_seeds` and corrupts
mean/std — the same near-duplicate inflation documented for `generate_figures_3seed` in
CLAUDE.md. Collapse to one row per `(model, seed)` before aggregating.

### CMP-007 (correctness) — `assert_split_coverage` can fail for arbitrary seeds, then is swallowed
`assert_split_coverage` (265-281) requires all ten digits in pos1 **and** pos2 within **both**
the 70-number train set and the 30-number held set. A random 30/70 split does not guarantee
that for every seed. On failure the assert raises inside `run_one_model`, and `main`'s broad
`try/except` (1195-1201) prints a traceback and drops that seed — a nominal 3-seed run can
silently become fewer. Resample the split until coverage holds, or restrict to verified seeds.

### CMP-008 (safety) — Module-level pickle-alias hack mutates `sys.modules["__main__"]` at import
Lines **87-95** inject `CachedNMNIST`/`CachedSHD`/`SubsampledByClass` into `__main__` so a
`torch.load(weights_only=False)` of a cache pickled under `__main__` resolves. This is an
import-time side effect (pollutes the `__main__` of whatever imports the file) duplicated
verbatim from `run_joint_diagnostic.py:67-74`, and the `torch.load` it enables executes
arbitrary pickle. It only helps when the cache was pickled while `run_joint_magnitude_ablation`
ran as `__main__`; no cache exists in-repo, so it currently guards a state that cannot occur
until the magnitude script is first run as main. **Fix:** rebuild caches under the importable
module (or use a custom `Unpickler` mapping) and delete the `__main__` mutation.

### CMP-009 (safety) — With no cache, running triggers a ~23 GB tonic render
`main` (1177-1186) → `_build_modality_loaders`, which renders N-MNIST + SHD from tonic when
the cache is absent. No `results/` cache exists here, yet the comment (87-91) claims it "loads
the existing 4.5 GB cache without a re-render". The `__main__` guard keeps *import* safe, but
an explicit run silently starts the full render/download. Fail fast when the cache is missing.

### CMP-013 (architecture) — Structural clone of the sibling / magnitude harness
`set_seed`, `_build_cfg`, `_mean`/`_std`, `_binom_z`, `load_rows_from_disk`,
`write_summary_and_report`, `_report_spiking_stats` and the main scaffolding are
near-duplicates of `run_joint_diagnostic.py` and `run_joint_magnitude_ablation.py` (~95 KB of
copy-pasted harness across the two audited files). Bug fixes (dedup, spike-metric, dead-config)
must be applied in several places. Extract a shared importable harness.

---

## LOW

- **CMP-010 (correctness/perf)** — Eval composites use TRAIN-set exemplars and the shared
  `builder` RNG (seed+7) is advanced by a *model-specific* amount of training, so the four
  models are evaluated on different exemplar draws (830, 848-849). No unseen-exemplar
  generalization is tested; cross-model comparison is not on identical inputs.
- **CMP-011 (dead-code)** — `within_set_visual`/`within_set_audio` are initialized NaN (859)
  and never set (`within_modality_gate` returns only `loo`, 789-791). Dead columns.
- **CMP-012 (quality)** — `d = slots.shape[2]` unused in `set_order_metrics` (683) and
  `_set_perm_null` (712); `whole_number_retrieval` docstring (733-737) describes a
  same-modality LOO path that never runs (only called cross-modally).
- **CMP-014 (quality)** — `_build_cfg` (153-165) sets ~30 fields but only device,
  `nmnist_train_per_class`, `batch_size` are consumed; `cfg.max_epochs/lr/patience` are
  ignored (training budget is the module constants; `lr=1e-3` is hardcoded at 490 and again as
  a literal in the sidecar 1208-1216). Editing cfg silently has no effect.
- **CMP-015 (research-integrity)** — Verdict thresholds `order>60`, `wn_held>2.0`, `asym<5`,
  `set>perm+1.0` (1016-1027) are magic numbers not in the pre-registration block; docstring
  chance values (37-42) and "Procrustes ~0%" (8) are analytic/unsourced.

## INFO

- **CMP-016 (safety)** — `subprocess.check_output(["git","rev-parse","HEAD"])` at 1204-1207;
  benign, `try/except`-wrapped.
- **CMP-017 (dependencies)** — Import chain is heavy (pulls `run_joint_magnitude_ablation` →
  `run_correspondence_anchor_v2` → provenance, plus `Dataloaders`, `tonic`, `snntorch`) but
  import-safe (all guarded, no import-time data load). `snntorch` is in `requirements.txt`.

---

## Cross-file notes
`_build_modality_loaders`, `nmnist_collate_fn`, `shd_collate_fn` come from
`run_joint_magnitude_ablation` (import-safe, guard at 1006). `CrossModalSupConLoss`,
`ExperimentConfig`, `get_device` come from `m7_trimodal_run_provenance` (guard at 1378).
`CrossModalSupConLoss` L2-normalizes internally, so feeding the concatenated 2-slot embedding
`Z=[z1||z2]` (norm √2) is fine. That loss includes the self/diagonal in its positive set (a
known deviation from canonical SupCon) but that is inherited from provenance, not introduced
here.

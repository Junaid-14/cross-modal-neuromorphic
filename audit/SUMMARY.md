# Audit Summary — `code/`

_Generated 2026-07-15 from `findings.jsonl` (331 findings; 316 canonical, 15 marked duplicate). Static analysis only — no code executed._

## Counts (canonical)

| Severity | n | | Category | n | | Status | n |
|---|---|---|---|---|---|---|---|
| critical | 10 | | research-integrity | 76 | | live | 272 |
| high | 69 | | correctness | 48 | | archive | 22 |
| medium | 112 | | dead-code | 43 | | dead | 22 |
| low | 105 | | dependencies | 30 | |  |  |
| info | 20 | | quality | 29 | |  |  |
|  |  | | safety | 24 | |  |  |
|  |  | | numerics | 23 | |  |  |
|  |  | | duplication | 17 | |  |  |
|  |  | | architecture | 13 | |  |  |
|  |  | | performance | 7 | |  |  |
|  |  | | naming | 3 | |  |  |
|  |  | | docs | 3 | |  |  |

## Cross-cutting themes

1. **Import-time execution is a repo-wide footgun.** `m7_trimodal_run.py` runs the whole 6-run
   experiment (and downloads/materialises ~23 GB, per project docs) on import with no `__main__` guard; several
   other scripts execute on import; `convert_mps.py` overwrites a source file on import; and
   `modify_notebook.py` would overwrite a (now-deleted) notebook in place. Read-only inspection
   of this codebase is genuinely dangerous.
2. **Two parallel generations, massively duplicated.** ~13.6k of ~29.9k lines are dead Era-1
   clones (four monoliths + smoke copies). The 7 core classes are defined 7x, `run_one_seed`
   exists 9x with 5 incompatible signatures, `ConvFrontend` 5x. `m7_trimodal_run.py` is an
   orphan third copy; the real hub is `m7_trimodal_run_provenance.py` (imported by ~19 files),
   and the two hubs have silently DIVERGED (extra projection layer, different fwt/CSV schema/seed count).
3. **The embedding-replay mechanism is broken three ways in BOTH live hubs** (task-ids used as
   class-ids; `label_offset=0` lets SHD overwrite N-MNIST in the buffer; replay routed through the
   current task's head). The `replay` vs `no_replay` comparison - the paper's core mechanism - is
   compromised, and `_task_labels` (the map that would fix it) is populated but never read.
4. **Research integrity is the single largest category.** Fabricated/hardcoded numbers are written
   to a results CSV (`run_rehearsal_transfer_eval_only.py`), to figures (`generate_figures.py`), and
   to stdout comparison tables (`run_cl_rehearsal.py`, `run_correspondence_anchor_v2.py`); statistics
   are mislabelled ("95% CI" without /sqrt(n); `fwt` = learning-gain; `forgetting_T1_T3` = incremental);
   and NO accuracy anywhere travels with spike-rate/sparsity/timestep cost. With no results CSV in the
   repo, every quoted number is unsourced (RULE 1).
5. **Metric definitions diverge and don't match the canonical loader.** `fwt` means different things
   in the two hubs; the docs' ~75pp "forward transfer" can only be produced by the mislabelled
   learning-gain formula.
6. **Held-out / control protocols are compromised across the experiment scripts.** Held-out masking is
   a no-op AND crashes out-of-bounds (correspondence), the Mantel test permutes the vectorised triangle
   (invalid p-value behind r=0.051/p=0.74), "shuffled-label" controls never shuffle, the compositional
   held-out set leaks into training, and one script reports SHD validation instead of test (RULE 4).
7. **The HGRN reset gate is inert** - `h_prev` is always zero, so the multiplicative reset `r*h_prev`
   (the paper's stated novelty and SpiNNaker risk R3) computes nothing and `W_r` never trains.
8. **RNG fragility invites silent numeric drift.** Training-loader shuffle draws from the global RNG,
   whose state depends on model construction and the `random_init_baseline` routine; no per-loader
   `generator=`. A benign-looking refactor (add a layer, change a dim) moves every trained number.
9. **Dependencies are broken.** `requirements.txt` is future-dated/likely unresolvable on py3.11 and
   omits >=5 imported packages; the HuggingFace-backup import sits under a bare `except`, so RULE-2
   backups silently no-op - the exact "lost data once" failure mode.
10. **Dead code & config theatre.** ~14 of 19 CONFIG keys are never read; dead backbone classifier
    layers compute-then-discard; stale `/Users/agent1/...` paths and a hardcoded `kill` PID; a live
    script imports a module that now lives only in `archive/`.


## Must-fix list — all CRITICAL and HIGH (canonical)

| id | sev | file:line | title | fix (short) |
|---|---|---|---|---|
| FIG-001 | CRITICAL | `code/generate_figures.py:82` | Both figure scripts write the identical 5 PNG filenames to the same dir - last run wins, can silently ship fabricated figures | Retire/delete generate_figures.py (superseded by generate_figures_3seed.py). If kept, write to a clearly-marked separate dir (e.g.… |
| FIG-002 | CRITICAL | `code/generate_figures.py:1` | No __main__ guard: importing generate_figures.py renders and writes all 5 fabricated figures (needs no CSV) | Delete the file (preferred). If retained, wrap all figure code in main() behind `if __name__=='__main__'` so a module import produ… |
| RUN-001 | CRITICAL | `code/m7_trimodal_run.py:699` | No __main__ guard: importing the module runs the full 6-run experiment and downloads/materializes datasets at import time | Move dataset loading, directory creation and the seed/condition loop into a main() guarded by 'if __name__ == "__main__": main()'.… |
| RUN-002 | CRITICAL | `code/m7_trimodal_run.py:380` | Replay sample() filters TASK ids against a CLASS-keyed store and draws <=1 embedding per class | Select classes via _task_labels (classes whose task id is requested) and sample multiple embeddings per class up to batch_size, in… |
| RUN-003 | CRITICAL | `code/m7_trimodal_run.py:361` | populate() uses label_offset=0 for every task, so SHD classes overwrite N-MNIST in the class-keyed store | Key the store so class collisions across tasks are impossible (e.g. by (task_id, class) or a per-task offset) and make sample() ta… |
| PROV-001 | CRITICAL | `code/m7_trimodal_run_provenance.py:446` | EmbeddingReplayBuffer replay is fundamentally broken: task_ids used as class keys, label_offset=0 collides tasks, wrong head/labels | Key the buffer per task so tasks cannot collide (offset labels by task, or nest by task_id); in sample() select stored classes via… |
| SNN-001 | CRITICAL | `code/m7_trimodal_run_provenance.py:804` | random_init_baseline silently reshuffles training data via the global RNG | Snapshot/restore RNG around the baseline (state=torch.get_rng_state() ... torch.set_rng_state(state)), or build baseline models wi… |
| ALGN-001 | CRITICAL | `code/run_correspondence_anchor_v2.py:311` | Held-out training path is doubly broken: CE reduced before mask, and 6 anchors vs labels 0-9 | Filter logits and labels by the keep-mask BEFORE calling ce (or set reduction='none' and average only the kept per-sample losses).… |
| ALGN-015 | CRITICAL | `code/run_ordinal_anchor.py:51` | Stage 1 inherits the broken held-out training from v2 | Fix train_correspondence per ALGN-001; then Stage 1 here runs. No separate change needed once the shared helper is corrected. |
| RA-001 | CRITICAL | `code/run_rehearsal_transfer_eval_only.py:104` | Hardcoded accuracies written into a results CSV | Delete the hardcoded accuracy/forgetting fields; write only columns computed live this run, or gate the whole CSV write behind a r… |
| ARCH-015 | HIGH | `code/archive/run_correspondence_anchor.py:412` | Held-out-class masking is a no-op: held-out digits are still trained on | To actually hold classes out: use reduction='none' to get per-sample losses BEFORE masking, or filter held-out samples out of the … |
| DEAD-018 | HIGH | `code/benchmark_mps_real.py:6` | benchmark_mps_real.py hardcodes another user's machine path and an MPS-only device | Delete. If ever revived, derive paths relative to the repo and auto-select the device MPS then CUDA then CPU. |
| DEAD-010 | HIGH | `code/convert_mps.py:7` | convert_mps.py overwrites code/m7_mps.py on import (no __main__ guard) | If retained, wrap the body in if __name__=='__main__': and write to a temp path / print a diff first. Preferred: delete the file. |
| CLE-008 | HIGH | `code/eval_transfer_fixed.py:60` | Missing config_nmnist.json crashes main() on the normal path | Guard the config path with a clear existence check and actionable error (which run/seed must be present first); do not assume the … |
| CLE-009 | HIGH | `code/eval_transfer_fixed.py:126` | Empty-results DataFrame raises KeyError before the 'in' guard can protect it | Handle the empty case before indexing: e.g. `if df.empty or "post_T2" not in df["stage"].values: post_t2_v2a = None` and branch th… |
| FIG-019 | HIGH | `code/final_evaluation.py:286` | Accuracy reported without spike rate / sparsity - violates 'accuracy never travels alone' | Report mean spikes/neuron (or spike rate) and sparsity for both sim and baseline alongside accuracy, and state the timestep count … |
| FIG-020 | HIGH | `code/final_evaluation.py:303` | 'SpiNNaker-deployed network' overclaim and unsourced Brian2-equivalence claim in the file's prose | Reword to 'discrete-time simulator of the mapped parameters' and drop 'deployed'. Either reproduce the Brian2 comparison and cite … |
| FIG-003 | HIGH | `code/generate_figures.py:30` | Hardcoded/fabricated accuracy values presented as M7 results (RULE 1) | Do not draw accuracy from literals. Delete this script (FIG-001/002); use generate_figures_3seed.py which sources accuracy from cl… |
| FIG-004 | HIGH | `code/generate_figures.py:36` | Hardcoded/fabricated forgetting values (RULE 1 / RULE 5) | Remove literal forgetting arrays; source from CSV via the corrected script. Avoid 'Catastrophic' in a title paired with hardcoded … |
| FIG-005 | HIGH | `code/generate_figures.py:39` | Fabricated 'simulated' gate trajectories drive the paper's central novelty claim | Delete; use generate_figures_3seed.py Figure 3, which renders from gate_trajectory.csv or skips loudly. |
| FIG-006 | HIGH | `code/generate_figures.py:78` | Mislabeled 'single-task' reference lines are legacy bi-modal numbers overlaid on a tri-modal plot | Remove the reference lines (as the corrected script did). If single-task baselines are wanted, run them and read from CSV; build t… |
| FIG-007 | HIGH | `code/generate_figures.py:157` | Hardcoded memory footprint [80,150,4] MB and invented 20x/37x compression | Delete; use generate_figures_3seed.py Figure 4 (derives bytes from CONFIG and prints the arithmetic, labeled 'derived from CONFIG,… |
| FIG-008 | HIGH | `code/generate_figures.py:1` | generate_figures.py is superseded and should be retired/deleted | Delete code/generate_figures.py (or move to code/archive/ with a FABRICATED marker and a non-colliding output path). Consolidate o… |
| RA-002 | HIGH | `code/generate_figures.py:30` | Hardcoded per-task accuracies and forgetting arrays drive Figures 1-2 | Delete the file (superseded by generate_figures_3seed.py); at minimum delete these literals and read from a CSV. |
| RA-003 | HIGH | `code/generate_figures.py:78` | Legacy bi-modal numbers mislabelled as 'single-task' reference lines | Delete both lines (already removed in the 3seed version); if single-task baselines are wanted, run them into the CSV. |
| RA-004 | HIGH | `code/generate_figures.py:39` | Gate trajectory 'simulated from log data' plotted as measured Figure 3 | Delete; the corrected path (generate_figures_3seed.py:233-278) reads a real gate_trajectory.csv or skips the figure. |
| DEAD-003 | HIGH | `code/m7_mps.py:1` | m7_mps.py is a dead machine-generated MPS clone that runs a training smoke-test on import | Delete - it is a regenerable derived artifact. Never import. If an MPS run is needed, use m7_trimodal_run_provenance.py. |
| DEP-004 | HIGH | `code/m7_mps.py:43` | Third-party 'seaborn' imported at module top level but absent from requirements.txt | Add seaborn (pip name: seaborn) to code/requirements.txt with a version that supports py3.11. |
| DEP-005 | HIGH | `code/m7_mps.py:70` | Third-party scikit-learn (import name sklearn) imported but absent from requirements.txt | Add scikit-learn (NOT sklearn) to code/requirements.txt with a py3.11-compatible version. |
| DEP-006 | HIGH | `code/m7_mps.py:2170` | 'huggingface_hub' absent from requirements.txt so RULE 2 backups silently skip | Add huggingface_hub to code/requirements.txt. Additionally narrow the except so a missing backup dependency is loud (do not swallo… |
| DEAD-008 | HIGH | `code/m7_nmnist_smoke_test.py:1` | m7_nmnist_smoke_test.py is a dead standalone smoke test that downloads N-MNIST and trains on import | Delete. Never import. |
| DEAD-002 | HIGH | `code/m7_raw.py:2` | m7_raw.py has uncommented !pip shell-magic lines - it is not valid Python (SyntaxError) | None if the file is deleted (recommended, see DEAD-001). If retained, every '!' line must be commented. Never try to import it to … |
| DEAD-007 | HIGH | `code/m7_smoke_test.py:1` | m7_smoke_test.py is a dead truncated copy of m7_trimodal_cl.py that trains on import | Delete. Never import. Fast validation is 'pytest tests/'. |
| DEAD-005 | HIGH | `code/m7_trimodal_cl.py:1` | m7_trimodal_cl.py is a dead notebook monolith that executes the ENTIRE experiment on import | Delete or move to code/archive/. Never import. Reproducibility entry point is m7_trimodal_run_provenance.py. |
| RUN-004 | HIGH | `code/m7_trimodal_run.py:453` | Replayed embeddings pass through the CURRENT task head, not the originating task head | Store the originating task per embedding (RUN-030) and route each replayed embedding through its own head; forward_from_embeddings… |
| RUN-005 | HIGH | `code/m7_trimodal_run.py:616` | 'fwt' is T2 learning gain (after minus zero-shot), mislabeled as forward transfer | Rename this field to learning_gain (or gain_T2); if a forward-transfer number is needed, define it against a from-scratch/random-i… |
| RUN-006 | HIGH | `code/m7_trimodal_run.py:614` | 'forgetting_T1_T3' is the incremental T2->T3 delta, not total T1 forgetting after T3 | Rename to forgetting_T1_incremental_T3, or redefine as acc_t1 - acc_t1_after_t3 to match the total-forgetting semantics of the oth… |
| RUN-007 | HIGH | `code/m7_trimodal_run.py:485` | Reported test accuracy uses last-epoch weights; the best-val checkpoint is saved but never reloaded | Either load best_*.pt before the final test eval and before saving final_*.pt, or drop the best-checkpoint machinery and state tha… |
| RUN-008 | HIGH | `code/m7_trimodal_run.py:488` | best_*.pt filename omits seed and condition, so all 6 runs overwrite the same 3 files | Include seed and condition (and ideally step) in the best_*.pt filename, matching the final_*.pt convention. |
| RUN-009 | HIGH | `code/m7_trimodal_run.py:537` | Silent checkpoint-skip loads stale final_*.pt with no provenance check and skips gate-trajectory logging | Record code/config version with each checkpoint and refuse or warn on mismatch; regenerate/reload the gate trajectory on skip so g… |
| RUN-010 | HIGH | `code/m7_trimodal_run.py:633` | Results CSV mixes two schemas and appends via read-concat-overwrite with no dedup | Write a single normalized schema (or one file per schema), key rows by (seed, condition, step), and upsert on that key instead of … |
| RUN-011 | HIGH | `code/m7_trimodal_run.py:411` | Accuracy is recorded and reported with no spike-rate/sparsity or timestep cost | Instrument the forward pass to accumulate mean spike rate/sparsity per layer and add spike-rate/sparsity and per-modality timestep… |
| RUN-012 | HIGH | `code/m7_trimodal_run.py:223` | HGRN reset gate is inert: h_prev is always zero, so r*h_prev is 0 and W_r/norm_r get no gradient | Decide what the HGRN should do: if recurrence is intended, thread a real hidden state; if not, remove the reset gate and its weigh… |
| RUN-013 | HIGH | `code/m7_trimodal_run.py:273` | Near-zero forgetting is structural: only hopfield/hgrn/gate are shared across tasks; backbones and heads are grad-isolated | No code change required for correctness, but reporting must state that forgetting is measured only through the shared hopfield/hgr… |
| SNN-006 | HIGH | `code/m7_trimodal_run.py:139` | SHD runs 100 timesteps while N-MNIST/DVS run 25; time_steps=25 misrepresents SHD; ~4x feature-magnitude asymmetry into the shared gate | Either bin SHD to 25 (respect time_steps) or record the true per-modality T and report it; if the asymmetry is intentional, state … |
| SNN-007 | HIGH | `code/m7_trimodal_run.py:127` | Training DataLoaders shuffle from the global RNG (no per-loader generator=) | Pass generator=torch.Generator().manual_seed(seed) to each training DataLoader so shuffling is decoupled from init RNG (mirrors ho… |
| PROV-002 | HIGH | `code/m7_trimodal_run_provenance.py:362` | Default forward path silently routes features through proj_shared/proj_private (512->128+384->512), diverging from the documented architecture and from the orphan copy | Only apply the shared/private projection when cfg.align_cross_modal (or an explicit flag) is set; otherwise feed raw backbone feat… |
| PROV-003 | HIGH | `code/m7_trimodal_run_provenance.py:409` | Replay path (forward_from_embeddings) bypasses _split_shared_private, so replayed embeddings hit a different subgraph than live inputs | Route replayed embeddings through the same _split_shared_private transform used for live inputs before gate/hopfield/hgrn, OR stor… |
| PROV-004 | HIGH | `code/m7_trimodal_run_provenance.py:662` | Best-validation weights are never restored: reported best_val, saved checkpoint, and test_acc all use the final (post-early-stopping) epoch | Snapshot state_dict when val improves (copy.deepcopy) and reload it before computing test_acc and before saving the checkpoint, so… |
| PROV-005 | HIGH | `code/m7_trimodal_run_provenance.py:1132` | forgetting_T1_T3 uses the after-T2 accuracy as its baseline, inconsistent with the other two forgetting metrics (mislabeled) | Decide one definition. For cumulative forgetting after T3 use acc_t1_after - acc_t1_after_t3; if the incremental T2->T3 drop is ge… |
| PROV-006 | HIGH | `code/m7_trimodal_run_provenance.py:849` | Results CSV is appended without dedup and aggregated without dedup; schema also diverges from the orphan and from the canonical loader | Deduplicate to one row per (seed, condition) before aggregating (reuse .claude/skills/analyze-cl-results/scripts/load_results.py).… |
| PROV-007 | HIGH | `code/m7_trimodal_run_provenance.py:1` | Two divergent copies of the model+trainer share class/function names and output filenames but differ in model math, metrics, CSV schema, freezing, and import-safety | Collapse to a single source of truth: keep the provenance hub, delete or clearly quarantine the orphan (it is imported by nobody),… |
| PROV-008 | HIGH | `code/m7_trimodal_run_provenance.py:1064` | 'fwt' means different things in the two hubs; the orphan's formula is learning-gain mislabeled as forward transfer, and only that mislabel can produce a large positive FWT | Rename the orphan's 'fwt' to 'learning_gain' (it is not forward transfer). Keep this file's FWT definition, and when any FWT numbe… |
| SNN-003 | HIGH | `code/m7_trimodal_run_provenance.py:883` | Results CSV append-without-dedup; provenance aggregator never dedups | Route all analysis through load_results.py (aggregates over unique seeds). In aggregate_results add df=df[df.get('step').isna()].d… |
| SNN-004 | HIGH | `code/m7_trimodal_run_provenance.py:992` | Checkpoint-skip has no staleness/config guard; provenance filename is config-blind | Embed {config_hash, git_sha, code_version} in the checkpoint; on load verify it matches and retrain on mismatch; put a config hash… |
| SNN-008 | HIGH | `code/m7_trimodal_run_provenance.py:41` | Provenance config JSON records max_epochs=50/patience=12 but the run hardcodes 15/5 | Pass cfg.max_epochs/cfg.patience into train_task (or set them to 15/5 so the JSON is truthful); replace the hardcoded 1e-4 in m7 w… |
| DEAD-012 | HIGH | `code/modify_notebook.py:8` | modify_notebook.py crashes immediately - its target notebook no longer exists | Move to code/archive/ (retains cell-structure provenance) or delete. Never import (see DEAD-013). |
| DEAD-013 | HIGH | `code/modify_notebook.py:8` | modify_notebook.py reads and overwrites a notebook in place on import, with no backup | Delete/archive. If ever adapted, write to a NEW path and never overwrite the source in place. |
| DEAD-016 | HIGH | `code/monitor_and_stop.py:12` | monitor_and_stop.py can send kill -15/-9 to an unrelated process (hardcoded PID 16672) | Delete. Never signal a process by a hardcoded literal PID. |
| CLE-013 | HIGH | `code/phase1_sanity_check.py:100` | Missing config_nmnist.json crashes main() (points at results/cl_confirmation) | Guard the config and each checkpoint path with existence checks and clear errors; note the directory name differs from eval_transf… |
| DEP-001 | HIGH | `code/requirements.txt:1` | Future-dated version pins likely unresolvable on PyPI for CPython 3.11 | For each pin, do a resolve dry-run for cp311 (pip index versions <pkg> or pip install --dry-run) and replace any nonexistent versi… |
| DEP-002 | HIGH | `code/requirements.txt:7` | numpy==1.26.4 is inconsistent with the rest of the stack and likely blocks resolution | Cross-check numpy against the numpy floors declared by the chosen scipy/pandas/torch versions; most likely bump numpy to a 2.x tha… |
| RA-005 | HIGH | `code/run_cl_rehearsal.py:435` | 3 of 4 columns in the printed COMPARISON TABLE are hardcoded literals | Load the comparison conditions from a CSV, or print only the live column and label the rest explicitly as 'prior-run literals, uns… |
| CMP-001 | HIGH | `code/run_compositional_crossmodal.py:538` | Near-miss negatives leak held-out numbers into training | Reject-sample d2p so that d1*10+d2p is not in held_numbers (or give near-miss items a label outside the evaluated number space), a… |
| ALGN-002 | HIGH | `code/run_correspondence_anchor_v2.py:50` | CKPT_SOURCE points to a checkpoint that does not exist (no results/ tree) | Document the required upstream artifact and guard with a clear error, or ship a small fixture; ultimately depends on repairing the… |
| ALGN-003 | HIGH | `code/run_correspondence_anchor_v2.py:147` | _load_cached_features reads a feature cache that does not exist | Provide a first-class cache-builder in a live module (the extractor already exists as dead code, see ALGN-006) instead of pointing… |
| ALGN-004 | HIGH | `code/run_correspondence_anchor_v2.py:716` | Hardcoded baseline numbers printed as a comparison table (RULE 1) | Remove the hardcoded rows or load them from an actual baseline CSV; do not print unsourced numbers next to measured ones. |
| ALGN-021 | HIGH | `code/run_cross_modal_alignment_study.py:211` | Checkpoint-skip staleness trap on T1/T2/T3 (silent stale results) | Stamp checkpoints with a code/version hash and refuse or warn loudly on mismatch; at minimum log the checkpoint mtime and a hash. |
| ALGN-027 | HIGH | `code/run_digit_only_alignment.py:615` | --confirm does 'import run_cl_confirmation' but that module lives only in code/archive/ | Call the local run_full_cl_confirmation(base_cfg) instead of importing the archived module, or restore/relocate a live run_cl_conf… |
| ALGN-028 | HIGH | `code/run_digit_only_alignment.py:318` | SHD accuracy reported and Wilcoxon-tested on VALIDATION, never test (RULE 4) | Add trainer.evaluate(data_loaders[task]['test'], ...) after training and report/aggregate/test on the test accuracy; keep val only… |
| JNT-001 | HIGH | `code/run_joint_magnitude_ablation.py:809` | Shuffled-label control does not shuffle labels (mislabeled null) | Add a shuffled_labels flag to train_model that permutes the labels feeding CrossModalSupCon (and per-modality CE), mirroring run_c… |
| JNT-002 | HIGH | `code/run_joint_magnitude_ablation.py:11` | Docstring 'held-out 6-9 never receive a cross-modal signal' is false; held-out set is confounded | Rewrite the docstring to state 6-9 DO receive (a) a shared magnitude signal in ON and (b) shared-core co-shaping via within-modali… |
| JNT-005 | HIGH | `code/run_joint_magnitude_ablation.py:85` | CachedNMNIST/CachedSHD pickled as __main__ couples the cache to run-as-main and forces alias hacks downstream | Move CachedNMNIST/CachedSHD/SubsampledByClass into an importable module (e.g. code/joint_datasets.py) so the pickle path is stable… |
| JNT-016 | HIGH | `code/run_joint_shared_encoder.py:989` | Hardcoded baseline numbers written into the report as if measured (RULE 1 violation) | Read baseline numbers from their source CSV at runtime or omit them entirely; never embed plus/minus-formatted result literals in … |
| ALGN-016 | HIGH | `code/run_ordinal_anchor.py:147` | Invalid Mantel permutation null (permutes vectorized triangle, not objects) | Permute the n=10 objects: draw a permutation of indices, apply it symmetrically to one RDM (rdm[np.ix_(perm,perm)]), then re-extra… |
| ALGN-017 | HIGH | `code/run_ordinal_anchor.py:63` | CKPT_SOURCE and feature cache missing (script un-runnable) | Same as ALGN-002/ALGN-003: repair the upstream producer and/or ship fixtures; correct the 'from v2' comment (cache is a v1 artifac… |
| CLE-001 | HIGH | `code/run_rehearsal_transfer_eval_only.py:1` | No __main__ guard: entire script runs at import time | Wrap lines 37-127 in `def main():` and gate with `if __name__ == "__main__": main()`, mirroring eval_transfer_fixed.py. |
| CLE-004 | HIGH | `code/run_rehearsal_transfer_eval_only.py:43` | torch.load of a non-existent checkpoint crashes the module on import | After adding the main guard (CLE-001), check `ckpt_t3.exists()` and exit with a clear message if absent; validate the loaded objec… |
| TST-016 | HIGH | `code/test_ordinal_anchor_smoke.py:14` | ImportError on import: _build_digit_loaders is not exported by run_ordinal_anchor | Import _build_digit_loaders from run_correspondence_anchor_v2 (where it lives), or add it to run_ordinal_anchor's re-export block … |

## How to consume
See `README.md` for the schema. The machine-readable spine is `findings.jsonl` (filter `duplicate_of == null` for the canonical set). Per-file detail in `by-file/`, concern views in `by-concern/`, file inventory in `MANIFEST.md`.

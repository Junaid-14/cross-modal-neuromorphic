# Code quality, dead code, performance, naming, docs

_Generated 2026-07-15. 85 canonical findings in categories: dead-code, docs, naming, performance, quality. Duplicates omitted (see `findings.jsonl`)._

Severity: HIGH 5, MED 17, LOW 59, INFO 4


## HIGH (5)

#### `DEAD-003` [HIGH] m7_mps.py is a dead machine-generated MPS clone that runs a training smoke-test on import
`code/m7_mps.py:1-3900` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** MPS port of m7_raw.py produced by convert_mps.py; re-defines the same 7 core classes plus legacy zoo and ablations (class defs L633-2944), superseded by code/m7_trimodal_run_provenance.py (which already auto-selects MPS). No __main__ guard: at module scope (L560-576) it seeds, builds M7_ContinualAdaptiveModel and ContinualLearner, and calls trainer.train_task(...), preceded by tonic dataset loading - so 'import m7_mps' downloads datasets and starts training (the same import-time hazard as the live m7_trimodal_run.py trap).
- **Evidence:** module-scope execution L560-576: torch.manual_seed L560-561, model_smoke=M7_ContinualAdaptiveModel(...) L569, trainer=ContinualLearner(...) L572, result_smoke=trainer.train_task(...) L576; grep '__main__' returns 0.
- **Fix:** Delete - it is a regenerable derived artifact. Never import. If an MPS run is needed, use m7_trimodal_run_provenance.py.
- **Fix at:** `code/m7_mps.py:1-3900`
- _related: DEAD-001, DEAD-004, DEAD-009_

#### `DEAD-008` [HIGH] m7_nmnist_smoke_test.py is a dead standalone smoke test that downloads N-MNIST and trains on import
`code/m7_nmnist_smoke_test.py:1-472` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** Hand-written (not a notebook dump) single-file N-MNIST smoke test that re-defines the core classes: SupervisedContrastiveLoss L121, ModernHopfieldLayer L138, ImprovedHGRNGate L152, SNN_Backbone L174, M7_ContinualAdaptiveModel L222 (audio backbone skipped, L228-229), EmbeddingReplayBuffer L267, ContinualLearner L318. No __main__ guard: at module scope it constructs tonic.datasets.NMNIST (L95-96, which downloads if absent) and runs 10-epoch training (L424-435), so 'import m7_nmnist_smoke_test' downloads and trains. Superseded by m7_trimodal_run_provenance.py. Minor extra issues (all secondary to retirement): the QUICK_SUBSET torch.randperm at L100 runs before torch.manual_seed(42) at L419 (non-deterministic subset), and results are printed without spike-rate/sparsity (accuracy-never-alone) - noted, not separately filed.
- **Evidence:** dataset ctor L95-96; QUICK_SUBSET randperm L99-100; trainer.train_task L430-435; grep '__main__' returns 0.
- **Fix:** Delete. Never import.
- **Fix at:** `code/m7_nmnist_smoke_test.py:1-472`
- _related: DEAD-007_

#### `DEAD-007` [HIGH] m7_smoke_test.py is a dead truncated copy of m7_trimodal_cl.py that trains on import
`code/m7_smoke_test.py:1-2070` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** Near-identical prefix clone of m7_trimodal_cl.py - the 7 core classes and legacy zoo sit at the SAME line numbers (SupervisedContrastiveLoss L560, ModernHopfieldLayer L638, ImprovedHGRNGate L720, SNN_Backbone L814, ... ContinualLearner L1815) - truncated after ContinualLearner (no EWC/ablation/energy sections). No __main__ guard: at module scope (L2003-2021) it seeds and calls trainer.train_task(...) after dataset loading, so importing it downloads datasets and trains. Superseded by m7_trimodal_run_provenance.py.
- **Evidence:** class line numbers identical to m7_trimodal_cl.py (560/638/720/814/956/.../1815); module-scope train L2003-2021 (model_smoke L2014, trainer L2017, result_smoke=trainer.train_task L2021); grep '__main__' returns 0.
- **Fix:** Delete. Never import. Fast validation is 'pytest tests/'.
- **Fix at:** `code/m7_smoke_test.py:1-2070`
- _related: DEAD-005_

#### `DEAD-005` [HIGH] m7_trimodal_cl.py is a dead notebook monolith that executes the ENTIRE experiment on import
`code/m7_trimodal_cl.py:1-3759` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** Full Era-1 tri-modal CL notebook dump; re-defines the 7 core classes plus legacy zoo and ablations (class defs L560-2871), superseded by code/m7_trimodal_run_provenance.py. No __main__ guard with heavy module-scope execution running to L3756: dataset loading, the CL training loop, SynOps/energy analysis writing energy_results.json (L3329-3361), building paper_tables.tex (L3568-3577) and publication figures (L3585+). 'import m7_trimodal_cl' therefore downloads datasets, trains for hours, and writes result/figure files.
- **Evidence:** module-scope statements to L3756: open(RESULTS_DIR/'energy_results.json','w') L3359, open(out_path=paper_tables.tex,'w') L3569, savefig fig_energy.pdf near L3745; grep '__main__' returns 0.
- **Fix:** Delete or move to code/archive/. Never import. Reproducibility entry point is m7_trimodal_run_provenance.py.
- **Fix at:** `code/m7_trimodal_cl.py:1-3759`
- _related: DEAD-006, DEAD-007_

#### `DEAD-012` [HIGH] modify_notebook.py crashes immediately - its target notebook no longer exists
`code/modify_notebook.py:8-9` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** The script opens 'NeurIPS2026_M7_clean (3) (1).ipynb' (L8-9) to patch it, but that notebook was deleted from the repo (git commit 837d7f9 'Remove original-M7 experiment artifacts ... notebooks'; no .ipynb exists anywhere in the tree). Any run raises FileNotFoundError at L9 before doing anything. It is both dead (the notebook workflow is retired) and non-functional. Per CLAUDE.md it is the ONLY surviving record of the old notebook's cell structure, so archiving is preferable to hard deletion if that provenance matters.
- **Evidence:** open(SCRIPT_DIR/'NeurIPS2026_M7_clean (3) (1).ipynb') at L8; 'find . -name *.ipynb' returns nothing; git log shows the notebook removed (837d7f9).
- **Fix:** Move to code/archive/ (retains cell-structure provenance) or delete. Never import (see DEAD-013).
- **Fix at:** `code/modify_notebook.py:1-569`
- _related: DEAD-013, DEAD-014_


## MED (17)

#### `ARCH-001` [MED] Superseded v1 of run_correspondence_anchor_v2.py
`code/archive/run_correspondence_anchor.py:1-767` · dead-code · status:archive · effort:trivial · confidence:high
- **What:** This is the v1 correspondence-anchor experiment. The live successor code/run_correspondence_anchor_v2.py (34 KB) exists and is the current version. v1 is import-safe (main() guarded, line 766) but dead: imported by nobody, and it also carries a real held-out-class bug (ARCH-015) that v2 presumably fixes. It cannot run today anyway because CKPT_SOURCE is missing (ARCH-005).
- **Evidence:** code/run_correspondence_anchor_v2.py present and live; docstring/design here matches the v2 lineage; no importer of this module.
- **Fix:** Retirement verdict DELETE: v2 is the record of this experiment. If retained for provenance, add a header comment marking it superseded and noting the held-out masking no-op (ARCH-015) so its printed held-out numbers are not trusted.
- **Fix at:** `code/archive/run_correspondence_anchor.py:1-767`
- _related: ARCH-005, ARCH-014, ARCH-015_

#### `DEAD-017` [MED] benchmark_mps_real.py is a dead one-off benchmark that imports the legacy bi-modal package
`code/benchmark_mps_real.py:1-54` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** One-off throughput benchmark. It imports the LEGACY package - Models.registry.build_model (L9) and Dataloaders.nmnist_loader.get_nmnist_loaders (L11) - not the M7 code, and runs the benchmark at module scope (L19-54) with no __main__ guard, so importing it would try to load N-MNIST and time training. Not part of the M7 pipeline and out of scope of the maintained code.
- **Evidence:** from Models.registry import build_model L9; from Dataloaders.nmnist_loader import get_nmnist_loaders L11; module-scope benchmark loop L19-54; grep '__main__' returns 0.
- **Fix:** Delete.
- **Fix at:** `code/benchmark_mps_real.py:1-54`
- _related: DEAD-018_

#### `DEAD-009` [MED] convert_mps.py is a dead one-off codegen whose only output (m7_mps.py) is also dead
`code/convert_mps.py:1-182` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** One-off script that string-transforms m7_raw.py into m7_mps.py. Both its input (m7_raw.py) and output (m7_mps.py) are retired notebook monoliths superseded by code/m7_trimodal_run_provenance.py, which already auto-selects MPS then CUDA then CPU. The tool has no remaining purpose.
- **Evidence:** reads m7_raw.py L7-8; writes m7_mps.py L173-174.
- **Fix:** Delete alongside m7_raw.py and m7_mps.py.
- **Fix at:** `code/convert_mps.py:1-182`
- _related: DEAD-010, DEAD-011_

#### `DEAD-001` [MED] m7_raw.py is a dead Kaggle/Jupyter notebook dump superseded by m7_trimodal_run_provenance.py
`code/m7_raw.py:1-3877` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** Raw export of the Era-1 notebook (retains '# %% CELL N' cell markers). Re-defines the 7 core M7 classes as copy-paste clones: SupervisedContrastiveLoss (L614), ModernHopfieldLayer (L692), ImprovedHGRNGate (L774), SNN_Backbone (L868), M7_ContinualAdaptiveModel (L1615), EmbeddingReplayBuffer (L1761), ContinualLearner (L1867); plus the entire legacy bi-modal model zoo (Model_1..5_SHD L1068-1158, Model_1..5 baselines L1158-1275) and ablation classes (EWC L2502, M7_NoSCL/M7_FixedGate/M7_SharedHead L2893-2921). The maintained implementation of the same classes is code/m7_trimodal_run_provenance.py (guarded, import-safe, L151-527). Imported by nobody.
- **Evidence:** class defs at listed lines; same class names present in code/m7_trimodal_run_provenance.py L151-527; no other module imports m7_raw.
- **Fix:** Delete, or move to code/archive/ if a provenance snapshot is wanted. Do not import (also blocked by DEAD-002 SyntaxError).
- **Fix at:** `code/m7_raw.py:1-3877`
- _related: DEAD-002, DEAD-003, DEAD-004_

#### `RUN-017` [MED] Entire SHD train+test set is densified into a Python list of [100,1,1,700] tensors at import
`code/m7_trimodal_run.py:149-150` · performance · status:live · effort:medium · confidence:high
- **What:** shd_train_data and shd_test_data are built by list comprehensions that convert every SHD sample to a dense [100,700] float tensor (unsqueezed to [100,1,1,700]) and hold them all in memory before training. events_to_dense also fills the dense tensor with a per-event Python for-loop (145-147) rather than vectorized indexing. This is eager full-dataset materialization at import (compounding RUN-001) and is both memory-heavy and slow.
- **Evidence:** lines 149-150 'shd_train_data = [events_to_dense(e, l) for e, l in tqdm(shd_train...)]' and same for shd_test; per-event loop 145-147
- **Fix:** Densify lazily inside a Dataset/transform (as N-MNIST and DVS do via tonic) or cache to disk, and vectorize the t/x index assignment instead of the Python loop.
- **Fix at:** `code/m7_trimodal_run.py:138-160`
- _related: RUN-001, RUN-033_

#### `RUN-021` [MED] 14 of 19 CONFIG keys are never read, implying configurability the code does not have
`code/m7_trimodal_run.py:45-65` · dead-code · status:live · effort:small · confidence:high
- **What:** Only batch_size, gradient_clip, beta, num_patterns and contrastive_temperature are read. The other keys - learning_rate, weight_decay, max_epochs, patience, dropout, hidden_dim, num_gru_layers, num_workers, time_steps, pin_memory, use_contrastive, contrastive_weight, seed, device - are never consumed; their behaviors are hardcoded elsewhere (RUN-014, RUN-020) or absent (dropout, num_gru_layers). This is misleading: it looks tunable but most knobs do nothing.
- **Evidence:** CONFIG dict 45-65; read keys at 46->127, 51->457, 52->238, 55->293, 61->427; the remaining keys have no read site in the file
- **Fix:** Delete the unused keys or wire them into the code they claim to control; keep CONFIG as the single source of truth for anything it lists.
- **Fix at:** `code/m7_trimodal_run.py:45-65`
- _related: RUN-014, RUN-020_

#### `RUN-022` [MED] Backbone fc3/lif5 (visual) and fc4/lif4 (audio) are dead: their outputs are always discarded
`code/m7_trimodal_run.py:246-247` · dead-code · status:live · effort:small · confidence:high
- **What:** SNN_Backbone.fc3/lif5 produce spk5, but both callers take only the second return ('_, features = bb(x)' at 305 and 324), so spk5 is computed for all 25 timesteps every forward and thrown away. In the audio backbone, fc4/lif4 are never referenced by the audio unroll (which stops at fc3/lif3), so they are pure dead parameters. These are vestigial standalone classifier heads left from before per-task self.heads took over.
- **Evidence:** line 246 'self.fc3 = nn.Linear(512, num_classes)'; 268-270 compute f3/s5/spk5_rec; consumers '_, features = bb(x)' at 305 and 324; audio fc4/lif4 288-289 unused by 312-318/331-336
- **Fix:** Remove fc3/lif5 and the spk5 accumulation from SNN_Backbone (and fc4/lif4 from the audio backbone), or if kept for standalone use document that M7 does not use them; this also trims per-timestep compute.
- **Fix at:** `code/m7_trimodal_run.py:246-247`, `code/m7_trimodal_run.py:268-271`, `code/m7_trimodal_run.py:288-289`
- _related: RUN-023_

#### `PROV-015` [MED] Many ExperimentConfig fields are never consumed; some advertise values the run silently ignores (max_epochs, patience, dropout)
`code/m7_trimodal_run_provenance.py:33-74` · dead-code · status:live · effort:medium · confidence:high
- **What:** Grep confirms zero cfg.-qualified reads for optimizer (37), max_epochs (41), patience (42), dropout (45), hidden_dim (46), num_gru_layers (50), num_workers (51), pin_memory (53), time_steps (52); and align_joint_training (66), align_mixed_batches (67), nmnist_train_per_class (74) are referenced nowhere. Several are actively misleading: max_epochs=50 and patience=12 are ignored because train_task is called with num_epochs=15, patience=5 hardcoded (999,1048,1100); dropout=0.2 is advertised but the model contains no dropout layer; num_workers is hardcoded to 0 in every DataLoader; time_steps=25 is hardcoded in the collate fns (see PROV-018); optimizer='AdamW' is hardcoded (580). A reader trusting the config for reproducibility gets the wrong epoch/patience/dropout.
- **Evidence:** cfg.max_epochs/patience/dropout/hidden_dim/num_gru_layers/num_workers/pin_memory/time_steps/optimizer: 0 cfg-qualified references (grep); train_task called num_epochs=15 patience=5 (999,1048,1100); AdamW hardcoded (580); num_workers=0 literal (e.g. 1187-1193).
- **Fix:** Either wire these fields into the code (pass cfg.max_epochs/cfg.patience/cfg.num_workers/cfg.time_steps/cfg.optimizer through, add the dropout layer) or delete them from ExperimentConfig so the config cannot misrepresent the run.
- **Fix at:** `code/m7_trimodal_run_provenance.py:33-74`, `code/m7_trimodal_run_provenance.py:999`, `code/m7_trimodal_run_provenance.py:1048`, `code/m7_trimodal_run_provenance.py:1100`
- _related: PROV-018_

#### `PROV-017` [MED] Entire SHD train and test sets are densified into Python lists in memory before any training
`code/m7_trimodal_run_provenance.py:1212-1213` · performance · status:live · effort:medium · confidence:medium
- **What:** build_data_loaders materialises every SHD sample into a dense (100,1,1,700) float tensor via list comprehensions over the full datasets (1212-1213) before constructing loaders. Each dense sample is 100*700=70000 floats (~0.27 MB); across the full SHD train+test this is on the order of gigabytes held resident, which is wasteful on MPS/limited-RAM machines and duplicates the same eager pattern as the orphan (m7_trimodal_run.py:149-150). It also runs at build time for every invocation regardless of whether SHD is the current task.
- **Evidence:** shd_train_data = [_events_to_dense(e,l) for e,l in tqdm(shd_train,...)] (1212); shd_test_data likewise (1213); dense zeros(100,700) per sample (1201-1202).
- **Fix:** Wrap SHD in a lazy Dataset that densifies per-item in __getitem__ (or a DiskCachedDataset like N-MNIST) instead of building full in-memory lists, so memory scales with batch size, not dataset size.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1198-1227`

#### `PROV-022` [MED] Output/data paths are CWD-relative ('./ICONS_M7'), so a hub used by 19 callers writes wherever each caller's CWD happens to be
`code/m7_trimodal_run_provenance.py:1285-1290` · quality · status:live · effort:small · confidence:high
- **What:** main() sets base_dir=Path('./ICONS_M7') (1285) and build_data_loaders uses Path('./ICONS_M7/datasets') (1169); both are relative to the current working directory, resolving as documented only when CWD is code/ (CLAUDE.md, python-numerics rule 6). Because this module is imported by 19 files that may run from different directories, results, checkpoints, and datasets can silently land in, or be read from, the wrong place — including the checkpoint-skip trap loading a stale final_*.pt from an unexpected directory.
- **Evidence:** base_dir = Path('./ICONS_M7') (1285); data_dir = Path('./ICONS_M7/datasets') (1169); checkpoints under results_dir (1287).
- **Fix:** Anchor paths to a stable base (e.g. Path(__file__).resolve().parent or an explicit argument/env var) rather than CWD, or require callers to pass an absolute results/ckpt dir (run_one_seed already accepts results_dir/ckpt_dir — main and build_data_loaders should too).
- **Fix at:** `code/m7_trimodal_run_provenance.py:1168-1170`, `code/m7_trimodal_run_provenance.py:1284-1290`

#### `DEAD-015` [MED] monitor_and_stop.py is a dead one-off log-watcher tied to a machine that no longer exists
`code/monitor_and_stop.py:1-61` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** A throwaway watcher that tailed one training log and killed one PID during an Era-1 run. It references a foreign absolute log path (/Users/agent1/...) and a fixed PID that have no meaning in this repo, and there is no train.log here. No reusable value.
- **Evidence:** LOG_FILE=/Users/agent1/.../train.log L9; PID=16672 L12; regex parses 'Val: NN.N%' L25.
- **Fix:** Delete.
- **Fix at:** `code/monitor_and_stop.py:1-61`
- _related: DEAD-016_

#### `ALGN-006` [MED] _extract_backbone_features defined but never called (dead clone of archived v1)
`code/run_correspondence_anchor_v2.py:120-144` · dead-code · status:live · effort:trivial · confidence:high
- **What:** _extract_backbone_features is a module-level function that is never invoked anywhere in the file (feature extraction goes through _load_cached_features and the model's own _get_backbone_features method). It is a verbatim clone of code/archive/run_correspondence_anchor.py:139. Dead code that also re-implements the audio backbone forward (drift risk, see ALGN-007).
- **Evidence:** grep shows the only occurrence of _extract_backbone_features in the file is its def at line 120; clone at code/archive/run_correspondence_anchor.py:139
- **Fix:** Delete the dead function; if a cache-builder is needed, expose one live helper and call the model's feature method.
- **Fix at:** `code/run_correspondence_anchor_v2.py:120-144`
- _related: ALGN-003, ALGN-007, ALGN-008_

#### `ALGN-009` [MED] run_one_seed and compute_held_out_metrics collide with other definitions (3 signatures each)
`code/run_correspondence_anchor_v2.py:524-525` · naming · status:live · effort:small · confidence:high
- **What:** This file defines module-level run_one_seed(seed,cfg,loaders,cached,device,head_type) at line 524, while m7_trimodal_run_provenance.py:962 defines run_one_seed(cfg,data_loaders,results_dir,ckpt_dir) and run_ordinal_anchor.py:395 defines yet another run_one_seed. Likewise compute_held_out_metrics has three incompatible signatures across the fleet (ALGN-008). Any future 'from run_correspondence_anchor_v2 import run_one_seed' would silently bind the wrong function; run_digit_only_alignment already imports the provenance one. Same-name/different-contract functions are an error waiting to happen.
- **Evidence:** run_correspondence_anchor_v2.py:524 def run_one_seed(seed, cfg, loaders, cached, device, head_type); m7_trimodal_run_provenance.py:962 def run_one_seed(cfg, data_loaders, results_dir, ckpt_dir); run_ordinal_anchor.py:395 def run_one_seed(seed, cfg, loaders, cached, device, head_type)
- **Fix:** Rename the per-script runners (e.g. run_one_seed_correspondence) or namespace them; keep one canonical run_one_seed.
- **Fix at:** `code/run_correspondence_anchor_v2.py:524`, `code/run_ordinal_anchor.py:395`, `code/m7_trimodal_run_provenance.py:962`
- _related: ALGN-008_

#### `ALGN-024` [MED] random_init_baseline recomputed for every condition and seed
`code/run_cross_modal_alignment_study.py:201-206` · performance · status:live · effort:small · confidence:high
- **What:** _run_alignment_seed computes random_init_baseline(..., n_seeds=3) for all three tasks at the start of every condition-seed (201-206). The RI baseline is condition-independent, so with 3 conditions x 5 seeds it re-instantiates and evaluates 3 fresh random models x 3 tasks x 15 = 135 baseline evaluations where 3 (one per task, once) would do. On MPS with DVS-Gesture this is a large redundant cost.
- **Evidence:** lines 204-206 for task_name in cfg.task_order: ri = random_init_baseline(TASKS, task_name, data_loaders[task_name]['test'], cfg, device, n_seeds=3); called inside per-condition-per-seed _run_alignment_seed
- **Fix:** Compute the RI baselines once in main() and pass them in; they do not depend on condition or the trained model.
- **Fix at:** `code/run_cross_modal_alignment_study.py:201-206`, `code/run_cross_modal_alignment_study.py:466-501`

#### `ALGN-029` [MED] run_full_cl_confirmation is orphaned yet is the producer of the cluster's required checkpoint
`code/run_digit_only_alignment.py:461-505` · dead-code · status:live · effort:small · confidence:high
- **What:** run_full_cl_confirmation (461-505) is documented 'NOT called by main() - invoke explicitly' and is never invoked anywhere (the --confirm flag imports the archived module instead, ALGN-027). It writes results/cl_confirmation/ checkpoints via provenance run_one_seed - which is exactly the CKPT_SOURCE (results/cl_confirmation/checkpoints/A_no_align_seed42/final_shd_shd.pt) that run_correspondence_anchor_v2 and run_ordinal_anchor load. So the entire cluster's data dependency rests on a function that nothing calls, and the one CLI flag that should call it is broken.
- **Evidence:** lines 461-505 def run_full_cl_confirmation(base_cfg): ... 'NOT called by main()'; no caller in repo; writes confirmation_dir=results/cl_confirmation with ckpt_dir=.../checkpoints matching v2 CKPT_SOURCE (run_correspondence_anchor_v2.py:50)
- **Fix:** Wire --confirm to run_full_cl_confirmation(base_cfg) (constructing base_cfg in that branch), or remove the dead function if the checkpoint is produced elsewhere; document the producer->consumer link.
- **Fix at:** `code/run_digit_only_alignment.py:461-505`, `code/run_digit_only_alignment.py:615-617`
- _related: ALGN-027, ALGN-002, ALGN-017_

#### `JNT-023` [MED] Eager full-dataset materialization to read labels (N-MNIST decoded multiple times)
`code/run_joint_shared_encoder.py:104-107` · performance · status:live · effort:medium · confidence:high
- **What:** Label-based filtering indexes the entire transformed dataset just to read labels, decoding every sample; the paired dataset then re-indexes the full datasets again.
- **Evidence:** ClassFilteredDataset.__init__ loops base_dataset[i] over the whole dataset discarding data (104-107); for cached_nmnist (DiskCachedDataset over ToFrame) this decodes every N-MNIST sample. PairedDigitDatasetFiltered.__init__ re-indexes the full visual+audio datasets (143-155). N-MNIST train is decoded several times per seed before training.
- **Fix:** Expose a labels-only accessor or cache label arrays once; avoid indexing the transformed dataset purely to filter by label.
- **Fix at:** `code/run_joint_shared_encoder.py:104-107`, `code/run_joint_shared_encoder.py:143-155`

#### `TST-001` [MED] None of the three code/ smoke tests are real pytest tests, and none are in the validation loop
`code/test_provenance_smoke.py:1` · quality · status:live · effort:medium · confidence:high
- **What:** All three files are named test_*.py (which signals a pytest suite) but define ZERO test_-prefixed functions: test_provenance_smoke.py and test_cross_modal_alignment.py put their whole body at module top level, and test_ordinal_anchor_smoke.py wraps everything in main(). pytest collects 0 test items from any of them. Worse, pytest.ini pins testpaths=tests (pytest.ini:2), so the canonical validation loop `pytest tests/` never even looks at code/. These files therefore contribute nothing to the safe validation loop and give a false impression of test coverage for the M7 provenance/alignment/ordinal code paths (which the repo-root tests/ suite does NOT exercise — it only tests the legacy bi-modal package).
- **Evidence:** pytest.ini:2 `testpaths = tests`; grep for `def test_` across the three files returns nothing; test_provenance_smoke.py has statements at module scope lines 17-110; test_cross_modal_alignment.py lines 67-169; test_ordinal_anchor_smoke.py logic is inside main() (line 22) under a __main__ guard (line 83).
- **Fix:** Convert each smoke script into pytest test_* functions (assertions inside functions, no work at import), add a __main__ guard, and relocate the dataset-free ones (provenance + alignment, both CPU/dummy-data) into tests/ so `pytest tests/` actually runs them. Keep dataset/checkpoint-dependent ordinal logic out of the default collection or mark it skip/xfail with a clear reason.
- **Fix at:** `code/test_provenance_smoke.py:17-110`, `code/test_cross_modal_alignment.py:67-169`, `code/test_ordinal_anchor_smoke.py:22-84`, `pytest.ini:2`
- _related: TST-002, TST-009, TST-019_


## LOW (59)

#### `DEP-016` [LOW] Stale .gitignore comment references notebooks that do not exist
`code/.gitignore:34-35` · docs · status:live · effort:trivial · confidence:high
- **What:** Line 34 comment reads 'Jupyter checkpoints (keep notebooks, ignore checkpoints)', implying notebooks are tracked. Per CLAUDE.md there is no .ipynb anywhere in the repo (the M7 notebook is gone). The comment is misleading; the .ipynb_checkpoints/ rule itself is harmless.
- **Evidence:** code/.gitignore:34 '# Jupyter checkpoints (keep notebooks, ignore checkpoints)' ; :35 .ipynb_checkpoints/
- **Fix:** Drop or reword the comment so it does not imply tracked notebooks exist.
- **Fix at:** `code/.gitignore:34-35`

#### `ARCH-009` [LOW] Legacy MPS throughput benchmark (targets bi-modal package, not M7)
`code/archive/benchmark_mps.py:1-63` · dead-code · status:archive · effort:trivial · confidence:high
- **What:** Batch-size throughput micro-benchmark on the LEGACY Models.registry model (line 9) using synthetic data. Prints samples/sec (timing, computed not hardcoded). Not tied to M7 and not a research result. Blocked by a stale absolute sys.path (ARCH-022) and a hardcoded MPS device with no fallback (ARCH-023).
- **Evidence:** build_model + train_step imports lines 9-10; DEVICE=torch.device('mps') line 17; samples_per_sec print line 63.
- **Fix:** Retirement verdict DELETE: ad-hoc benchmark on the legacy package; no reproducibility or M7 role.
- **Fix at:** `code/archive/benchmark_mps.py:1-63`
- _related: ARCH-022, ARCH-023_

#### `ARCH-023` [LOW] Hardcoded MPS device with no CPU/CUDA fallback
`code/archive/benchmark_mps.py:17` · quality · status:archive · effort:trivial · confidence:high
- **What:** DEVICE = torch.device('mps') is hardcoded (line 17), so the benchmark crashes on any non-Apple-Silicon machine (CUDA box, Linux CI) instead of falling back. The rest of the repo uses get_device() auto-selection (MPS->CUDA->CPU). Minor since the file is being retired and only prints timing, not research numbers.
- **Evidence:** DEVICE = torch.device('mps') line 17; torch.mps.synchronize guarded by hasattr lines 44,58.
- **Fix:** None for retirement (DELETE per ARCH-009). If revived, use the shared device auto-select.
- **Fix at:** `code/archive/benchmark_mps.py:17`
- _related: ARCH-009_

#### `ARCH-007` [LOW] One-off debug scratch ('why is exact SpiNNaker sim silent')
`code/archive/debug_brian2_exact.py:1-195` · dead-code · status:archive · effort:trivial · confidence:high
- **What:** Single-sample debug script (runs only label=0, line 148) built to diagnose why the exact PyNN simulation produced no output spikes. Reaches into Brian2 internals via input_pop.brian2_group.set_spikes (line 169). Un-runnable today: imports pyNN/brian2 (ARCH-010), loads missing checkpoint (ARCH-012), duplicates ConvFrontend (ARCH-011). Imports Synapses and nA (line 14) that are never used.
- **Evidence:** single-sample loop label=0 lines 148-188; brian2_group.set_spikes line 169; unused imports Synapses,nA line 14.
- **Fix:** Retirement verdict DELETE: transient debugging artifact with no reproducibility role; its conclusion is folded into verify_full.py / final_evaluation.py.
- **Fix at:** `code/archive/debug_brian2_exact.py:1-195`
- _related: ARCH-010, ARCH-011, ARCH-012_

#### `ARCH-008` [LOW] Legacy gradient-flow sanity script (targets bi-modal package, not M7)
`code/archive/debug_gradient.py:1-98` · dead-code · status:archive · effort:trivial · confidence:high
- **What:** Generic SNN gradient-flow / tiny-overfit sanity check that builds the LEGACY bi-modal model via Models.registry.build_model('baseline', ...) (line 9,17). It does not exercise M7 at all, and the tests/ suite now covers the legacy package. Self-contained (synthetic data, main() guarded), but blocked by a stale absolute sys.path (ARCH-022).
- **Evidence:** from Models.registry import build_model line 9; build_model call lines 17,70; synthetic torch.randn inputs lines 22-23,73.
- **Fix:** Retirement verdict DELETE: superseded by tests/ for the legacy package; no M7 relevance.
- **Fix at:** `code/archive/debug_gradient.py:1-98`
- _related: ARCH-022_

#### `ARCH-019` [LOW] forgetting_T2_T3 and total_forgetting_T2_T3 are identical duplicate columns
`code/archive/run_cl_confirmation.py:413-415` · quality · status:archive · effort:trivial · confidence:high
- **What:** forgetting_T2_T3 = acc_t2_after - acc_t2_after_t3 (line 413) and total_forgetting_T2_T3 = acc_t2_after - acc_t2_after_t3 (line 415) are computed from the same two quantities and are therefore always equal. Both are written to the CSV and both appear in the aggregation metrics list, producing two identically-valued summary rows and inviting double-counting.
- **Evidence:** identical RHS at lines 413 and 415; both in EXPECTED_COLUMNS (lines 76-77) and metrics list (lines 486-487).
- **Fix:** Drop one of the two columns (keep total_forgetting_T2_T3 for consistency with T1) in any revived version. Retirement: flag so old summaries are read knowing the two columns are redundant.
- **Fix at:** `code/archive/run_cl_confirmation.py:413`, `code/archive/run_cl_confirmation.py:415`
- _related: ARCH-018_

#### `ARCH-016` [LOW] Dead vis_indices/aud_indices loop that still consumes the RNG
`code/archive/run_correspondence_anchor.py:313-322` · dead-code · status:archive · effort:trivial · confidence:high
- **What:** In build_paired_feature_dataset the first per-class loop builds vis_indices/aud_indices via rng.choice (lines 313-322) and they are never read — a follow-up comment 'Actually we need PAIRED samples' recomputes everything in a second loop (lines 326-343). The dead loop is not inert: it advances the same rng (created line 312), so the real paired selection at lines 335-336 draws different samples than it would without the dead loop. Removing it would change which samples train the projection heads, i.e. it is dead code with a numeric side effect.
- **Evidence:** unused vis_indices/aud_indices lines 313-322 vs paired recompute lines 326-343; shared rng line 312 consumed by both.
- **Fix:** Delete the dead loop, but note it will change results (RNG stream shifts); acceptable only because the file is being retired. If preserving v1 outputs bit-for-bit, leave it. Prefer DELETE the whole file (ARCH-001).
- **Fix at:** `code/archive/run_correspondence_anchor.py:313-322`
- _related: ARCH-001_

#### `ARCH-005` [LOW] Redundant raw-Brian2 speed variant of verify_full.py
`code/archive/verify_brian2_raw.py:1-376` · dead-code · status:archive · effort:trivial · confidence:high
- **What:** Same full-network verification but using raw Brian2 Synapses/NeuronGroup (prefs.codegen.target='numpy', line 25) instead of the PyNN wrapper, for speed. Functionally redundant with verify_full.py once the discretization was settled and final_evaluation.py locked. Un-runnable today: imports brian2 (ARCH-010), loads missing checkpoint (ARCH-012), duplicates ConvFrontend (ARCH-011). Also the source of the strongest overclaim comment (ARCH-013).
- **Evidence:** raw brian2 imports lines 20-25; eqs comment 'exact same as PyNN/SpiNNaker' line 180.
- **Fix:** Retirement verdict KEEP-ARCHIVED (or DELETE): documents the raw-Brian2 cross-check; no unique live value.
- **Fix at:** `code/archive/verify_brian2_raw.py:1-376`
- _related: ARCH-010, ARCH-011, ARCH-012, ARCH-013_

#### `ARCH-003` [LOW] Superseded by live final_evaluation.py (SpiNNaker discrete-time sim verify)
`code/archive/verify_full.py:1-355` · dead-code · status:archive · effort:trivial · confidence:high
- **What:** Full-network build-once verification of the PyTorch discrete-time simulator against Brian2/PyNN IF_curr_exp (Variant D, cross-factor). The live code/final_evaluation.py ('Final locked evaluation of the corrected SpiNNaker discrete-time simulator', SCALE=7.0) is the settled successor of this verification lineage. Un-runnable today: missing dependency (ARCH-010) and missing checkpoint (ARCH-012); duplicates ConvFrontend (ARCH-011).
- **Evidence:** docstring lines 1-7; run_pytorch_sim uses cross_factor (line 136); final_evaluation.py header lines 3-10.
- **Fix:** Retirement verdict KEEP-ARCHIVED (or DELETE). It is the closest provenance record of the real-network verification; final_evaluation.py is the locked version. Safe to delete once provenance is captured elsewhere.
- **Fix at:** `code/archive/verify_full.py:1-355`
- _related: ARCH-010, ARCH-011, ARCH-012, ARCH-013_

#### `ARCH-004` [LOW] Superseded by verify_full.py (earlier per-sample-rebuild variant)
`code/archive/verify_simulators.py:1-362` · dead-code · status:archive · effort:trivial · confidence:high
- **What:** Earlier version of the full-network verification that uses PyTorch Variant A ((ie+b)*v_factor, line 139) and rebuilds the entire PyNN network inside the per-sample loop (ARCH-021). verify_full.py replaced it with Variant D and a build-once network. Redundant intermediate. Un-runnable today (ARCH-010, ARCH-012); duplicates ConvFrontend (ARCH-011).
- **Evidence:** PyTorch sim uses (ie+b)*v_factor at line 139 (Variant A) vs verify_full.py cross_factor; per-sample Population/Projection construction lines 186-238.
- **Fix:** Retirement verdict DELETE: strictly superseded by verify_full.py; keeps no unique capability.
- **Fix at:** `code/archive/verify_simulators.py:1-362`
- _related: ARCH-021, ARCH-010, ARCH-011, ARCH-012_

#### `ARCH-021` [LOW] Rebuilds the entire PyNN network (Populations + Projections) inside the per-sample loop
`code/archive/verify_simulators.py:186-238` · performance · status:archive · effort:trivial · confidence:high
- **What:** run_brian2_sim reconstructs all Populations and all Projections from scratch for each of the 20 samples (loop at line 186; Population/Projection built lines 190-238), including re-enumerating every above-threshold weight of the 1600->1024->512->10 network per sample. verify_full.py fixes exactly this by building once and calling sim.reset() between samples. This per-sample rebuild is the concrete reason verify_simulators.py was superseded.
- **Evidence:** per-sample loop line 186; input/FC1/FC2/head Populations rebuilt lines 190-216; build_proj called per sample lines 236-238; contrast verify_full.py build-once at lines 179-222 + sim.reset() line 240.
- **Fix:** None for retirement (DELETE per ARCH-004). Recorded as the supersession rationale.
- **Fix at:** `code/archive/verify_simulators.py:186-238`
- _related: ARCH-004, ARCH-003_

#### `CLE-012` [LOW] Unused imports: numpy and random_split
`code/eval_transfer_fixed.py:11-14` · dead-code · status:live · effort:trivial · confidence:high
- **What:** `import numpy as np` (11) and `random_split` (14) are never used in this file (grep: np. count 0; random_split count 1 = import only).
- **Evidence:** Line 11 import numpy; line 14 imports random_split; no other references.
- **Fix:** Remove the two unused imports.
- **Fix at:** `code/eval_transfer_fixed.py:11`, `code/eval_transfer_fixed.py:14`

#### `CLE-031` [LOW] Dead else branch in _events_to_dense duplicated across four files
`code/eval_transfer_fixed.py:50` · dead-code · status:live · effort:trivial · confidence:high
- **What:** The redundant `... if len(events) > 0 else 1` inside an `if len(events) > 0:` block (dead else) recurs in eval_transfer_fixed.py:50, run_rehearsal_transfer_eval_only.py:69, run_cl_confirmation_fixed.py:105, and run_cl_rehearsal.py:83. One shared helper (CLE-030) would remove all four at once.
- **Evidence:** Line 50 (and the three siblings) `max_time = events["t"].max() if len(events) > 0 else 1` under an outer len>0 guard.
- **Fix:** Remove the dead ternary in all four copies (best done when consolidating into a shared helper).
- **Fix at:** `code/eval_transfer_fixed.py:50`, `code/run_rehearsal_transfer_eval_only.py:69`, `code/run_cl_confirmation_fixed.py:105`, `code/run_cl_rehearsal.py:83`
- _related: CLE-023, CLE-030_

#### `FIG-026` [LOW] mean_confusion_matrix computes a single (non-mean) CM; dead 'p>=0' guard; transposed cm[pred,true] orientation
`code/final_evaluation.py:187-192` · quality · status:live · effort:trivial · confidence:high
- **What:** mean_confusion_matrix (187-192) computes counts for ONE seed's predictions, not a mean - the averaging happens later at np.mean(spinnaker_cms, axis=0) (276) - so the name is misleading. The guard `if p >= 0` (190) is dead: p comes from np.argmax and is always >=0 (see FIG-025 - all-zero rows still yield 0, never negative). cm[p, t] += 1 (191) stores [predicted, true], the transpose of the conventional cm[true, pred]; the print loop labels rows r and columns by class index (294-297), so a reader assuming sklearn convention would misread rows/columns.
- **Evidence:** def mean_confusion_matrix(...): if p >= 0: cm[p, t] += 1
- **Fix:** Rename to confusion_matrix; drop the dead p>=0 guard (or make it meaningful once no-prediction sentinels exist, FIG-025); document the [pred,true] orientation or switch to [true,pred].
- **Fix at:** `code/final_evaluation.py:187-192`, `code/final_evaluation.py:294-297`
- _related: FIG-025_

#### `FIG-011` [LOW] POSITIVE: dedup is correct (one row per seed+condition); minor: .first() is first-non-null-per-column, not 'first row'
`code/generate_figures_3seed.py:65-77` · quality · status:live · effort:trivial · confidence:high
- **What:** load_summary correctly (a) filters to summary rows via df['step'].isna() OR blank-step (65) - more robust than the documented isna()-only ritual, (b) exact-dedups (66), (c) warns on near-duplicate seed+condition rows (68-75), and (d) collapses to one row per (seed,condition) with groupby(...).first() (77). This is the CORRECT population for mean/std and avoids the n-inflation trap flagged in CLAUDE.md. Minor precision note: GroupBy.first() returns the first NON-NULL value per column, not the first physical row, so the comment 'Keeping the FIRST row' (74) is slightly inaccurate; for near-duplicate rows with scattered NaNs it could assemble a column-wise mix. Harmless for complete summary rows.
- **Evidence:** summary.groupby(['seed','condition'], as_index=False).first()
- **Fix:** No action required for the dedup. Optionally use .head(1)/.nth(0) if 'first physical row' semantics are intended, and reword the comment at 74.
- **Fix at:** `code/generate_figures_3seed.py:74`, `code/generate_figures_3seed.py:77`

#### `FIG-015` [LOW] Expected seed set {42,123,456} hardcoded in the plotting script
`code/generate_figures_3seed.py:120` · quality · status:live · effort:trivial · confidence:high
- **What:** expected = {42, 123, 456} (120) duplicates the seed policy also hardcoded at m7_trimodal_run.py:516. It is used only to emit a 'not a 3-seed result' warning (121-124), so it is a policy check, not a plotted result number - benign. But it is a second copy of the seed policy that can drift from the training script.
- **Evidence:** expected = {42, 123, 456}
- **Fix:** Optional: source the expected seeds from a shared config/constant rather than re-hardcoding.
- **Fix at:** `code/generate_figures_3seed.py:120`
- _related: FIG-021_

#### `RUN-024` [LOW] safe_empty_cache is never called and ignores MPS despite its 'MPS/CPU compatibility' banner
`code/m7_trimodal_run.py:20-22` · dead-code · status:live · effort:trivial · confidence:high
- **What:** The function only empties the CUDA cache and sits under a 'MPS/CPU compatibility helpers' header, but it has no MPS branch (torch.mps.empty_cache) and is never called anywhere in the file. Dead code.
- **Evidence:** lines 20-22 def safe_empty_cache with only 'if torch.cuda.is_available(): torch.cuda.empty_cache()'; no call site
- **Fix:** Remove it, or if MPS memory pressure is real, call it at task boundaries and add a torch.mps.empty_cache branch.
- **Fix at:** `code/m7_trimodal_run.py:19-22`

#### `RUN-025` [LOW] Unused stdlib imports os, sys, time
`code/m7_trimodal_run.py:4-6` · dead-code · status:live · effort:trivial · confidence:high
- **What:** os, sys and time are imported at the top of the file but never used anywhere in it.
- **Evidence:** lines 4-6 'import os', 'import sys', 'import time'; no references elsewhere
- **Fix:** Remove the three unused imports.
- **Fix at:** `code/m7_trimodal_run.py:4-6`
- _related: RUN-026, RUN-027_

#### `RUN-026` [LOW] Dead local import 'from torch.nn import Linear, LayerNorm' inside the model constructor
`code/m7_trimodal_run.py:280` · dead-code · status:live · effort:trivial · confidence:high
- **What:** Inside the tasks loop the constructor does 'from torch.nn import Linear, LayerNorm', but the code below builds modules with nn.Linear / nn.LayerNorm and never uses the bare Linear or LayerNorm names. The import is dead and runs once per audio task.
- **Evidence:** line 280 'from torch.nn import Linear, LayerNorm'; subsequent code uses nn.Linear (282-288)
- **Fix:** Remove the local import.
- **Fix at:** `code/m7_trimodal_run.py:280`
- _related: RUN-025_

#### `RUN-027` [LOW] pandas imported mid-file at line 648, after the functions that use it are defined
`code/m7_trimodal_run.py:648` · quality · status:live · effort:trivial · confidence:high
- **What:** run_one_seed (521-643) and the two CSV helpers reference pd, but 'import pandas as pd' appears at line 648. It works only because those functions are first called at line 710, after the import executes; calling run_one_seed before line 648 would raise NameError. A latent trap that also hides a heavy dependency mid-module.
- **Evidence:** line 648 'import pandas as pd'; pd first referenced at 636 inside run_one_seed defined at 521
- **Fix:** Move 'import pandas as pd' to the top-of-file import block with the other imports.
- **Fix at:** `code/m7_trimodal_run.py:648`, `code/m7_trimodal_run.py:4-17`

#### `RUN-028` [LOW] ModernHopfieldLayer default num_patterns=256 is dead, and it has two redundant scaling knobs
`code/m7_trimodal_run.py:202-209` · quality · status:live · effort:trivial · confidence:high
- **What:** The constructor default num_patterns=256 is never used because the call site passes CONFIG['num_patterns']=100 (293); reading the signature alone wrongly suggests 256 (a known repo trap). Separately, forward divides by both self.temperature (fixed 1.0) and self.scale=sqrt(dim), two scaling factors where modern Hopfield normally uses a single inverse-temperature beta; with temperature=1.0 one is a no-op and the pair is confusing.
- **Evidence:** line 202 'def __init__(self, dim=512, num_patterns=256, temperature=1.0)'; call site 293 passes CONFIG['num_patterns']=100; line 206 'self.scale = np.sqrt(dim)'; line 209 '/ self.temperature / self.scale'
- **Fix:** Set the constructor default to the intended 100 (or remove the default), and collapse the two scaling factors into one documented beta.
- **Fix at:** `code/m7_trimodal_run.py:202`, `code/m7_trimodal_run.py:206`, `code/m7_trimodal_run.py:209`

#### `RUN-029` [LOW] Model moved to device twice and device passed as a string rather than torch.device
`code/m7_trimodal_run.py:529-532` · quality · status:live · effort:trivial · confidence:high
- **What:** The model is moved with .to(device) at creation (529) and again inside ContinualLearner.__init__ (402). The buffer and trainer are constructed with str(device) (530-531), so self.device is the string 'mps'/'cpu' rather than a torch.device, inconsistent with the module-level torch.device object and with populate() which uses next(model.parameters()).device. It works but is redundant and type-inconsistent.
- **Evidence:** line 529 'model = M7_ContinualAdaptiveModel(...).to(device)'; line 402 'self.model = model.to(device)'; lines 530-531 'device=str(device)'
- **Fix:** Move the model once and pass the torch.device object (not str) consistently to the buffer and trainer.
- **Fix at:** `code/m7_trimodal_run.py:402`, `code/m7_trimodal_run.py:529-532`

#### `RUN-030` [LOW] _task_labels is populated but never read - the exact map needed to fix the replay bugs
`code/m7_trimodal_run.py:359-378` · dead-code · status:live · effort:small · confidence:high
- **What:** EmbeddingReplayBuffer records self._task_labels[c] = task_id in populate (378) but nothing ever reads _task_labels. This is the per-class task identity that sample() (RUN-002) and forward_from_embeddings routing (RUN-004) need to be correct; it is computed and then ignored.
- **Evidence:** line 359 'self._task_labels = {}'; line 378 'self._task_labels[c] = task_id'; no read of _task_labels anywhere
- **Fix:** Consume _task_labels in sample() to select by task, and carry it through to route replay embeddings to the correct head (see RUN-002, RUN-004).
- **Fix at:** `code/m7_trimodal_run.py:359`, `code/m7_trimodal_run.py:378`, `code/m7_trimodal_run.py:380-392`
- _related: RUN-002, RUN-004_

#### `RUN-031` [LOW] Stale comment claims the caller renames the final checkpoint; no rename exists
`code/m7_trimodal_run.py:502` · docs · status:live · effort:trivial · confidence:high
- **What:** The comment 'Save final checkpoint (caller handles condition suffix via rename after return)' describes a rename that never happens - the caller writes a fresh final_*.pt via torch.save (543/571/598) and no code renames anything. The comment misleads a reader about the checkpoint flow.
- **Evidence:** line 502 comment; caller torch.save at 543, 571, 598; no os.rename/Path.rename in the file
- **Fix:** Delete or correct the comment to reflect that the caller writes final_*.pt directly.
- **Fix at:** `code/m7_trimodal_run.py:502`
- _related: RUN-007_

#### `RUN-032` [LOW] PLOT_DIR is created at import but never used
`code/m7_trimodal_run.py:40-42` · dead-code · status:live · effort:trivial · confidence:high
- **What:** PLOT_DIR is defined and mkdir'd at import but nothing in the file writes to it (figures are generated by a separate script). It is dead, and its creation is part of the import-time side effects.
- **Evidence:** line 40 'PLOT_DIR = RESULTS_DIR / "plots"'; line 41 mkdir loop; no other reference to PLOT_DIR
- **Fix:** Remove PLOT_DIR (and its mkdir) from this file, or move directory creation into the guarded main().
- **Fix at:** `code/m7_trimodal_run.py:40-42`
- _related: RUN-001_

#### `RUN-035` [LOW] Derived metrics survive only in the end-of-seed summary row; an interrupted seed loses forgetting/fwt
`code/m7_trimodal_run.py:549-604` · quality · status:live · effort:small · confidence:medium
- **What:** Only the raw T*_after_T* accuracies are persisted incrementally after each task (549/577/604). All derived metrics (forgetting_T1_T2, forgetting_T1_T3, forgetting_T2_T3, fwt) are computed and written only in the final summary row (612-640). If a seed is interrupted after some tasks, the incremental rows remain but no derived metrics are recoverable, and the checkpoints alone do not contain them.
- **Evidence:** incremental saves 549, 577, 604 store only accuracies; derived metrics computed 613-616 and saved only in the summary row 618-640
- **Fix:** Persist derived metrics incrementally as they become computable (e.g. after T2 write forgetting_T1_T2 and fwt) so an interrupted run is analyzable.
- **Fix at:** `code/m7_trimodal_run.py:549`, `code/m7_trimodal_run.py:577`, `code/m7_trimodal_run.py:604`, `code/m7_trimodal_run.py:612-640`
- _related: RUN-010_

#### `PROV-019` [LOW] Unused imports: os, sys, time, copy, and typing Optional/Tuple/Dict
`code/m7_trimodal_run_provenance.py:7-14` · dead-code · status:live · effort:trivial · confidence:high
- **What:** Grep over the whole file (excluding import lines) finds no use of os (7), sys (8), time (9), or the copy module (10) — the only 'copy' token is the pandas .copy() method at 885. From typing (14), Optional and Tuple and Dict are never used (List is). These are harmless but are dead code in the shared hub.
- **Evidence:** os/sys/time/copy: 0 uses outside imports (grep); typing Optional/Tuple/Dict: 0 uses; List used at 36,62-63,873.
- **Fix:** Remove the unused imports (os, sys, time, copy, Optional, Tuple, Dict). Note copy would become NEEDED if PROV-004 is fixed with copy.deepcopy.
- **Fix at:** `code/m7_trimodal_run_provenance.py:7-14`
- _related: PROV-004_

#### `PROV-020` [LOW] safe_empty_cache is defined but never called
`code/m7_trimodal_run_provenance.py:89-91` · dead-code · status:live · effort:trivial · confidence:high
- **What:** safe_empty_cache (89-91) empties the CUDA cache but is not called anywhere in this file (grep). It is dead code carried over from the orphan (m7_trimodal_run.py:20-22, also unused there).
- **Evidence:** def safe_empty_cache at 89; no call sites in file.
- **Fix:** Remove it, or call it at appropriate points (e.g. between tasks) if cache pressure is a real concern; currently it is inert.
- **Fix at:** `code/m7_trimodal_run_provenance.py:89-91`

#### `PROV-025` [LOW] Redundant duplicate DataLoader instances constructed over the same split
`code/m7_trimodal_run_provenance.py:1186-1195` · quality · status:live · effort:small · confidence:high
- **What:** For each modality build_data_loaders builds a 'train'/'val'/'test' loader AND a separate 'train_val_test' tuple of loaders over the SAME underlying splits (e.g. N-MNIST 1186-1194). The training path uses train_val_test (997-998) while buffer.populate uses the separate 'train' loader (1011). This doubles loader construction and, because both train loaders have shuffle=True with no fixed generator, they iterate in different orders; it is redundant and mildly confusing rather than incorrect.
- **Evidence:** nmnist_loaders 'train' (1187) and 'train_val_test'[0] (1191) both wrap train_split; used at 1011 vs 997-998.
- **Fix:** Construct each loader once and reference it in both places, or drop the train_val_test tuple in favour of named keys.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1186-1248`

#### `CLE-014` [LOW] Unused imports: numpy, pandas, random_split
`code/phase1_sanity_check.py:14-29` · dead-code · status:live · effort:trivial · confidence:high
- **What:** numpy (14), pandas (15), and random_split (29) are imported but never used (grep: np. 0, pd. 0, random_split 1 = import only). The seed note flagged numpy/pandas as ~unused; confirmed, plus random_split.
- **Evidence:** Lines 14, 15, 29; no np./pd./random_split usage elsewhere in the file.
- **Fix:** Remove all three unused imports.
- **Fix at:** `code/phase1_sanity_check.py:14-15`, `code/phase1_sanity_check.py:29`

#### `CLE-017` [LOW] Unused import: dataclasses
`code/run_cl_confirmation_fixed.py:10` · dead-code · status:live · effort:trivial · confidence:high
- **What:** `import dataclasses` (10) is never used (grep: dataclasses. count 0). Same unused import appears in run_cl_rehearsal.py (see CLE-026).
- **Evidence:** Line 10; no dataclasses.* reference in the file.
- **Fix:** Remove the import.
- **Fix at:** `code/run_cl_confirmation_fixed.py:10`
- _related: CLE-026_

#### `CLE-019` [LOW] cfg.pin_memory and cfg.num_workers are ignored by the DataLoaders
`code/run_cl_confirmation_fixed.py:88-95` · quality · status:live · effort:small · confidence:high
- **What:** Every DataLoader hardcodes num_workers=0 and omits pin_memory, so cfg.pin_memory=True (371) and cfg.num_workers=0 (371) are never actually consumed. num_workers=0 is correct on macOS/MPS, but silently ignoring config fields means the saved config.json advertises settings the run did not use (pin_memory=True in particular). Same in run_cl_rehearsal.py.
- **Evidence:** Loaders at 88-95, 120-127, 141-148 pass num_workers=0 literally and no pin_memory; grep pin_memory=cfg count 0.
- **Fix:** Either pass cfg.pin_memory (and keep num_workers=0 with a comment on the macOS constraint) or drop the unused fields from the config so provenance matches behaviour.
- **Fix at:** `code/run_cl_confirmation_fixed.py:88-95`, `code/run_cl_confirmation_fixed.py:371`
- _related: CLE-020_

#### `CLE-023` [LOW] Dead else branch in _events_to_dense max_time guard
`code/run_cl_confirmation_fixed.py:105` · dead-code · status:live · effort:trivial · confidence:high
- **What:** `max_time = events["t"].max() if len(events) > 0 else 1` (105) sits inside `if len(events) > 0:` (104), so the `else 1` is unreachable. Cosmetic but confusing, and copy-pasted verbatim into 4 files (see CLE-031).
- **Evidence:** Line 104 guard, line 105 redundant ternary.
- **Fix:** Drop the redundant ternary: inside the guard, `max_time = events["t"].max()`.
- **Fix at:** `code/run_cl_confirmation_fixed.py:105`
- _related: CLE-031_

#### `CLE-026` [LOW] Unused import: dataclasses
`code/run_cl_rehearsal.py:15` · dead-code · status:live · effort:trivial · confidence:high
- **What:** `import dataclasses` (15) is never used (grep: dataclasses. count 0). Same as run_cl_confirmation_fixed.py (CLE-017).
- **Evidence:** Line 15; no dataclasses.* reference.
- **Fix:** Remove the import.
- **Fix at:** `code/run_cl_rehearsal.py:15`
- _related: CLE-017_

#### `CMP-011` [LOW] within_set_visual/within_set_audio columns are always NaN
`code/run_compositional_crossmodal.py:859` · dead-code · status:live · effort:trivial · confidence:high
- **What:** run_one_model initializes within_set_visual/audio to NaN and never assigns them; within_modality_gate returns only 'loo' with a comment that within-modality SET is 'not available here'. These two CSV columns are dead in every output row.
- **Evidence:** 'within_set_visual': float('nan'), 'within_set_audio': float('nan')  # never set; gate returns only {'loo': ...}
- **Fix:** Either implement within-modality SET or remove the two columns from CSV_COLUMNS and the row dict.
- **Fix at:** `code/run_compositional_crossmodal.py:859`, `code/run_compositional_crossmodal.py:789-791`, `code/run_compositional_crossmodal.py:125-138`

#### `CMP-012` [LOW] Unused locals and a stale whole_number_retrieval docstring
`code/run_compositional_crossmodal.py:683` · quality · status:live · effort:trivial · confidence:high
- **What:** d = slots.shape[2] is computed but unused in set_order_metrics (683) and _set_perm_null (712). whole_number_retrieval's docstring describes same-modality LOO handled 'by caller', but the function is only ever called cross-modally (vZ vs Pa, aZ vs Pv), so that path never runs; within-modality LOO is done separately in within_modality_gate.
- **Evidence:** d = slots.shape[2]  # unused;  docstring: 'LOO handled by caller ... when same-modality' (never same-modality)
- **Fix:** Delete the unused d locals; correct the whole_number_retrieval docstring to state it is cross-modal only.
- **Fix at:** `code/run_compositional_crossmodal.py:683`, `code/run_compositional_crossmodal.py:712`, `code/run_compositional_crossmodal.py:733-737`

#### `CMP-014` [LOW] _build_cfg populates ~30 fields but only 3 are consumed
`code/run_compositional_crossmodal.py:153-165` · quality · status:live · effort:small · confidence:high
- **What:** _build_cfg constructs a full tri-modal/Hopfield/replay/align ExperimentConfig, but this file uses cfg only via get_device(cfg) (device) and _build_modality_loaders(cfg) (nmnist_train_per_class, batch_size). cfg.max_epochs/lr/patience are IGNORED - the real training budget is the module constants PHASE_A_EPOCHS/PHASE_B_EPOCHS/lr=1e-3 hardcoded in train_compositional. The sidecar even records lr=1e-3 as a literal, not from cfg. Editing cfg silently has no effect on training, which is misleading.
- **Evidence:** _build_cfg sets beta/dropout/num_patterns/num_gru_layers/align_*/... none used; train_compositional hardcodes lr=1e-3
- **Fix:** Reduce _build_cfg to the fields actually consumed, and source the sidecar values from the real constants.
- **Fix at:** `code/run_compositional_crossmodal.py:153-165`, `code/run_compositional_crossmodal.py:490`, `code/run_compositional_crossmodal.py:1208-1216`
- _related: CMP-025_

#### `ALGN-010` [LOW] Unused import List
`code/run_correspondence_anchor_v2.py:23` · quality · status:live · effort:trivial · confidence:high
- **What:** typing.List is imported at line 23 (from typing import Dict, List, Tuple) but never used in the file.
- **Evidence:** line 23 from typing import Dict, List, Tuple; only reference to List is the import itself
- **Fix:** Drop List from the typing import.
- **Fix at:** `code/run_correspondence_anchor_v2.py:23`

#### `ALGN-026` [LOW] Unused imports copy, os, Dict
`code/run_cross_modal_alignment_study.py:32-38` · quality · status:live · effort:trivial · confidence:high
- **What:** import copy (32) and import os (36) have zero real attribute uses (copy. and os. never called); typing.Dict (38) is imported but unused (only lowercase dict is used in annotations). Also note seed set here is [42,123,456,789,101112] (line 413) where 101112 looks like a typo divergence from the 1011 used in run_ordinal_anchor and run_digit_only.
- **Evidence:** line 32 import copy; line 36 import os; line 38 from typing import Dict, List (Dict unused); line 413 seeds=[42,123,456,789,101112]
- **Fix:** Remove unused copy/os imports and Dict; reconcile the 101112 vs 1011 seed if intended to match the other studies.
- **Fix at:** `code/run_cross_modal_alignment_study.py:32`, `code/run_cross_modal_alignment_study.py:36`, `code/run_cross_modal_alignment_study.py:38`, `code/run_cross_modal_alignment_study.py:413`

#### `ALGN-032` [LOW] Unused imports copy, os, time, List, Dict
`code/run_digit_only_alignment.py:31-38` · quality · status:live · effort:trivial · confidence:high
- **What:** import copy (31), import os (34), import time (36) have zero real attribute uses; typing.List and typing.Dict (38) are imported but unused. argparse/dataclasses/numpy/pandas/torch/tonic/scipy are used.
- **Evidence:** lines 31,34,36 import copy/os/time (no copy./os./time. calls); line 38 from typing import Dict, List (both unused)
- **Fix:** Remove copy, os, time and the unused typing names.
- **Fix at:** `code/run_digit_only_alignment.py:31`, `code/run_digit_only_alignment.py:34`, `code/run_digit_only_alignment.py:36`, `code/run_digit_only_alignment.py:38`

#### `CMP-023` [LOW] Redundant unused import _EC 'to ensure import path'
`code/run_joint_diagnostic.py:65` · dead-code · status:live · effort:trivial · confidence:high
- **What:** Line 65 imports ExperimentConfig as _EC from m7_trimodal_run_provenance with the comment 'ensures import path', but the import path is already established by importing run_joint_magnitude_ablation (which imports provenance). _EC is never used (flagged F401), so the rationale is moot.
- **Evidence:** from m7_trimodal_run_provenance import ExperimentConfig as _EC  # noqa: E402,F401  (ensures import path)
- **Fix:** Remove the unused import.
- **Fix at:** `code/run_joint_diagnostic.py:65`

#### `CMP-025` [LOW] _build_cfg populates ~30 fields; ~7 consumed, the rest inert
`code/run_joint_diagnostic.py:624-636` · quality · status:live · effort:small · confidence:high
- **What:** Consumed fields: device (get_device), nmnist_train_per_class and batch_size (_build_modality_loaders), and lr/weight_decay/max_epochs/patience (train_model_diag). All the tri-modal/Hopfield/replay/align fields (beta, dropout, hidden_dim, shared_dim, num_patterns, hopfield_temperature, num_gru_layers, time_steps, use_contrastive, contrastive_*, use_replay, replay_*, align_*) are inert noise, so the effective configuration is obscured.
- **Evidence:** train_model_diag(..., max_epochs=cfg.max_epochs, lr=cfg.lr, weight_decay=cfg.weight_decay, patience=cfg.patience); rest of cfg unused
- **Fix:** Reduce _build_cfg to the consumed fields, or add a comment enumerating which fields are actually used here.
- **Fix at:** `code/run_joint_diagnostic.py:624-636`
- _related: CMP-014_

#### `CMP-026` [LOW] Docstring cites unsourced result numbers and fragile cross-file line references
`code/run_joint_diagnostic.py:9-16` · docs · status:live · effort:trivial · confidence:high
- **What:** The docstring asserts 'held-out V2A=50% (seed 42)' and 'an UNTRAINED model also scores 50% (:769)' and cites run_joint_magnitude_ablation :378-379, :572-632, :769. Under RULE 1 the 50% figures are unsourced in-repo (there is no results CSV). The line references are accurate today (verified: separate classifiers at 378-379; compute_held_out_metrics at 572-632; the untrained control near 768-769) but will rot when that file is edited.
- **Evidence:** 'the magnitude-ablation OFF arm held-out V2A=50% (seed 42)'; '(compute_held_out_metrics, :572-632)'; 'an UNTRAINED model also scores 50% (:769)'
- **Fix:** Mark the 50% as an unsourced prior observation pending a CSV, and replace hard line numbers with symbol names.
- **Fix at:** `code/run_joint_diagnostic.py:9-16`

#### `JNT-010` [LOW] Unused imports: math, json, TensorDataset, SHDDigitIdentityDataset
`code/run_joint_magnitude_ablation.py:23-44` · dead-code · status:live · effort:trivial · confidence:high
- **What:** Four imports are never referenced anywhere in the file.
- **Evidence:** grep confirms no math., json., TensorDataset, or SHDDigitIdentityDataset usages beyond the import lines: json (23), math (24), TensorDataset (34), SHDDigitIdentityDataset (44).
- **Fix:** Remove the four unused imports.
- **Fix at:** `code/run_joint_magnitude_ablation.py:23-24`, `code/run_joint_magnitude_ablation.py:34`, `code/run_joint_magnitude_ablation.py:44`
- _related: JNT-028_

#### `JNT-012` [LOW] Paired loaders fully iterated an extra time just for the (broken) leakage assert
`code/run_joint_magnitude_ablation.py:775-778` · performance · status:live · effort:small · confidence:medium
- **What:** Before training, both paired loaders are iterated end-to-end only to run the leakage assert, then train_model iterates them again.
- **Evidence:** run_one_arm iterates paired_train and paired_val at 775-778 for the assert; train_model re-iterates them for training. Combined with the broken idiom (JNT-004) this is a redundant full read per arm/seed.
- **Fix:** Fold the (corrected element-wise) leakage check into the first training pass, or validate labels once at dataset-build time.
- **Fix at:** `code/run_joint_magnitude_ablation.py:775-778`
- _related: JNT-004_

#### `JNT-013` [LOW] ExperimentConfig over-specified with M7/replay/hopfield/align fields this experiment never uses
`code/run_joint_magnitude_ablation.py:868-902` · quality · status:live · effort:small · confidence:medium
- **What:** main() fills ~30 ExperimentConfig fields, most irrelevant to a two-modality non-continual ablation, including a task_order containing 'dvs' although DVS and CL never appear here.
- **Evidence:** Config block 868-902 sets num_patterns, hopfield_temperature, num_gru_layers, time_steps, replay_*, align_* and task_order=['nmnist','shd','dvs']; only lr, weight_decay, batch_size, max_epochs, patience, nmnist_train_per_class and device are consumed.
- **Fix:** Use a minimal experiment-specific config (or a documented subset of ExperimentConfig) and drop the misleading task_order/dvs entry.
- **Fix at:** `code/run_joint_magnitude_ablation.py:868-902`
- _related: JNT-017_

#### `JNT-025` [LOW] run_one_seed is a fourth same-named per-seed runner with a different signature
`code/run_joint_shared_encoder.py:764` · naming · status:live · effort:trivial · confidence:high
- **What:** Multiple modules define run_one_seed with mutually incompatible signatures, and the ablation calls its equivalent run_one_arm, making cross-referencing error-prone.
- **Evidence:** run_one_seed(seed, cfg, shuffled_labels) here (764); m7_trimodal_run_provenance.run_one_seed(cfg, data_loaders, results_dir, ckpt_dir) (962); run_correspondence_anchor_v2.run_one_seed(seed, cfg, loaders, cached, device, head_type) (524); run_joint_magnitude_ablation.run_one_arm (756).
- **Fix:** Give experiment-specific runners distinct names (e.g. run_one_seed_shared_encoder) or converge on a shared interface.
- **Fix at:** `code/run_joint_shared_encoder.py:764`

#### `JNT-028` [LOW] Unused imports/var (os, surrogate, Optional, NUM_HELD_OUT_CLASSES) and a dead SharedCore beta param
`code/run_joint_shared_encoder.py:24-64` · dead-code · status:live · effort:trivial · confidence:high
- **What:** Several imports and a module constant are never used, and SharedCore accepts a beta it ignores.
- **Evidence:** grep confirms no usages: os (24), snntorch.surrogate (42), typing.Optional (31); NUM_HELD_OUT_CLASSES (64) is defined but never referenced; SharedCore.__init__ takes beta (358) but has no LIF and never uses it.
- **Fix:** Remove the unused imports and NUM_HELD_OUT_CLASSES, and drop the dead beta parameter from SharedCore (or use it).
- **Fix at:** `code/run_joint_shared_encoder.py:24`, `code/run_joint_shared_encoder.py:42`, `code/run_joint_shared_encoder.py:31`, `code/run_joint_shared_encoder.py:64`, `code/run_joint_shared_encoder.py:358`
- _related: JNT-010_

#### `JNT-029` [LOW] Module-global DEVICE omits CUDA and is fixed at import
`code/run_joint_shared_encoder.py:66` · quality · status:live · effort:trivial · confidence:high
- **What:** The device is chosen at import as MPS-or-CPU only, so a CUDA-only host silently runs on CPU, diverging from the provenance get_device policy the other scripts use.
- **Evidence:** DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') (66). provenance get_device() does MPS->CUDA->CPU (m7_trimodal_run_provenance.py:94-101); run_joint_magnitude_ablation.py uses get_device (903).
- **Fix:** Use get_device(cfg) like the other scripts and do not fix the device at import time.
- **Fix at:** `code/run_joint_shared_encoder.py:66`

#### `JNT-032` [LOW] Leakage assert iterates the full train and val loaders with a per-label Python loop before training
`code/run_joint_shared_encoder.py:782-787` · performance · status:live · effort:small · confidence:high
- **What:** The leakage check reads every batch and loops over every label in Python before training starts; correct (unlike JNT-004) but an extra full data pass per seed.
- **Evidence:** run_one_seed loops 'for _,_,labels in train_loader: for lbl in labels: assert int(lbl.item()) in TRAIN_CLASSES' (782-787), and again for val_loader.
- **Fix:** Vectorize per batch (assert torch.isin(labels, torch.tensor(TRAIN_CLASSES)).all()) and/or check once at dataset-build time.
- **Fix at:** `code/run_joint_shared_encoder.py:782-787`
- _related: JNT-012_

#### `ALGN-019` [LOW] Unused imports hashlib and List
`code/run_ordinal_anchor.py:21-26` · quality · status:live · effort:trivial · confidence:high
- **What:** hashlib is imported (line 21) but never used here (0 hashlib. calls; _hash_params is imported from v2, where hashlib actually lives). typing.List (line 26) is also unused.
- **Evidence:** line 21 import hashlib (no hashlib. reference); line 26 from typing import Dict, List, Tuple (List unused)
- **Fix:** Remove the hashlib import and List from typing.
- **Fix at:** `code/run_ordinal_anchor.py:21`, `code/run_ordinal_anchor.py:26`

#### `CLE-006` [LOW] CSV column 'silhouette_score' stores only the visual silhouette
`code/run_rehearsal_transfer_eval_only.py:116` · quality · status:live · effort:trivial · confidence:high
- **What:** The column named silhouette_score is assigned transfer_metrics['sil_visual'] only (116); sil_audio is computed by the transfer eval but silently dropped. The generic column name implies a single/combined silhouette, obscuring that it is visual-only.
- **Evidence:** Line 116 `"silhouette_score": transfer_metrics["sil_visual"],`; the eval also returns sil_audio (printed at 89 but not persisted).
- **Fix:** Rename the column sil_visual and add a sil_audio column, matching the schema eval_transfer_fixed.py uses.
- **Fix at:** `code/run_rehearsal_transfer_eval_only.py:101`, `code/run_rehearsal_transfer_eval_only.py:116`

#### `CLE-007` [LOW] Unused imports and dead objects (random_split, dvs_collate_fn, trainer, buffer)
`code/run_rehearsal_transfer_eval_only.py:39-51` · dead-code · status:live · effort:trivial · confidence:high
- **What:** random_split (51) and dvs_collate_fn (17) are imported but never used. `buffer` (39) and `trainer` (40) are constructed but never used again -- only `model` is used for the transfer eval -- so both are dead objects (buffer exists solely to feed the unused trainer). These add import-time work and imply capabilities the script does not exercise.
- **Evidence:** grep: random_split count 1 (import only), dvs_collate_fn count 1 (import only); trainer referenced once (its assignment, line 40); buffer referenced twice (assignment 39 + passed to the unused trainer 40).
- **Fix:** Remove the unused imports and the buffer/trainer construction.
- **Fix at:** `code/run_rehearsal_transfer_eval_only.py:17`, `code/run_rehearsal_transfer_eval_only.py:39-40`, `code/run_rehearsal_transfer_eval_only.py:51`

#### `TST-012` [LOW] Vacuous assertions that are true by construction and validate nothing
`code/test_cross_modal_alignment.py:104-114` · quality · status:live · effort:small · confidence:high
- **What:** Two assertions cannot meaningfully fail. Line 105 `assert pen.item() >= 0` — orthogonal_penalty returns cross.pow(2).sum() (m7_trimodal_run_provenance.py:218-219), a sum of squares, which is always >= 0. Line 114 `assert loss_xmscl.item() > 0` — a SupCon loss is a mean of non-negative -log_prob terms and is essentially always positive. These checks only confirm the calls return a scalar; they give no evidence the quantities are correct. The scalar-ness checks (lines 104, 113 `ndim == 0`) are fine but weak.
- **Evidence:** code/test_cross_modal_alignment.py:105 `assert pen.item() >= 0`; orthogonal_penalty sum-of-squares at m7_trimodal_run_provenance.py:218-219; line 114 `assert loss_xmscl.item() > 0`.
- **Fix:** Replace tautological bounds with checks that could fail: e.g. penalty of orthogonalized vs correlated inputs differs as expected; SupCon loss decreases when identical-label embeddings are made more similar; or compare against a known reference value.
- **Fix at:** `code/test_cross_modal_alignment.py:104-114`
- _related: TST-013_

#### `TST-014` [LOW] Unused import numpy as np
`code/test_cross_modal_alignment.py:11` · dead-code · status:live · effort:trivial · confidence:high
- **What:** numpy is imported (line 11) but never used in the file (no `np.` reference). Dead import.
- **Evidence:** code/test_cross_modal_alignment.py:11 `import numpy as np`; grep for `np.` returns no usage.
- **Fix:** Remove the unused import.
- **Fix at:** `code/test_cross_modal_alignment.py:11`
- _related: TST-006_

#### `TST-015` [LOW] Imports placed mid-module, relying on the second sys.path hack
`code/test_cross_modal_alignment.py:123-124` · quality · status:live · effort:trivial · confidence:medium
- **What:** `from torch.utils.data import DataLoader` (line 123) and `from Dataloaders.paired_digit_loader import paired_digit_collate_fn` (line 124) appear in the middle of the executable body rather than with the other imports at the top. Dataset was already imported at line 12, so the mid-file DataLoader import is redundant-adjacent, and the Dataloaders import depends on the repo-root sys.path.insert at line 8 (the file has no code/ package). This is fragile and hurts readability; if the body is later refactored into functions these buried imports are easy to miss.
- **Evidence:** code/test_cross_modal_alignment.py:123 mid-file DataLoader import; line 124 Dataloaders import; repo-root path inserted at line 8; torch.utils.data.Dataset imported at top (line 12).
- **Fix:** Move both imports to the top import block; keep the sys.path.insert but consolidate imports so the dependency on it is explicit.
- **Fix at:** `code/test_cross_modal_alignment.py:123-124`

#### `TST-019` [LOW] Has a __main__ guard and main() (good) but still not a pytest test and asserts almost nothing
`code/test_ordinal_anchor_smoke.py:22-84` · quality · status:live · effort:medium · confidence:high
- **What:** Unlike the other two, this file correctly wraps its logic in main() (line 22) under an `if __name__ == '__main__'` guard (line 83), so importing it does not run the body. But it defines no test_* function (pytest collects 0 items — see TST-001), and its only 'assertion' is a print of 'Smoke test passed.' (line 80) with no assert statements at all: it just runs run_one_seed and prints results. Even if it ran, it would validate nothing programmatically. It is a manual demo script, not a test.
- **Evidence:** code/test_ordinal_anchor_smoke.py:22 def main(); line 83 __main__ guard; line 80 prints 'Smoke test passed.'; no `assert` anywhere in the file.
- **Fix:** Rename to a run_/demo_ script if it is meant to be run manually, OR convert to real pytest test_* functions with concrete assertions (e.g. on rdm keys, res dict fields) and dummy inputs so it can live in the validation loop.
- **Fix at:** `code/test_ordinal_anchor_smoke.py:22-84`
- _related: TST-001_

#### `TST-006` [LOW] Unused import numpy as np
`code/test_provenance_smoke.py:7` · dead-code · status:live · effort:trivial · confidence:high
- **What:** numpy is imported (line 7) but never referenced anywhere in the file (no `np.` usage). Dead import.
- **Evidence:** code/test_provenance_smoke.py:7 `import numpy as np`; grep for `np.` in the file returns no usage.
- **Fix:** Remove the unused import.
- **Fix at:** `code/test_provenance_smoke.py:7`
- _related: TST-014_

#### `TST-007` [LOW] DummyLoader hardcodes batch size 4 and tensor shapes, ignoring cfg.batch_size
`code/test_provenance_smoke.py:44-48` · quality · status:live · effort:trivial · confidence:medium
- **What:** DummyLoader.__iter__ hardcodes a batch dimension of 4 and fixed tensor shapes (lines 44-48) while cfg.batch_size is also set to 4 (line 27). The two are coupled only by coincidence: changing cfg.batch_size would not change the batches the loader yields, so the config knob is silently ignored. The magic shapes ([4,25,2,34,34] and [4,100,1,1,700]) are also duplicated from the real dataset specs without a named constant.
- **Evidence:** code/test_provenance_smoke.py:44 `x = torch.randn(4, 25, 2, 34, 34)`; line 46 `x = torch.randn(4, 100, 1, 1, 700)`; line 47 `y = torch.randint(0, self.n_classes, (4,))`; cfg.batch_size=4 at line 27.
- **Fix:** Parameterize DummyLoader by batch size (pass cfg.batch_size) or assert/comment that the fixed 4 is intentional; name the modality shapes as constants.
- **Fix at:** `code/test_provenance_smoke.py:36-48`


## INFO (4)

#### `DEP-017` [INFO] code/.gitignore is narrower than repo-root and ignores HDF5 globally
`code/.gitignore:16-21` · quality · status:live · effort:trivial · confidence:medium
- **What:** Observations, no defect. (1) code/.gitignore ignores only ICONS_M7/datasets/ (L17) and ICONS_M7/results/checkpoints/ (L21), plus *.pt/*.pth/*.ckpt/*.h5/*.hdf5 (L42-46); it does NOT ignore results CSVs or figure PNGs. Generated results are nonetheless protected because the repo-root .gitignore ignores ICONS_M7/, code/ICONS_M7 and results/. So there is redundant/overlapping coverage between the two files, and code/.gitignore ALONE would not shield a results CSV written from a different CWD. (2) The global *.h5/*.hdf5 rule (L45-46) would also exclude any legitimately tracked HDF5 fixture, should one ever be added.
- **Evidence:** code/.gitignore:16-21 (ICONS_M7/datasets/, ICONS_M7/results/checkpoints/) and :42-46 (*.pt,*.pth,*.ckpt,*.h5,*.hdf5) ; repo-root .gitignore:1-2,8-9 (ICONS_M7/, results/, code/ICONS_M7, ICONS_M7)
- **Fix:** Optional: consolidate ignore rules in one place; scope the HDF5 ignore to dataset directories if a tracked .h5 is ever needed. No action strictly required.
- **Fix at:** `code/.gitignore:16-21`, `code/.gitignore:45-46`

#### `ARCH-006` [INFO] Proof-of-concept variant search (tiny synthetic net); keep-archived
`code/archive/verify_tiny.py:1-216` · dead-code · status:archive · effort:trivial · confidence:high
- **What:** Smallest verify script: a 5->3->2 synthetic network sweeping four discretization variants A/D/H/I against Brian2/PyNN to find which matches IF_curr_exp exactly. Unlike the rest of the cluster it needs NO checkpoint and NO dataset (fully synthetic), so its ONLY blocker is the missing brian2/pyNN dependency (ARCH-010). It has no ConvFrontend. It documents WHY Variant D was chosen for verify_full.py.
- **Evidence:** synthetic weights lines 149-154; variant dict lines 179-184; no torch.load, no ConvFrontend.
- **Fix:** Retirement verdict KEEP-ARCHIVED: it is the methodological record of the variant selection and is self-contained. No action required beyond the shared dependency note (ARCH-010).
- **Fix at:** `code/archive/verify_tiny.py:1-216`
- _related: ARCH-010_

#### `FIG-029` [INFO] POSITIVE: final_evaluation.py has a __main__ guard (import-safe)
`code/final_evaluation.py:316-317` · quality · status:live · effort:trivial · confidence:high
- **What:** Unlike both figure scripts (FIG-002/FIG-009), final_evaluation.py wraps its work in main() behind `if __name__ == '__main__'` (316-317), so importing it for static inspection has no side effects (only module-level constants and spike_grad=surrogate.atan() at 24 evaluate). Noted as the correct pattern the figure scripts should adopt.
- **Evidence:** if __name__ == '__main__': main()
- **Fix:** No action. Use this as the template for guarding generate_figures*.py.
- **Fix at:** `code/final_evaluation.py:316-317`
- _related: FIG-002, FIG-009_

#### `JNT-015` [INFO] Seed set differs from repo policy and from the provenance default's 5th seed
`code/run_joint_magnitude_ablation.py:62` · naming · status:live · effort:trivial · confidence:high
- **What:** This ablation uses a 5-seed list whose 5th seed differs from the provenance default, and neither matches the 3-seed headline policy.
- **Evidence:** SEEDS=[42,123,456,789,1011] (62). Repo policy is [42,123,456] (CLAUDE.md); provenance default seeds use 101112 as the 5th (m7_trimodal_run_provenance.py:63) whereas both joint scripts use 1011 (run_joint_shared_encoder.py:85).
- **Fix:** Standardize the extra-seed set across the joint experiments and note it deviates from the 3-seed headline policy by design.
- **Fix at:** `code/run_joint_magnitude_ablation.py:62`, `code/run_joint_shared_encoder.py:85`

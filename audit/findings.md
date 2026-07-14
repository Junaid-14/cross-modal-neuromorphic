# All findings (human-readable) — generated 2026-07-15

316 canonical findings, grouped by severity. 15 duplicates omitted (see `findings.jsonl`). Machine-readable spine: `findings.jsonl`.


---
## CRITICAL (10)

#### `FIG-001` [CRITICAL] Both figure scripts write the identical 5 PNG filenames to the same dir - last run wins, can silently ship fabricated figures
`code/generate_figures.py:82-250` · research-integrity · status:live · effort:small · confidence:high
- **What:** generate_figures.py (RESULTS_DIR='./ICONS_M7/results', PLOT_DIR=results/plots, lines 10-11) and generate_figures_3seed.py (lines 28-29) both save the SAME five basenames (fig1_accuracy_comparison.png, fig2_forgetting.png, fig3_gate_trajectory.png, fig4_memory_efficiency.png, fig5_architecture_schematic.png) into the SAME directory. Legacy writes at 82/118/149/181/250; corrected writes at 179/222/276/326/376. Whichever runs last overwrites the other under identical names. generate_figures.py draws every data panel from hardcoded/fabricated numbers and needs no CSV, so running it last (or at all) silently replaces the corrected CSV-sourced figures with fabricated ones bearing the trusted filenames. Neither file namespaces or version-stamps its output.
- **Evidence:** PLOT_DIR / 'fig1_accuracy_comparison.png' in both files
- **Fix:** Retire/delete generate_figures.py (superseded by generate_figures_3seed.py). If kept, write to a clearly-marked separate dir (e.g. plots_LEGACY_FABRICATED/) with renamed outputs. Consolidate on one generator.
- **Fix at:** `code/generate_figures.py:10-11`, `code/generate_figures.py:82`, `code/generate_figures.py:118`, `code/generate_figures.py:149`, `code/generate_figures.py:181`, `code/generate_figures.py:250`, `code/generate_figures_3seed.py:28-29`, `code/generate_figures_3seed.py:179`, `code/generate_figures_3seed.py:222`, `code/generate_figures_3seed.py:276`, `code/generate_figures_3seed.py:326`, `code/generate_figures_3seed.py:376`
- _related: FIG-002, FIG-008_

#### `FIG-002` [CRITICAL] No __main__ guard: importing generate_figures.py renders and writes all 5 fabricated figures (needs no CSV)
`code/generate_figures.py:1-254` · safety · status:live · effort:small · confidence:high
- **What:** The entire body (hardcoded data block 25-48, figure code 50-252) executes at module import; there is no `if __name__ == '__main__'` guard (grep: 0 occurrences). Unlike generate_figures_3seed.py it reads NO CSV - all data is hardcoded - so `import generate_figures` (or any tool importing it to inspect) unconditionally creates ./ICONS_M7/results/plots/ (line 12) and writes fig1..fig5 built from fabricated numbers, with no error and no input required. This is the most dangerous import in the cluster: it always succeeds at producing fabricated research output.
- **Evidence:** grep __main__ -> 0 hits; savefig at 82/118/149/181/250
- **Fix:** Delete the file (preferred). If retained, wrap all figure code in main() behind `if __name__=='__main__'` so a module import produces no plots. See FIG-001 for the collision that makes this acute.
- **Fix at:** `code/generate_figures.py:1`, `code/generate_figures.py:50-252`
- _related: FIG-001, FIG-008_

#### `RUN-001` [CRITICAL] No __main__ guard: importing the module runs the full 6-run experiment and downloads/materializes datasets at import time
`code/m7_trimodal_run.py:699-711` · safety · status:live · effort:medium · confidence:high
- **What:** The RUN block executes at module top level: 'for seed in SEEDS: for condition in CONDITIONS: run_one_seed(...)'. There is no 'if __name__ == "__main__"' guard. Datasets are downloaded/materialized (109-180) and output directories created (41-42) at import. Any 'import m7_trimodal_run', static tool, or test collection therefore starts a 4-5 hour training run and a multi-GB download.
- **Evidence:** lines 707-711 at module scope: 'all_results = []' then 'for seed in SEEDS: for condition in CONDITIONS: result = run_one_seed(seed, condition)'; dataset loaders 119-179; 'for d in [...]: d.mkdir(...)' 41-42
- **Fix:** Move dataset loading, directory creation and the seed/condition loop into a main() guarded by 'if __name__ == "__main__": main()'. The provenance sibling already does this and can be mirrored.
- **Fix at:** `code/m7_trimodal_run.py:35-42`, `code/m7_trimodal_run.py:109-180`, `code/m7_trimodal_run.py:699-711`
- _related: RUN-017, RUN-032_

#### `RUN-002` [CRITICAL] Replay sample() filters TASK ids against a CLASS-keyed store and draws <=1 embedding per class
`code/m7_trimodal_run.py:380-392` · correctness · status:live · effort:medium · confidence:high
- **What:** _store is keyed by class label (0-19), but callers pass replay_task_ids=[1] (T2) and [1,2] (T3) meaning TASK numbers. sample() computes 'available = [c for c in task_ids if c in self._store]', so it keeps only the class(es) whose integer label equals a task id: T2 replay uses only class 1, T3 replay uses only classes 1 and 2. It then draws exactly one embedding per selected class, so a requested replay_batch of 32 yields at most 1-2 samples. self._task_labels (the map that would make this correct) is never consulted.
- **Evidence:** line 383 'available = [c for c in task_ids if c in self._store]'; line 386 'np.random.choice(available, size=min(batch_size, len(available)), replace=False)'; lines 388-391 one 'idx = np.random.randint(len(self._store[c]))' per class; callers 569 (task_ids=[1]) and 596 ([1,2])
- **Fix:** Select classes via _task_labels (classes whose task id is requested) and sample multiple embeddings per class up to batch_size, instead of one per class. Cloned in the provenance sibling.
- **Fix at:** `code/m7_trimodal_run.py:380-392`, `code/m7_trimodal_run.py:569`, `code/m7_trimodal_run.py:596`, `code/m7_trimodal_run_provenance.py:446-458`
- _related: RUN-003, RUN-004, RUN-030_

#### `RUN-003` [CRITICAL] populate() uses label_offset=0 for every task, so SHD classes overwrite N-MNIST in the class-keyed store
`code/m7_trimodal_run.py:361-378` · correctness · status:live · effort:medium · confidence:high
- **What:** _store is keyed by 'c = int(lbl.item()) + label_offset' and populate() is called with label_offset=0 for T1 (nmnist, classes 0-9) and T2 (shd, classes 0-19). SHD classes 0-9 overwrite the stored N-MNIST embeddings under the same keys. By T3, replay 'of T1 and T2' contains no N-MNIST embeddings at all, so T1 gets zero replay protection during T3. Combined with RUN-002 the replay signal is both misrouted and largely destroyed.
- **Evidence:** line 369 'c = int(lbl.item()) + label_offset'; call sites line 553 '...task_id=1, label_offset=0' and line 584 '...task_id=2, label_offset=0'
- **Fix:** Key the store so class collisions across tasks are impossible (e.g. by (task_id, class) or a per-task offset) and make sample() task-aware (RUN-002). A real per-task label_offset is the minimal fix. Cloned in the provenance sibling.
- **Fix at:** `code/m7_trimodal_run.py:361-378`, `code/m7_trimodal_run.py:553`, `code/m7_trimodal_run.py:584`, `code/m7_trimodal_run_provenance.py:427-444`, `code/m7_trimodal_run_provenance.py:1011`, `code/m7_trimodal_run_provenance.py:1079`
- _related: RUN-002, RUN-030_

#### `PROV-001` [CRITICAL] EmbeddingReplayBuffer replay is fundamentally broken: task_ids used as class keys, label_offset=0 collides tasks, wrong head/labels
`code/m7_trimodal_run_provenance.py:446-458` · correctness · status:live · effort:medium · confidence:high
- **What:** populate() keys the store by raw class label c=int(lbl.item())+label_offset (435) with label_offset=0 at BOTH call sites (1011, 1079), so SHD classes 0-9 OVERWRITE N-MNIST classes 0-9 in the shared _store dict (self._store[c]=... at 443). sample(task_ids) then treats task_ids as store KEYS, not task ids: available=[c for c in task_ids if c in self._store] (449). Call sites pass replay_task_ids=[1] (1050) and [1,2] (1102), so replay draws only class-index 1 (and 2) embeddings from whatever task last wrote those keys, never a task's worth of samples and never N-MNIST (it was overwritten). The sampled labels are the class indices (457) and are fed to forward_from_embeddings(embs, task_name) through the CURRENT task's head (611-615), so replay optimizes the wrong head against wrong labels. _task_labels[c]=task_id (444) records the true task but sample() never consults it. Net effect: 'embedding replay' rehearses one-to-two wrong-task embeddings through the current head, so it cannot prevent forgetting of prior tasks. This is the hub imported by 19 files, so the defect propagates everywhere.
- **Evidence:** key c=int(lbl.item())+label_offset (435); self._store[c]=torch.stack(samples) (443); self._task_labels[c]=task_id (444, unused by sample); available=[c for c in task_ids if c in self._store] (449); label_offset=0 (1011,1079); replay_task_ids=[1] (1050), [1,2] (1102); forward_from_embeddings(replay_embs, task_name) (612-614). The CORRECT pattern is present in the same file: RawInputReplayBuffer.sample filters by task_names (498-508).
- **Fix:** Key the buffer per task so tasks cannot collide (offset labels by task, or nest by task_id); in sample() select stored classes via _task_labels==requested task_id (mirror RawInputReplayBuffer.sample, 498); replay each sample through its OWN task head with its own labels, not the current task_name. Same defect and the same fix apply to the orphan clone code/m7_trimodal_run.py.
- **Fix at:** `code/m7_trimodal_run_provenance.py:435`, `code/m7_trimodal_run_provenance.py:444`, `code/m7_trimodal_run_provenance.py:446-458`, `code/m7_trimodal_run_provenance.py:611-615`, `code/m7_trimodal_run_provenance.py:1011`, `code/m7_trimodal_run_provenance.py:1050`, `code/m7_trimodal_run_provenance.py:1079`, `code/m7_trimodal_run_provenance.py:1102`, `code/m7_trimodal_run.py:380-392`, `code/m7_trimodal_run.py:553`, `code/m7_trimodal_run.py:569`, `code/m7_trimodal_run.py:584`, `code/m7_trimodal_run.py:596`
- _related: PROV-003_

#### `SNN-001` [CRITICAL] random_init_baseline silently reshuffles training data via the global RNG
`code/m7_trimodal_run_provenance.py:804-823` · numerics · status:live · effort:small · confidence:high
- **What:** After per-seed seeding (L964-965) and construction of the real model (L971), random_init_baseline calls torch.manual_seed(cfg.seed+1000+s) and builds 3 more models, leaving the GLOBAL torch generator at a state that depends on the model's parameter count. Training is not re-seeded before T1 (L997), and DataLoader(shuffle=True) draws its per-epoch shuffle from that global generator. So training data order - and every trained accuracy - is coupled to the baseline routine: change n_seeds, shared_dim, num_patterns, or add a layer and the main result moves with no other edit. Inherited by run_cl_confirmation_fixed.py (call L212) and run_cl_rehearsal.py (call L187).
- **Evidence:** def L804-823; torch.manual_seed L812; called L983; no re-seed before T1 train L997
- **Fix:** Snapshot/restore RNG around the baseline (state=torch.get_rng_state() ... torch.set_rng_state(state)), or build baseline models with a local torch.Generator, and re-seed torch.manual_seed(cfg.seed) immediately before T1.
- **Fix at:** `code/m7_trimodal_run_provenance.py:804-823`, `code/m7_trimodal_run_provenance.py:980-997`
- _related: SNN-007_

#### `ALGN-001` [CRITICAL] Held-out training path is doubly broken: CE reduced before mask, and 6 anchors vs labels 0-9
`code/run_correspondence_anchor_v2.py:311-324` · correctness · status:live · effort:small · confidence:high
- **What:** train_correspondence uses ce=nn.CrossEntropyLoss() (default reduction='mean', line 287). In the held-out config, run_one_seed builds corr_ho with num_anchors=6 (line 575) so model.anchors is (6, anchor_dim) and logits_vis/logits_aud have 6 columns (315-316), but the training set from build_paired_feature_dataset contains labels 0-9 (loops range(10), lines 251,261). ce(6-col logits, target=9) raises RuntimeError 'Target out of bounds', so held-out training crashes every seed. Even if it did not crash, the mask is applied AFTER the mean reduction: loss is already a scalar, so loss=loss*mask.float() broadcasts to (B,) and loss.sum()/mask.sum() collapses back to the same scalar, making the held-out exclusion a no-op. main() wraps run_one_seed in try/except (669-672), so the crash is swallowed and NO per-seed CSV rows are ever written; the script's central 'held-out 6-9 retrieval' deliverable never runs. The paper's held-out-retrieval-at-chance numbers cannot have been produced by this code path.
- **Evidence:** line 287 ce=nn.CrossEntropyLoss(); lines 318-324 loss_vis=ce(logits_vis,labels); loss=loss_vis+loss_aud; if len(held_out_classes)>0: loss=loss*mask.float(); loss=loss.sum()/(mask.sum().clamp(min=1)); line 575 corr_ho=...num_anchors=6; lines 251,261 build_paired_feature_dataset loops range(10) so labels are 0-9
- **Fix:** Filter logits and labels by the keep-mask BEFORE calling ce (or set reduction='none' and average only the kept per-sample losses). The held-out config must train on 0-5 samples only so targets are in [0,5] and match the 6 anchors; masking the data also removes the target-out-of-bounds crash.
- **Fix at:** `code/run_correspondence_anchor_v2.py:287`, `code/run_correspondence_anchor_v2.py:311-324`, `code/run_correspondence_anchor_v2.py:575-579`
- _related: ALGN-015_

#### `ALGN-015` [CRITICAL] Stage 1 inherits the broken held-out training from v2
`code/run_ordinal_anchor.py:51-58` · correctness · status:live · effort:small · confidence:high
- **What:** run_ordinal_anchor imports train_correspondence and compute_held_out_metrics from run_correspondence_anchor_v2 (lines 51-58) and drives the identical held-out path: corr_ho = OrdinalCorrespondenceModel(..., num_anchors=6) (483-485) trained via train_correspondence(..., held_out_classes=(6,7,8,9)) (487-497) on labels 0-9. This hits the same CE-target-out-of-bounds crash plus no-op mask described in ALGN-001, and main() swallows it via try/except (680-684). Stage 1 (the ordinal held-out retrieval that is the script's entire reason to exist) never completes.
- **Evidence:** lines 51-58 from run_correspondence_anchor_v2 import (... train_correspondence, compute_held_out_metrics); lines 483-497 corr_ho num_anchors=6 + train_correspondence held_out_classes=(6,7,8,9); build_paired_feature_dataset (imported) yields labels 0-9
- **Fix:** Fix train_correspondence per ALGN-001; then Stage 1 here runs. No separate change needed once the shared helper is corrected.
- **Fix at:** `code/run_ordinal_anchor.py:483-497`, `code/run_correspondence_anchor_v2.py:311-324`
- _related: ALGN-001_

#### `RA-001` [CRITICAL] Hardcoded accuracies written into a results CSV
`code/run_rehearsal_transfer_eval_only.py:104-118` · research-integrity · status:live · effort:small · confidence:high
- **What:** The result dict written to cl_rehearsal_results.csv mixes live transfer metrics with hardcoded accuracy literals (T1_after_T1=97.50, T2_after_T2=72.26, T3_after_T3=76.89, T1_after_T2=95.81, T1_after_T3=95.65, T2_after_T3=73.14) and derives every forgetting field from those literals. This manufactures the exact artifact RULE 1 designates as the source of truth; downstream tools would ingest fabricated accuracies as measured. The checkpoint it claims to evaluate (final_dvs_dvs.pt) does not exist.
- **Evidence:** lines 106-107 accuracy literals; 108-112 forgetting arithmetic on them; verdict UNSOURCED
- **Fix:** Delete the hardcoded accuracy/forgetting fields; write only columns computed live this run, or gate the whole CSV write behind a real evaluation.
- **Fix at:** `code/run_rehearsal_transfer_eval_only.py:106-112`
- _confirmed by CLE-002; related: RA-012_


---
## HIGH (69)

#### `ARCH-015` [HIGH] Held-out-class masking is a no-op: held-out digits are still trained on
`code/archive/run_correspondence_anchor.py:412-419` · correctness · status:archive · effort:small · confidence:high
- **What:** train_correspondence uses ce = nn.CrossEntropyLoss() with default reduction='mean', so loss_vis/loss_aud/loss are SCALARS (lines 412-414). The held-out masking then does loss = loss * mask.float() (scalar broadcast to a per-sample vector) then loss.sum()/mask.sum() (lines 418-419), which algebraically returns the original scalar mean over ALL samples — the held-out classes were never excluded. Compounding this, build_paired_feature_dataset accepts held_out_classes (line 299) but never uses it, so held-out samples are also present in the training set. Net effect: the step-7 'held-out-class' test (digits 8,9, lines 616-628) evaluates alignment on classes the model was actually trained on, invalidating that result and the derived 'discovered vs engineered' verdict (line 651).
- **Evidence:** nn.CrossEntropyLoss() default mean at line 377; scalar loss lines 412-414; scalar*mask then sum/sum lines 418-419; held_out_classes param unused in build_paired_feature_dataset (declared line 299); held-out run at lines 622-628.
- **Fix:** To actually hold classes out: use reduction='none' to get per-sample losses BEFORE masking, or filter held-out samples out of the dataset in build_paired_feature_dataset. As-is the held-out numbers are meaningless. For retirement this is a primary reason v1 was superseded (ARCH-001) — confirm v2 fixed it.
- **Fix at:** `code/archive/run_correspondence_anchor.py:377`, `code/archive/run_correspondence_anchor.py:412-419`, `code/archive/run_correspondence_anchor.py:299`
- _related: ARCH-001, ARCH-017_

#### `DEAD-018` [HIGH] benchmark_mps_real.py hardcodes another user's machine path and an MPS-only device
`code/benchmark_mps_real.py:6-15` · dependencies · status:dead · effort:trivial · confidence:high
- **What:** L6 does sys.path.insert(0, /Users/agent1/Documents/neuromorphic/m7-trimodal-cl) - a foreign absolute path from a different developer's machine that does not exist here; the subsequent legacy imports (L9-11) depend on it and will ModuleNotFoundError on this repo. L15 hardcodes DEVICE=torch.device('mps') with no CUDA/CPU fallback, so it also crashes on any non-Apple-Silicon host. The file cannot run anywhere as written.
- **Evidence:** sys.path.insert(0, '/Users/agent1/Documents/neuromorphic/m7-trimodal-cl') L6; DEVICE=torch.device('mps') L15.
- **Fix:** Delete. If ever revived, derive paths relative to the repo and auto-select the device MPS then CUDA then CPU.
- **Fix at:** `code/benchmark_mps_real.py:6`, `code/benchmark_mps_real.py:15`
- _related: DEAD-017_

#### `DEAD-010` [HIGH] convert_mps.py overwrites code/m7_mps.py on import (no __main__ guard)
`code/convert_mps.py:7-174` · safety · status:dead · effort:small · confidence:high
- **What:** All work runs at module scope: it reads m7_raw.py (L7-8) and unconditionally writes/overwrites code/m7_mps.py (L173-174) with no guard, no backup, and no confirmation. Any 'import convert_mps' (for example a tool statically importing code/ modules) silently regenerates m7_mps.py - and because the current m7_raw.py has uncommented !pip lines, the regenerated file would differ from the current one, worsening the drift in DEAD-004. Violates the repo rule against overwriting a file without showing a diff.
- **Evidence:** no __main__ (grep returns 0); read at L7-8; open('m7_mps.py','w') + f.write(code) at L173-174, all at column 0.
- **Fix:** If retained, wrap the body in if __name__=='__main__': and write to a temp path / print a diff first. Preferred: delete the file.
- **Fix at:** `code/convert_mps.py:7-8`, `code/convert_mps.py:173-174`
- _related: DEAD-009, DEAD-004_

#### `CLE-008` [HIGH] Missing config_nmnist.json crashes main() on the normal path
`code/eval_transfer_fixed.py:60-63` · safety · status:live · effort:small · confidence:high
- **What:** main() opens CKPT_DIR/config_nmnist.json with no existence check (61-62). CKPT_DIR = results/cl_confirmation_fixed/checkpoints/C_split_128_seed42 does not exist in the repo, so `open()` raises FileNotFoundError before any checkpoint is even considered. This is the first statement in main(), so the script cannot run at all today.
- **Evidence:** Lines 61-63; results/cl_confirmation_fixed MISSING on disk.
- **Fix:** Guard the config path with a clear existence check and actionable error (which run/seed must be present first); do not assume the results tree exists.
- **Fix at:** `code/eval_transfer_fixed.py:60-64`
- _related: CLE-013_

#### `CLE-009` [HIGH] Empty-results DataFrame raises KeyError before the 'in' guard can protect it
`code/eval_transfer_fixed.py:126` · correctness · status:live · effort:small · confidence:high
- **What:** If both checkpoints are missing but the config loaded (the 'only checkpoints missing' path), the loop skips every stage and results stays []. Then df = pd.DataFrame([]) has no columns, and the verdict line evaluates the ternary condition `"post_T2" in df["stage"].values` first -- df["stage"] raises KeyError: 'stage' before the membership test or the `else 0` fallback can run. So the intended empty-case fallback never fires; the script crashes.
- **Evidence:** Line 120 `df = pd.DataFrame(results)`; line 126 `... if "post_T2" in df["stage"].values else 0`. With results==[] the DataFrame has no 'stage' column.
- **Fix:** Handle the empty case before indexing: e.g. `if df.empty or "post_T2" not in df["stage"].values: post_t2_v2a = None` and branch the verdict on None (report 'no data' rather than a number).
- **Fix at:** `code/eval_transfer_fixed.py:120-137`
- _related: CLE-010_

#### `FIG-019` [HIGH] Accuracy reported without spike rate / sparsity - violates 'accuracy never travels alone'
`code/final_evaluation.py:286-291` · research-integrity · status:live · effort:medium · confidence:high
- **What:** The PyTorch baseline accuracy (pt_acc, computed 251, printed 252/283) and the SpiNNaker-sim accuracies (spinnaker_accs mean/std/CI, 286-289) are reported with NO spike-rate or sparsity metric. The simulator accumulates head spike counts (s3_acc, 147/179) and tracks all-zero samples (zero_count/total_zero_count, 182/271/291) but never computes or reports network spike rate or sparsity. Timestep count is partial - DURATION_MS=100 (100 steps) is printed for the sim (202/285) - but the PyTorch baseline's timestep count (T from data, ~25 for N-MNIST) is not reported. Per research-integrity.md an SNN accuracy without its spike-rate/sparsity and timestep cost is the part of the result that flatters us.
- **Evidence:** pt_acc / spinnaker_accs printed with no spikes/sparsity
- **Fix:** Report mean spikes/neuron (or spike rate) and sparsity for both sim and baseline alongside accuracy, and state the timestep count for each; the sim already has s3_acc and could accumulate per-layer spike totals cheaply.
- **Fix at:** `code/final_evaluation.py:251-252`, `code/final_evaluation.py:282-291`
- _related: FIG-018_

#### `FIG-020` [HIGH] 'SpiNNaker-deployed network' overclaim and unsourced Brian2-equivalence claim in the file's prose
`code/final_evaluation.py:303-308` · research-integrity · status:live · effort:small · confidence:high
- **What:** The docstring (line 3) says 'the corrected SpiNNaker discrete-time simulator', but the printed HONEST STATEMENT escalates to 'the final, locked evaluation of the SpiNNaker-deployed network' (303) and claims it 'uses the EXACT IF_curr_exp parameters from build_spinnaker_network() and was verified against raw Brian2 to produce identical argmax predictions on 20 stratified reference samples' (306-308). Per CLAUDE.md/spinnaker.md nothing has run on a board (HARDWARE_ACCESS_CHECKLIST entirely unchecked); this is a CPU discrete-time SIMULATOR, so 'SpiNNaker-deployed' conflates simulation with deployment (Mapping != inference != real hardware). build_spinnaker_network() is neither defined nor imported here, and no Brian2 comparison runs in this file (Brian2 not imported), so the '20 stratified reference samples' equivalence is asserted as fact without any in-file/in-repo artifact. The corresponding verify_brian2_raw.py lives in code/archive/, not here.
- **Evidence:** 'evaluation of the SpiNNaker-deployed network' + 'verified against raw Brian2 ... 20 stratified reference samples'
- **Fix:** Reword to 'discrete-time simulator of the mapped parameters' and drop 'deployed'. Either reproduce the Brian2 comparison and cite its output artifact, or remove the specific 20-samples/identical-argmax claim; do not state cross-simulator equivalence as fact without a sourced artifact.
- **Fix at:** `code/final_evaluation.py:3`, `code/final_evaluation.py:302-312`
- _related: FIG-027_

#### `FIG-003` [HIGH] Hardcoded/fabricated accuracy values presented as M7 results (RULE 1)
`code/generate_figures.py:30-32` · research-integrity · status:live · effort:small · confidence:high
- **What:** acc_t1=[97.30, 97.37], acc_t2=[80.26, 81.98], acc_t3=[84.85, 84.47] are hardcoded in code (30-32), commented 'Accuracies [no_replay, replay]'. They are rendered as bar heights with '%.1f%%' labels (63-70) under title 'M7 Tri-Modal Continual Learning: Accuracy by Condition' (73). No CSV backs them - the repo has no results CSV at all - so under RULE 1 these are fabricated numbers shipped as a publication figure. No seed count, no mean/std, no test-vs-val disclosure.
- **Evidence:** acc_t1=[97.30, 97.37] hardcoded in code
- **Fix:** Do not draw accuracy from literals. Delete this script (FIG-001/002); use generate_figures_3seed.py which sources accuracy from cl_trimodal_results.csv.
- **Fix at:** `code/generate_figures.py:30-32`, `code/generate_figures.py:55-70`
- _related: FIG-008_

#### `FIG-004` [HIGH] Hardcoded/fabricated forgetting values (RULE 1 / RULE 5)
`code/generate_figures.py:36-37` · research-integrity · status:live · effort:small · confidence:high
- **What:** forget_noreplay=[-0.01, 0.01, -0.09] and forget_replay=[0.00, 0.00, 0.00] are hardcoded (36-37) and plotted (91-107) under title 'M7: Catastrophic Forgetting Across Task Transitions' (110). The all-zero replay row rendered as '+0.00' visually asserts perfectly eliminated forgetting - the exact RULE 5 overclaim (~0 pp, NOT 100 percent reduction). No CSV backs these.
- **Evidence:** forget_replay = [0.00, 0.00, 0.00] hardcoded in code
- **Fix:** Remove literal forgetting arrays; source from CSV via the corrected script. Avoid 'Catastrophic' in a title paired with hardcoded zeros.
- **Fix at:** `code/generate_figures.py:36-37`, `code/generate_figures.py:86-115`
- _related: FIG-008_

#### `FIG-005` [HIGH] Fabricated 'simulated' gate trajectories drive the paper's central novelty claim
`code/generate_figures.py:39-48` · research-integrity · status:live · effort:small · confidence:high
- **What:** gate_noreplay_shd and gate_replay_shd are hardcoded literal arrays (39-48) with the comment 'simulated from log data' (39). They are plotted as the gate-specialisation trajectory (122-146) and annotated 'Replay triggers HGRN specialisation' (144-146) - the paper's core novelty claim. Drawing that claim from invented literals is the most damaging fabrication here. contested-facts.md records this exact figure was previously fabricated; m7_trimodal_run.py now persists the real per-epoch gate split to gate_trajectory.csv, which the corrected script reads.
- **Evidence:** comment 'simulated from log data' at line 39
- **Fix:** Delete; use generate_figures_3seed.py Figure 3, which renders from gate_trajectory.csv or skips loudly.
- **Fix at:** `code/generate_figures.py:39-48`, `code/generate_figures.py:122-146`
- _related: FIG-001, FIG-008_

#### `FIG-006` [HIGH] Mislabeled 'single-task' reference lines are legacy bi-modal numbers overlaid on a tri-modal plot
`code/generate_figures.py:78-79` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** ax.axhline(y=97.68, label='N-MNIST single-task') and ax.axhline(y=82.16, label='SHD single-task') (78-79) hardcode two numbers that (per the corrected script's docstring and contested-facts.md) are the LEGACY BI-MODAL paper's M3/M2 results - a different experiment - mislabeled 'single-task' and drawn on a tri-modal accuracy plot. Compounding bug: ax.legend() is called at line 76 BEFORE these axhlines, so their label= never enters the legend - two dashed lines appear on the figure with no explanation. Neither value is in any CSV.
- **Evidence:** axhline(y=97.68 ...) / axhline(y=82.16 ...) after ax.legend() at 76
- **Fix:** Remove the reference lines (as the corrected script did). If single-task baselines are wanted, run them and read from CSV; build the legend after all artists are added.
- **Fix at:** `code/generate_figures.py:76`, `code/generate_figures.py:78-79`
- _related: FIG-008_

#### `FIG-007` [HIGH] Hardcoded memory footprint [80,150,4] MB and invented 20x/37x compression
`code/generate_figures.py:157` · research-integrity · status:live · effort:small · confidence:high
- **What:** memory=[80, 150, 4] MB is hardcoded (157) with no source, plotted as the replay-buffer footprint (160-165), and annotated '20x smaller'/'37x smaller' (173-178) - ratios that are simply 80/4 and 150/4 of the invented values. No measurement or CONFIG derivation. The corrected script replaces this with a CONFIG-derived calculation whose arithmetic is printed.
- **Evidence:** memory = [80, 150, 4]  # MB
- **Fix:** Delete; use generate_figures_3seed.py Figure 4 (derives bytes from CONFIG and prints the arithmetic, labeled 'derived from CONFIG, not measured').
- **Fix at:** `code/generate_figures.py:157`, `code/generate_figures.py:173-178`
- _related: FIG-008, FIG-012_

#### `FIG-008` [HIGH] generate_figures.py is superseded and should be retired/deleted
`code/generate_figures.py:1-254` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** Every data-bearing panel in this file is fabricated (FIG-003..007), it runs on import with no guard (FIG-002), and it collides output filenames with the corrected generator (FIG-001). It is the LEGACY generator; generate_figures_3seed.py is its corrected replacement. Keeping a fabricated-data script in the tree under near-identical output names is a standing research-integrity liability.
- **Evidence:** whole-file legacy generator; corrected replacement exists
- **Fix:** Delete code/generate_figures.py (or move to code/archive/ with a FABRICATED marker and a non-colliding output path). Consolidate on generate_figures_3seed.py.
- **Fix at:** `code/generate_figures.py:1`
- _related: FIG-001, FIG-002, FIG-003, FIG-004, FIG-005, FIG-006, FIG-007_

#### `RA-002` [HIGH] Hardcoded per-task accuracies and forgetting arrays drive Figures 1-2
`code/generate_figures.py:30-37` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** acc_t1=[97.30,97.37], acc_t2=[80.26,81.98], acc_t3=[84.85,84.47], forget_noreplay/forget_replay are literal arrays plotted as this experiment's results. This is the pre-2026-07-14 generator that generate_figures_3seed.py documents as fabricated and replaced, but it was never removed and still writes PNGs.
- **Evidence:** lines 30-37 literal arrays; verdict UNSOURCED
- **Fix:** Delete the file (superseded by generate_figures_3seed.py); at minimum delete these literals and read from a CSV.
- **Fix at:** `code/generate_figures.py:30-37`, `code/generate_figures.py:1-254`
- _related: RA-003, RA-004, RA-006, FIG-collision_

#### `RA-003` [HIGH] Legacy bi-modal numbers mislabelled as 'single-task' reference lines
`code/generate_figures.py:78-79` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** axhline(y=97.68,'N-MNIST single-task') and axhline(y=82.16,'SHD single-task') overlay two constants on a tri-modal plot. Cross-referencing m7_mps.py:2286-2287 confirms these are legacy bi-modal Paper-2 M3/M2 results, not single-task baselines of this experiment. The label asserts a different experiment's numbers as this one's.
- **Evidence:** lines 78-79; verdict CONTRADICTED
- **Fix:** Delete both lines (already removed in the 3seed version); if single-task baselines are wanted, run them into the CSV.
- **Fix at:** `code/generate_figures.py:78-79`
- _related: RA-002_

#### `RA-004` [HIGH] Gate trajectory 'simulated from log data' plotted as measured Figure 3
`code/generate_figures.py:39-48` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** The comment on line 39 reads 'simulated from log data'; gate_noreplay_shd/gate_replay_shd are hardcoded hop/hgrn weight sequences plotted as Figure 3 - the paper's central gate-specialisation claim. A fabricated trajectory presented as the measured gate behaviour.
- **Evidence:** lines 39-48 and Figure 3 block 122-151; verdict UNSOURCED (fabricated)
- **Fix:** Delete; the corrected path (generate_figures_3seed.py:233-278) reads a real gate_trajectory.csv or skips the figure.
- **Fix at:** `code/generate_figures.py:39-48`, `code/generate_figures.py:122-151`
- _related: RA-002_

#### `DEAD-003` [HIGH] m7_mps.py is a dead machine-generated MPS clone that runs a training smoke-test on import
`code/m7_mps.py:1-3900` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** MPS port of m7_raw.py produced by convert_mps.py; re-defines the same 7 core classes plus legacy zoo and ablations (class defs L633-2944), superseded by code/m7_trimodal_run_provenance.py (which already auto-selects MPS). No __main__ guard: at module scope (L560-576) it seeds, builds M7_ContinualAdaptiveModel and ContinualLearner, and calls trainer.train_task(...), preceded by tonic dataset loading - so 'import m7_mps' downloads datasets and starts training (the same import-time hazard as the live m7_trimodal_run.py trap).
- **Evidence:** module-scope execution L560-576: torch.manual_seed L560-561, model_smoke=M7_ContinualAdaptiveModel(...) L569, trainer=ContinualLearner(...) L572, result_smoke=trainer.train_task(...) L576; grep '__main__' returns 0.
- **Fix:** Delete - it is a regenerable derived artifact. Never import. If an MPS run is needed, use m7_trimodal_run_provenance.py.
- **Fix at:** `code/m7_mps.py:1-3900`
- _related: DEAD-001, DEAD-004, DEAD-009_

#### `DEP-004` [HIGH] Third-party 'seaborn' imported at module top level but absent from requirements.txt
`code/m7_mps.py:43` · dependencies · status:live · effort:trivial · confidence:high
- **What:** `import seaborn as sns` runs at module import time (unguarded, top level) in four live scripts. seaborn is not declared in code/requirements.txt. After a clean install, running or importing any of these modules raises ImportError before any work begins.
- **Evidence:** code/m7_mps.py:43, code/m7_raw.py:43, code/m7_smoke_test.py:43, code/m7_trimodal_cl.py:43 each: import seaborn as sns
- **Fix:** Add seaborn (pip name: seaborn) to code/requirements.txt with a version that supports py3.11.
- **Fix at:** `code/requirements.txt:13`, `code/m7_mps.py:43`, `code/m7_raw.py:43`, `code/m7_smoke_test.py:43`, `code/m7_trimodal_cl.py:43`
- _related: DEP-005_

#### `DEP-005` [HIGH] Third-party scikit-learn (import name sklearn) imported but absent from requirements.txt
`code/m7_mps.py:70-71` · dependencies · status:live · effort:trivial · confidence:high
- **What:** `from sklearn.manifold import TSNE` and `from sklearn.metrics import (...)` run at module top level in four live scripts. The pip package is scikit-learn (import name sklearn) and is not declared. Clean-install import of any of these modules raises ImportError. Naming trap: the requirement must read scikit-learn, not sklearn.
- **Evidence:** code/m7_mps.py:70-71, code/m7_raw.py:62-63, code/m7_smoke_test.py:70-71, code/m7_trimodal_cl.py:70-71 (from sklearn.manifold import TSNE ; from sklearn.metrics import (...))
- **Fix:** Add scikit-learn (NOT sklearn) to code/requirements.txt with a py3.11-compatible version.
- **Fix at:** `code/requirements.txt:13`, `code/m7_mps.py:70-71`, `code/m7_raw.py:62-63`, `code/m7_smoke_test.py:70-71`, `code/m7_trimodal_cl.py:70-71`
- _confirmed by ALGN-005, CLE-033; related: DEP-004_

#### `DEP-006` [HIGH] 'huggingface_hub' absent from requirements.txt so RULE 2 backups silently skip
`code/m7_mps.py:2170` · dependencies · status:live · effort:small · confidence:high
- **What:** `from huggingface_hub import HfApi` sits inside upload_to_hf() in three live scripts, and the surrounding block is wrapped in `try: ... except Exception as e: print('HF upload skipped ...')`. On a clean install (no huggingface_hub) the ImportError is swallowed: the function prints a skip message and the run continues with NO backup. RULE 2 mandates backing up to HuggingFace after every seed and the team has already lost data once, so a missing pin turns every backup into a silent no-op. Not a crash, but a research-integrity/data-loss hazard. (Note: the canonical runners m7_trimodal_run.py and m7_trimodal_run_provenance.py contain no HF code at all.)
- **Evidence:** code/m7_mps.py:2170 from huggingface_hub import HfApi ; code/m7_mps.py:2179 except Exception as e: print('  HF upload skipped ...') ; same pattern code/m7_raw.py:2147 and code/m7_trimodal_cl.py:2097
- **Fix:** Add huggingface_hub to code/requirements.txt. Additionally narrow the except so a missing backup dependency is loud (do not swallow ImportError silently), otherwise a broken RULE 2 backup is invisible.
- **Fix at:** `code/requirements.txt:13`, `code/m7_mps.py:2169-2180`, `code/m7_raw.py:2147`, `code/m7_trimodal_cl.py:2097`

#### `DEAD-008` [HIGH] m7_nmnist_smoke_test.py is a dead standalone smoke test that downloads N-MNIST and trains on import
`code/m7_nmnist_smoke_test.py:1-472` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** Hand-written (not a notebook dump) single-file N-MNIST smoke test that re-defines the core classes: SupervisedContrastiveLoss L121, ModernHopfieldLayer L138, ImprovedHGRNGate L152, SNN_Backbone L174, M7_ContinualAdaptiveModel L222 (audio backbone skipped, L228-229), EmbeddingReplayBuffer L267, ContinualLearner L318. No __main__ guard: at module scope it constructs tonic.datasets.NMNIST (L95-96, which downloads if absent) and runs 10-epoch training (L424-435), so 'import m7_nmnist_smoke_test' downloads and trains. Superseded by m7_trimodal_run_provenance.py. Minor extra issues (all secondary to retirement): the QUICK_SUBSET torch.randperm at L100 runs before torch.manual_seed(42) at L419 (non-deterministic subset), and results are printed without spike-rate/sparsity (accuracy-never-alone) - noted, not separately filed.
- **Evidence:** dataset ctor L95-96; QUICK_SUBSET randperm L99-100; trainer.train_task L430-435; grep '__main__' returns 0.
- **Fix:** Delete. Never import.
- **Fix at:** `code/m7_nmnist_smoke_test.py:1-472`
- _related: DEAD-007_

#### `DEAD-002` [HIGH] m7_raw.py has uncommented !pip shell-magic lines - it is not valid Python (SyntaxError)
`code/m7_raw.py:2-23` · dependencies · status:dead · effort:trivial · confidence:high
- **What:** Lines 2-4 and 17-23 are raw IPython/Jupyter shell-magic ('!pip install ...') left uncommented. As a .py module this is a SyntaxError: CPython cannot parse, import, or run the file at all. Confirms it is an unconverted notebook cell dump. This is unique among the dead group - the machine-converted sibling m7_mps.py has these exact lines commented out ('# !pip ...' at its L2-4/L17-23).
- **Evidence:** L2 '!pip install --upgrade jinja2 -q', L3-4, L17-19, L22-23 all begin with '!'; contrast m7_mps.py L2-4/L17-23 which are commented.
- **Fix:** None if the file is deleted (recommended, see DEAD-001). If retained, every '!' line must be commented. Never try to import it to 'check' it.
- **Fix at:** `code/m7_raw.py:2-4`, `code/m7_raw.py:17-23`
- _related: DEAD-001_

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

#### `RUN-004` [HIGH] Replayed embeddings pass through the CURRENT task head, not the originating task head
`code/m7_trimodal_run.py:453` · correctness · status:live · effort:medium · confidence:high
- **What:** During task T, replay is scored via 'forward_from_embeddings(replay_embs, task_name)' where task_name is the current task. Old-task embeddings are pushed through the current head and trained against old-class integer labels. This does nothing to preserve the previous task's head (the purpose of rehearsal) and instead trains the current head on foreign-class targets. forward_from_embeddings has no way to select the originating head.
- **Evidence:** line 453 'replay_logits = self.model.forward_from_embeddings(replay_embs, task_name)'; def 344-349 uses 'self.heads[task_name]'
- **Fix:** Store the originating task per embedding (RUN-030) and route each replayed embedding through its own head; forward_from_embeddings must accept a per-group task identity. Cloned in the provenance sibling.
- **Fix at:** `code/m7_trimodal_run.py:344-349`, `code/m7_trimodal_run.py:451-454`, `code/m7_trimodal_run_provenance.py:409-414`, `code/m7_trimodal_run_provenance.py:614`
- _related: RUN-002, RUN-003, RUN-030_

#### `RUN-005` [HIGH] 'fwt' is T2 learning gain (after minus zero-shot), mislabeled as forward transfer
`code/m7_trimodal_run.py:616` · research-integrity · status:live · effort:small · confidence:high
- **What:** fwt = acc_t2_after_t2 - acc_t2_zero_shot measures how much SHD improved by training on SHD; it is a learning gain, not forward transfer. The zero-shot baseline (559) evaluates SHD through a randomly initialized SHD backbone and head, so it is near chance and 'fwt' is essentially SHD test accuracy minus chance. Reporting this as 'forward transfer ~75 pp' overstates a transfer effect the quantity does not measure. The provenance sibling keeps 'learning_gain' and 'fwt' as separate quantities and defines fwt against a random-init baseline.
- **Evidence:** line 616 'fwt = acc_t2_after_t2 - acc_t2_zero_shot'; line 559 zero-shot eval; contrast provenance 1065-1066 which separates learning_gain_t2 from fwt_t2
- **Fix:** Rename this field to learning_gain (or gain_T2); if a forward-transfer number is needed, define it against a from-scratch/random-init baseline as the provenance sibling does. Do not let downstream docs call this 'forward transfer'.
- **Fix at:** `code/m7_trimodal_run.py:616`, `code/m7_trimodal_run.py:630`
- _confirmed by SNN-002; related: RUN-006, RUN-013_

#### `RUN-006` [HIGH] 'forgetting_T1_T3' is the incremental T2->T3 delta, not total T1 forgetting after T3
`code/m7_trimodal_run.py:614` · research-integrity · status:live · effort:small · confidence:high
- **What:** forgetting_t1_t3 = acc_t1_after_t2 - acc_t1_after_t3 measures only the additional T1 drop during T3 training. Its sibling metrics use start-to-end semantics: forgetting_T1_T2 = acc_t1 - acc_t1_after_t2 and forgetting_T2_T3 = acc_t2_after_t2 - acc_t2_after_t3. Under a name that reads as 'T1 forgetting measured after T3', a reader expects acc_t1 - acc_t1_after_t3. The inconsistent definition will be misread when aggregated.
- **Evidence:** line 614 'forgetting_t1_t3 = acc_t1_after_t2 - acc_t1_after_t3' stored as 'forgetting_T1_T3' at 628; compare 613 (T1_T2 total) and 615 (T2_T3 total)
- **Fix:** Rename to forgetting_T1_incremental_T3, or redefine as acc_t1 - acc_t1_after_t3 to match the total-forgetting semantics of the other two fields; document which is intended.
- **Fix at:** `code/m7_trimodal_run.py:613-615`, `code/m7_trimodal_run.py:627-629`
- _confirmed by SNN-010; related: RUN-005_

#### `RUN-007` [HIGH] Reported test accuracy uses last-epoch weights; the best-val checkpoint is saved but never reloaded
`code/m7_trimodal_run.py:485-504` · research-integrity · status:live · effort:small · confidence:high
- **What:** train_task saves best_*.pt whenever val improves (485-495) but never reloads it. test_acc is computed on whatever weights exist after the loop (504, last or early-stopped epoch), and the caller saves final_*.pt from that same last-epoch state (543/571/598). Early stopping thus governs when training halts, but the model carried forward and reported is the last-epoch model, not the best-val model. best_*.pt is dead output.
- **Evidence:** lines 485-495 save 'best_{task_name}_{modality}.pt'; line 504 'test_acc = self.evaluate(test_loader, ...)' after the loop; caller torch.save of final_*.pt at 543/571/598; no torch.load of best_*.pt anywhere
- **Fix:** Either load best_*.pt before the final test eval and before saving final_*.pt, or drop the best-checkpoint machinery and state that reporting is last-epoch. Decide deliberately and document it.
- **Fix at:** `code/m7_trimodal_run.py:485-505`, `code/m7_trimodal_run.py:543`, `code/m7_trimodal_run.py:571`, `code/m7_trimodal_run.py:598`
- _confirmed by SNN-009; related: RUN-008_

#### `RUN-008` [HIGH] best_*.pt filename omits seed and condition, so all 6 runs overwrite the same 3 files
`code/m7_trimodal_run.py:488` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** The best-checkpoint path is 'best_{task_name}_{modality}.pt' with no seed or condition. Across 3 seeds x 2 conditions the loop writes only best_nmnist_nmnist.pt, best_shd_shd.pt, best_dvs_dvs.pt, so every run clobbers the previous run's best checkpoint. (final_*.pt is correctly namespaced by seed and condition at 536/562/589.) Even ignoring RUN-007, no per-run best weights survive, which also violates the repo's back-up-every-seed rule.
- **Evidence:** line 488 'ckpt_path = CKPT_DIR / f"best_{task_name}_{modality}.pt"' vs final path line 536 'final_nmnist_nmnist_seed{seed}_{condition}.pt'
- **Fix:** Include seed and condition (and ideally step) in the best_*.pt filename, matching the final_*.pt convention.
- **Fix at:** `code/m7_trimodal_run.py:488`
- _related: RUN-007_

#### `RUN-009` [HIGH] Silent checkpoint-skip loads stale final_*.pt with no provenance check and skips gate-trajectory logging
`code/m7_trimodal_run.py:537-544` · research-integrity · status:live · effort:medium · confidence:high
- **What:** For each task, if final_{task}_{task}_seed{seed}_{condition}.pt exists the code loads it and skips training (prints only a skip glyph). There is no check that the checkpoint matches current code/config, so a stale checkpoint yields stale results presented as fresh. Because _save_gate_history is only called in the training branch (544/572/599), a skipped task writes no gate trajectory, so gate_trajectory.csv under-populates or is empty on any reuse run - and that CSV is the sole source for the paper's core gate-specialization figure.
- **Evidence:** lines 537-539 'if ckpt_t1.exists(): ... model.load_state_dict(torch.load(...))'; repeated 563-565 and 590-592; _save_gate_history only inside the else branch at 544/572/599
- **Fix:** Record code/config version with each checkpoint and refuse or warn on mismatch; regenerate/reload the gate trajectory on skip so gate_trajectory.csv is never silently partial. At minimum fail loudly instead of skipping silently.
- **Fix at:** `code/m7_trimodal_run.py:537-544`, `code/m7_trimodal_run.py:563-572`, `code/m7_trimodal_run.py:590-599`
- _related: RUN-010_

#### `RUN-010` [HIGH] Results CSV mixes two schemas and appends via read-concat-overwrite with no dedup
`code/m7_trimodal_run.py:633-697` · research-integrity · status:live · effort:medium · confidence:high
- **What:** cl_trimodal_results.csv receives end-of-seed summary rows (no 'step' key, 618-640) and per-task incremental rows (with a 'step' key, 687-697) in one file. Both writers do read-concat-write with no de-duplication, so re-running any seed appends duplicate rows that silently inflate n and corrupt mean/std. The only discriminator between the schemas is presence of 'step', so the mandatory 'df[df.step.isna()].drop_duplicates()' ritual is load-bearing and fragile. gate_trajectory.csv (650-684) has the same append-without-dedup pattern.
- **Evidence:** summary write 635-640 'df = pd.concat([df, pd.DataFrame([result])])' (result has no 'step'); incremental write 690-696 with a row containing 'step'; gate write 681-683 same concat pattern
- **Fix:** Write a single normalized schema (or one file per schema), key rows by (seed, condition, step), and upsert on that key instead of blind append; or centralize on the canonical loader and never append raw. Document the schema.
- **Fix at:** `code/m7_trimodal_run.py:633-640`, `code/m7_trimodal_run.py:650-684`, `code/m7_trimodal_run.py:687-697`
- _related: RUN-009, RUN-035_

#### `RUN-011` [HIGH] Accuracy is recorded and reported with no spike-rate/sparsity or timestep cost
`code/m7_trimodal_run.py:411-419` · research-integrity · status:live · effort:medium · confidence:high
- **What:** This is an SNN paper, yet evaluate() returns only accuracy and the saved result dict contains only accuracies, forgetting and fwt (618-631). The forward pass produces spike counts (backbone sums over T) but spike rate/sparsity is never computed, logged, or written to any CSV, and the timestep counts (25 visual, 100 SHD) are not recorded either. The repo rule requires accuracy to travel with spike rate/sparsity and timestep count, so every accuracy this script emits is incomplete as a reported result.
- **Evidence:** lines 417-419 evaluate returns '100.0 * correct / total'; result dict 618-631 has no spike-rate/sparsity/timestep fields; no spike-rate computation anywhere in the file
- **Fix:** Instrument the forward pass to accumulate mean spike rate/sparsity per layer and add spike-rate/sparsity and per-modality timestep columns to the results CSV so every accuracy is reported with its cost.
- **Fix at:** `code/m7_trimodal_run.py:248-271`, `code/m7_trimodal_run.py:411-419`, `code/m7_trimodal_run.py:618-631`, `code/m7_trimodal_run_provenance.py:539-548`, `code/m7_trimodal_run_provenance.py:1137-1159`
- _confirmed by SNN-005; related: RUN-014, SNN-005_

#### `RUN-012` [HIGH] HGRN reset gate is inert: h_prev is always zero, so r*h_prev is 0 and W_r/norm_r get no gradient
`code/m7_trimodal_run.py:223-231` · correctness · status:live · effort:medium · confidence:high
- **What:** ImprovedHGRNGate is only ever called with a single argument (self.hgrn(features) at 340, self.hgrn(embs) at 347), so h_prev defaults to zeros on every call and the gate is never used recurrently. With h_prev=0, 'r * h_prev' is identically 0, h_tilde reduces to tanh(norm_h(W_h([x,0]))) and the output is 'z * h_tilde'. The reset gate r (and W_r, norm_r) has zero effect on the output and receives zero gradient, so those parameters stay at init. The multiplicative reset r*h_prev is exactly the operation flagged as the central SpiNNaker risk (R3) and a novelty of the paper - and in the model as run it does nothing.
- **Evidence:** line 226 'h_prev = torch.zeros(batch_size, ...)'; line 230 'h_tilde = torch.tanh(self.norm_h(self.W_h(torch.cat([x, r * h_prev], dim=-1))))'; line 231 'return (1 - z) * h_prev + z * h_tilde'; call sites 340 and 347 pass no h_prev
- **Fix:** Decide what the HGRN should do: if recurrence is intended, thread a real hidden state; if not, remove the reset gate and its weights and revise the novelty/SpiNNaker-R3 claims. Either way it must not be described as a working multiplicative-reset recurrent gate.
- **Fix at:** `code/m7_trimodal_run.py:214-231`, `code/m7_trimodal_run.py:340`, `code/m7_trimodal_run.py:347`
- _related: RUN-013_

#### `RUN-013` [HIGH] Near-zero forgetting is structural: only hopfield/hgrn/gate are shared across tasks; backbones and heads are grad-isolated
`code/m7_trimodal_run.py:273-300` · research-integrity · status:live · effort:small · confidence:high
- **What:** Each task has its own backbone (keyed by task name) and its own head, and the optimizer only updates parameters in the current task's forward graph, so an inactive task's backbone/head receive no gradient and are effectively frozen. The only parameters shared and trainable across all three tasks are the Hopfield layer, the HGRN gate, and the adaptive gate. Any measured forgetting can only come from drift in those three modules, so near-zero forgetting is largely a consequence of architectural isolation, not of replay. Summaries must not attribute the ~0 pp forgetting to replay preventing catastrophic forgetting.
- **Evidence:** per-task backbones 277-292 and per-task heads 298-300; optimizer over all params 424 but forward/replay only touch current-task modules; T1 re-eval after T2/T3 (580, 607) uses the unchanged nmnist backbone/head plus drifted shared modules
- **Fix:** No code change required for correctness, but reporting must state that forgetting is measured only through the shared hopfield/hgrn/gate modules and is bounded by architectural isolation; keep the claim to 'replay enables gate specialization' per repo RULE 5.
- **Fix at:** `code/m7_trimodal_run.py:273-300`, `code/m7_trimodal_run.py:421-458`
- _related: RUN-005, RUN-012_

#### `SNN-006` [HIGH] SHD runs 100 timesteps while N-MNIST/DVS run 25; time_steps=25 misrepresents SHD; ~4x feature-magnitude asymmetry into the shared gate
`code/m7_trimodal_run.py:139` · numerics · status:live · effort:medium · confidence:high
- **What:** SHD events_to_dense uses time_bins=100 (m7 L139; provenance L1201; and confirmation/rehearsal/joint files) while visual/DVS collates clamp to 25 (m7 L77-80/94-97) and CONFIG['time_steps']=25 (m7 L58). Features are a SUM over T (m7 L271/318). So the audio backbone integrates over 100 steps and visual over 25: time_steps=25 is false for SHD (any reported timestep count would be wrong, and the SpiNNaker beta=exp(-dt/tau) mapping is off for audio), and SHD embeddings feed the shared, non-normalized adaptive_gate (m7 L295-296) at ~4x the magnitude of visual embeddings, coupling gate/Hopfield behaviour to the step-count asymmetry.
- **Evidence:** events_to_dense time_bins=100 vs collate clamp 25 vs CONFIG time_steps=25
- **Fix:** Either bin SHD to 25 (respect time_steps) or record the true per-modality T and report it; if the asymmetry is intentional, state it and normalize features before the gate.
- **Fix at:** `code/m7_trimodal_run.py:139`, `code/m7_trimodal_run.py:338`, `code/m7_trimodal_run_provenance.py:1201`
- _confirmed by RUN-014; related: SNN-005_

#### `SNN-007` [HIGH] Training DataLoaders shuffle from the global RNG (no per-loader generator=)
`code/m7_trimodal_run.py:127` · numerics · status:live · effort:small · confidence:high
- **What:** Every shuffle=True training loader omits generator= (m7 L127; provenance L1187/1191; confirmation L88; rehearsal L66; run_joint_shared_encoder L438). With no dedicated generator, RandomSampler seeds each epoch's shuffle from the global torch generator, whose state is whatever model construction (and SNN-001) left it. Any change to init order or parameter count reshuffles the training stream and moves trained numbers even with the same seed. This is the general mechanism SNN-001 exploits.
- **Evidence:** m7 L127 DataLoader(...,shuffle=True) with no generator=
- **Fix:** Pass generator=torch.Generator().manual_seed(seed) to each training DataLoader so shuffling is decoupled from init RNG (mirrors how the splits are already pinned).
- **Fix at:** `code/m7_trimodal_run.py:127`, `code/m7_trimodal_run_provenance.py:1187`, `code/run_cl_confirmation_fixed.py:88`, `code/run_cl_rehearsal.py:66`, `code/run_joint_shared_encoder.py:438`
- _related: SNN-001_

#### `PROV-002` [HIGH] Default forward path silently routes features through proj_shared/proj_private (512->128+384->512), diverging from the documented architecture and from the orphan copy
`code/m7_trimodal_run_provenance.py:362-399` · architecture · status:live · effort:medium · confidence:high
- **What:** forward() (401-407) always calls forward_with_shared() (372-399), which always calls _split_shared_private() (394): shared=proj_shared(features) (364), private=proj_private(features) (368), full=cat([shared,private]) (369). The memory stack then consumes full, not the raw backbone features (gate=adaptive_gate(full) 395, hop=hopfield(full) 396, hgrn=hgrn(full) 397). proj_shared and proj_private are independent trainable nn.Linear layers, so full = W*features + b is an EXTRA learned 512->512 reparametrisation inserted before Hopfield/HGRN/gate. This projection is documented (forward_with_shared docstring, 372-376) as used only 'during cross-modal alignment training', yet it is active on every forward pass even with align_cross_modal=False (the default, 1321). The orphan copy code/m7_trimodal_run.py feeds RAW features to the same modules (m7_trimodal_run.py:338-340), and CLAUDE.md documents the pipeline as backbone->Hopfield||HGRN->gate with no shared/private split. So the shared hub runs a different, higher-capacity model than both the docs and the nominal reproducibility entry point, changing numerics.
- **Evidence:** self.proj_shared=nn.Linear(cfg.shared_dim, cfg.align_shared_dim) built unconditionally (337); _split_shared_private returns full=cat([shared,private]) (369); forward_with_shared uses gate/hop/hgrn on full (395-398); forward delegates to forward_with_shared (406). Contrast orphan m7_trimodal_run.py:338-340 which uses raw features.
- **Fix:** Only apply the shared/private projection when cfg.align_cross_modal (or an explicit flag) is set; otherwise feed raw backbone features to gate/hopfield/hgrn so the default model matches the documented architecture and the orphan. If the projection is intended to be permanent, document it in CLAUDE.md/EXPERIMENTAL_MANIFEST and reconcile the orphan copy. Any change here moves numbers: call it out with before/after.
- **Fix at:** `code/m7_trimodal_run_provenance.py:362-399`, `code/m7_trimodal_run_provenance.py:336-341`
- _related: PROV-003, PROV-007_

#### `PROV-003` [HIGH] Replay path (forward_from_embeddings) bypasses _split_shared_private, so replayed embeddings hit a different subgraph than live inputs
`code/m7_trimodal_run_provenance.py:409-414` · correctness · status:live · effort:small · confidence:high
- **What:** Live inputs are transformed by _split_shared_private into full before gate/hopfield/hgrn (394-398). But forward_from_embeddings(embs, task_name) (409-414) applies adaptive_gate/hopfield/hgrn DIRECTLY to the stored embeddings, skipping proj_shared/proj_private entirely. The stored embeddings are raw backbone features (populate calls model._get_features, 433, which returns backbone output). So replay trains gate/Hopfield/HGRN/heads on raw-feature distribution while inference uses the projected full distribution — a train/inference mismatch that corrupts the replay gradient even if PROV-001 were fixed. This inconsistency is PROVENANCE-SPECIFIC: in the orphan both live and replay paths use raw features (m7_trimodal_run.py:338-340 vs 344-349), so they are consistent there. The bug was introduced by the proj insertion in PROV-002.
- **Evidence:** forward_from_embeddings: gate=adaptive_gate(embs), hop=hopfield(embs) on raw embs (410-411); live path uses full via _split_shared_private (394-398); populate stores model._get_features(...) raw features (433, 359-360).
- **Fix:** Route replayed embeddings through the same _split_shared_private transform used for live inputs before gate/hopfield/hgrn, OR store the post-projection full representation in the buffer. Keep replay and inference on the identical subgraph.
- **Fix at:** `code/m7_trimodal_run_provenance.py:409-414`, `code/m7_trimodal_run_provenance.py:427-444`
- _related: PROV-002, PROV-001_

#### `PROV-004` [HIGH] Best-validation weights are never restored: reported best_val, saved checkpoint, and test_acc all use the final (post-early-stopping) epoch
`code/m7_trimodal_run_provenance.py:662-673` · research-integrity · status:live · effort:small · confidence:high
- **What:** train_task tracks best_val as a scalar (662-663) but never snapshots the weights at that epoch. Early stopping breaks after patience epochs of no improvement (665-669), leaving the model in a degraded post-peak state. test_acc is then computed on that final state (671) and the checkpoint saved by run_one_seed is model.state_dict() at end of training (1003, 1054, 1106) — i.e. final-epoch, not best. So the returned best_val does not correspond to the weights that produce test_acc or to the weights saved and reloaded on the next task. Reporting best_val alongside a test number from different weights is misleading. Note the orphan clone at least saves a best_*.pt (m7_trimodal_run.py:488-495) but also never reloads it before test — provenance dropped even that save.
- **Evidence:** best_val updated but no best-state kept (662-664); early stop break (667-669); test_acc=self.evaluate(test_loader,...) after loop on final weights (671); torch.save(model.state_dict()) final-epoch (1003,1054,1106).
- **Fix:** Snapshot state_dict when val improves (copy.deepcopy) and reload it before computing test_acc and before saving the checkpoint, so best_val, the test metric, and the persisted weights are mutually consistent. Do the same in the orphan.
- **Fix at:** `code/m7_trimodal_run_provenance.py:662-673`, `code/m7_trimodal_run_provenance.py:1003`, `code/m7_trimodal_run_provenance.py:1054`, `code/m7_trimodal_run_provenance.py:1106`, `code/m7_trimodal_run.py:484-505`
- _related: PROV-007_

#### `PROV-005` [HIGH] forgetting_T1_T3 uses the after-T2 accuracy as its baseline, inconsistent with the other two forgetting metrics (mislabeled)
`code/m7_trimodal_run_provenance.py:1132-1135` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** forgetting_T1_T2 = acc_t1_after - acc_t1_after_t2 (1133) and forgetting_T2_T3 = acc_t2_after - acc_t2_after_t3 (1135) both measure drop from a task's OWN post-training accuracy (cumulative forgetting). But forgetting_T1_T3 = acc_t1_after_t2 - acc_t1_after_t3 (1134) measures the INCREMENTAL drop between the after-T2 and after-T3 stages, not cumulative T1 forgetting after T3. A reader interpreting 'forgetting_T1_T3' as T1 forgetting measured after T3 would expect acc_t1_after - acc_t1_after_t3. The three metrics therefore do not share one definition, and any total-forgetting summary built from them is wrong. Identical bug in the orphan (m7_trimodal_run.py:613-615). Forgetting is a RULE-5-sensitive claim.
- **Evidence:** forgetting_t1_t2 = acc_t1_after - acc_t1_after_t2 (1133); forgetting_t1_t3 = acc_t1_after_t2 - acc_t1_after_t3 (1134); forgetting_t2_t3 = acc_t2_after - acc_t2_after_t3 (1135). acc_t1_after set at 1007, acc_t1_after_t2 at 1061, acc_t1_after_t3 at 1113.
- **Fix:** Decide one definition. For cumulative forgetting after T3 use acc_t1_after - acc_t1_after_t3; if the incremental T2->T3 drop is genuinely wanted, rename it (e.g. forgetting_T1_incremental_T3). Apply the same rename/fix to the orphan.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1132-1135`, `code/m7_trimodal_run.py:613-615`
- _related: PROV-007_

#### `PROV-006` [HIGH] Results CSV is appended without dedup and aggregated without dedup; schema also diverges from the orphan and from the canonical loader
`code/m7_trimodal_run_provenance.py:849-949` · research-integrity · status:live · effort:medium · confidence:high
- **What:** save_seed_result appends each seed's row to cl_trimodal_results.csv via read+concat+write (853-856) with NO dedup, so reruns accumulate duplicate rows. aggregate_results reads that master (883), filters only df['T1_after_T1'].notna() (885), and computes mean/std/CI/Wilcoxon over the raw rows (888-936) — duplicates inflate n and corrupt std/CI exactly as warned in the repo's own load_results.py. Separately, the schema DIVERGES: this file writes summary rows with NO 'step' column and columns T2_fwt/T3_fwt (1154-1155, plus T2_zero_shot/T3_zero_shot/T2_learning_gain/etc.), whereas the orphan writes incremental rows WITH a 'step' column plus a single 'fwt' column (m7_trimodal_run.py:630, 687-697). The documented pandas ritual df[df['step'].isna()] (python-numerics rule 3) KeyErrors on this file's CSV because there is no 'step' column, and the canonical loader looks for a 'fwt' column (load_results.py:206) this file never writes. Because both hubs write the SAME filename ./ICONS_M7/results/cl_trimodal_results.csv, mixing runs yields inconsistent columns and the notna filter would even keep the orphan's step-rows (their T1_after_T1 is populated).
- **Evidence:** append without dedup (853-856); aggregate filters notna only, no drop_duplicates (885-905); ci/std over raw rows (894-896); T2_fwt/T3_fwt (1154-1155); orphan writes 'step' rows and 'fwt' (m7_trimodal_run.py:687-697, 630); canonical loader expects step + fwt (load_results.py:82,206).
- **Fix:** Deduplicate to one row per (seed, condition) before aggregating (reuse .claude/skills/analyze-cl-results/scripts/load_results.py). Standardise a single CSV schema across both hubs (or write distinct filenames) and align column names (fwt vs T2_fwt/T3_fwt, step column) with the canonical loader.
- **Fix at:** `code/m7_trimodal_run_provenance.py:849-867`, `code/m7_trimodal_run_provenance.py:873-905`, `code/m7_trimodal_run_provenance.py:1137-1159`, `code/m7_trimodal_run.py:618-697`
- _related: PROV-007, PROV-009_

#### `PROV-007` [HIGH] Two divergent copies of the model+trainer share class/function names and output filenames but differ in model math, metrics, CSV schema, freezing, and import-safety
`code/m7_trimodal_run_provenance.py:1-1379` · duplication · status:live · effort:large · confidence:high
- **What:** m7_trimodal_run_provenance.py (the shared hub, referenced by 19 files) and code/m7_trimodal_run.py (the orphan, imported by nobody) define the SAME public names — M7_ContinualAdaptiveModel, ContinualLearner, EmbeddingReplayBuffer, SNN_Backbone, run_one_seed, TASKS, collate fns — yet have DIVERGED substantially: (1) provenance inserts a proj_shared/proj_private reparametrisation into the forward path, the orphan does not (PROV-002); (2) provenance freezes completed backbones/heads via _freeze_completed_tasks (550-568), the orphan freezes nothing so it trains all params every task; (3) provenance defines 'fwt' as zero_shot - random_init (1066) while the orphan defines 'fwt' as after - zero_shot (m7_trimodal_run.py:616) — a different, mislabeled quantity (PROV-008); (4) different CSV schema and filename collision (PROV-006); (5) provenance is import-safe with a __main__ guard, the orphan runs a 4-5h job at import. Both write ./ICONS_M7/results/cl_trimodal_results.csv and identical checkpoint names (final_{task}_{task}_seed{seed}_{condition}.pt), so their artifacts are mutually confusable while their numerics differ. Edits to one will not propagate to the other.
- **Evidence:** Shared names across both files; _freeze_completed_tasks present only in provenance (550-568); fwt formulas differ (1066 vs m7_trimodal_run.py:616); identical checkpoint naming (1) prov 992/1041/1093 (2) orphan 536/562/589; __main__ guard prov 1378 vs orphan top-level run 707-711.
- **Fix:** Collapse to a single source of truth: keep the provenance hub, delete or clearly quarantine the orphan (it is imported by nobody), or extract the shared model/trainer into one module both import. At minimum stop them writing identical filenames.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1-1379`, `code/m7_trimodal_run.py:1-720`
- _related: PROV-002, PROV-005, PROV-006, PROV-008_

#### `PROV-008` [HIGH] 'fwt' means different things in the two hubs; the orphan's formula is learning-gain mislabeled as forward transfer, and only that mislabel can produce a large positive FWT
`code/m7_trimodal_run_provenance.py:1064-1066` · research-integrity · status:live · effort:small · confidence:high
- **What:** This file computes fwt_t2 = acc_t2_zero_shot - random_init_baselines[task2] (1066): zero-shot accuracy on T2 (with an untrained T2 head, measured after T1) minus a fresh-random-init baseline. Because the T2 head is randomly initialised in BOTH terms, both are near chance and this FWT is ~0 by construction (print_task_sanity even warns if zero_shot deviates >5pp from chance, 840-841). That is the honest forward-transfer measure. The orphan clone instead computes fwt = acc_t2_after_t2 - acc_t2_zero_shot (m7_trimodal_run.py:616), which is the TRAINING gain on T2, not transfer, yet is stored under the name 'fwt'. Any large positive 'FWT' figure (the docs quote ~75 pp, unsourced per RULE 1) can only arise from the orphan's mislabeled learning-gain formula, not from a real forward-transfer computation. This directly bears on contested-fact #4 (the two cross-modal-transfer narratives).
- **Evidence:** prov: fwt_t2 = acc_t2_zero_shot - random_init_baselines[task2] (1066); learning_gain_t2 = acc_t2_after - acc_t2_zero_shot computed separately (1065); orphan: fwt = acc_t2_after_t2 - acc_t2_zero_shot (m7_trimodal_run.py:616). Sanity warns if zero_shot far from chance (840).
- **Fix:** Rename the orphan's 'fwt' to 'learning_gain' (it is not forward transfer). Keep this file's FWT definition, and when any FWT number is reported, state which formula produced it and report it with its near-chance baseline. Do not repeat the ~75 pp FWT claim without a CSV.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1064-1066`, `code/m7_trimodal_run_provenance.py:1120`, `code/m7_trimodal_run.py:616`, `code/m7_trimodal_run.py:630`
- _related: PROV-007_

#### `SNN-003` [HIGH] Results CSV append-without-dedup; provenance aggregator never dedups
`code/m7_trimodal_run_provenance.py:883-936` · numerics · status:live · effort:medium · confidence:high
- **What:** m7_trimodal_run.py _save_incremental (L687-697) reuses metric_name as BOTH the step value AND a populated column (L690) so T1_after_T1/T2_after_T2/T3_after_T3 appear in both incremental and summary rows; summary write L634-640 appends with no dedup. provenance save_seed_result L849-860 appends with no dedup; aggregate_results filters notna (L885, good) but never drop_duplicates, then computes mean/std(ddof=1)/n=len(sub)/ci95 (L894-905) and a Wilcoxon set_index('seed') (L913-915). A rerun (routine, because checkpoint-skip re-evaluates) appends a second summary row per seed: n inflates, std/ci95 collapse, the Wilcoxon join explodes on duplicate seed indices, and m7's naive reader gets different n per metric.
- **Evidence:** provenance L883-936; m7 L687-697, L634-640
- **Fix:** Route all analysis through load_results.py (aggregates over unique seeds). In aggregate_results add df=df[df.get('step').isna()].drop_duplicates(['seed','condition'],keep='last') before aggregation. In _save_incremental stop reusing metric_name as the column name.
- **Fix at:** `code/m7_trimodal_run_provenance.py:883-936`, `code/m7_trimodal_run.py:687-697`, `code/m7_trimodal_run.py:634-640`

#### `SNN-004` [HIGH] Checkpoint-skip has no staleness/config guard; provenance filename is config-blind
`code/m7_trimodal_run_provenance.py:992-996` · numerics · status:live · effort:medium · confidence:high
- **What:** m7 L537-539/563-565/590-592 and provenance L993-995/1042-1044/1094-1096 skip training when a matching-named checkpoint exists, with no warning. Filename final_{task}_{task}_seed{seed}_{condition}.pt (provenance L992) is keyed only by task/seed/condition, NOT by shared_dim, num_patterns, or align_*. If architecture/loss changes, a stale old checkpoint is loaded and evaluated as fresh; an ablation that flips align_cross_modal or changes shared_dim reuses a different-config checkpoint. run_cl_confirmation_fixed._is_complete_result (L53-60) skips a whole recompute purely because the CSV has expected columns. On skip, _save_gate_history is not written, so Figure 3 silently loses those seeds.
- **Evidence:** save bare {'model_state_dict'} m7 L543/571/598; provenance filename L992
- **Fix:** Embed {config_hash, git_sha, code_version} in the checkpoint; on load verify it matches and retrain on mismatch; put a config hash in the filename; gate _is_complete_result on the same hash.
- **Fix at:** `code/m7_trimodal_run.py:543`, `code/m7_trimodal_run_provenance.py:1003`, `code/run_cl_confirmation_fixed.py:53-60`
- _related: SNN-003_

#### `SNN-008` [HIGH] Provenance config JSON records max_epochs=50/patience=12 but the run hardcodes 15/5
`code/m7_trimodal_run_provenance.py:41-42` · research-integrity · status:live · effort:small · confidence:high
- **What:** ExperimentConfig.max_epochs=50, patience=12 (L41-42) are saved next to the checkpoint (cfg.to_json L1005), but train_task is called with num_epochs=15, patience=5 (L999/1047/1099). The provenance edition's whole purpose is faithful config capture, yet the persisted JSON misstates the training length actually used. Separately m7 hardcodes weight_decay=1e-4 (L424) ignoring CONFIG['weight_decay'], and dropout=0.2 (L53) and pin_memory (L59) are never used.
- **Evidence:** provenance L41-42 vs call L999; m7 L424 vs CONFIG
- **Fix:** Pass cfg.max_epochs/cfg.patience into train_task (or set them to 15/5 so the JSON is truthful); replace the hardcoded 1e-4 in m7 with CONFIG['weight_decay']; drop or wire the unused keys.
- **Fix at:** `code/m7_trimodal_run_provenance.py:999`, `code/m7_trimodal_run_provenance.py:41-42`, `code/m7_trimodal_run.py:424`

#### `DEAD-012` [HIGH] modify_notebook.py crashes immediately - its target notebook no longer exists
`code/modify_notebook.py:8-9` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** The script opens 'NeurIPS2026_M7_clean (3) (1).ipynb' (L8-9) to patch it, but that notebook was deleted from the repo (git commit 837d7f9 'Remove original-M7 experiment artifacts ... notebooks'; no .ipynb exists anywhere in the tree). Any run raises FileNotFoundError at L9 before doing anything. It is both dead (the notebook workflow is retired) and non-functional. Per CLAUDE.md it is the ONLY surviving record of the old notebook's cell structure, so archiving is preferable to hard deletion if that provenance matters.
- **Evidence:** open(SCRIPT_DIR/'NeurIPS2026_M7_clean (3) (1).ipynb') at L8; 'find . -name *.ipynb' returns nothing; git log shows the notebook removed (837d7f9).
- **Fix:** Move to code/archive/ (retains cell-structure provenance) or delete. Never import (see DEAD-013).
- **Fix at:** `code/modify_notebook.py:1-569`
- _related: DEAD-013, DEAD-014_

#### `DEAD-013` [HIGH] modify_notebook.py reads and overwrites a notebook in place on import, with no backup
`code/modify_notebook.py:8-564` · safety · status:dead · effort:small · confidence:high
- **What:** No __main__ guard: at module scope it reads the notebook (L8-9) and at the end writes json.dump back to the SAME path (L563-564) with no backup copy. Were the target present, 'import modify_notebook' would destructively rewrite it. Today it crashes first (DEAD-012), but the destructive read-then-overwrite-in-place pattern is the hazard to retire. Violates the repo rule against overwriting a file without showing a diff.
- **Evidence:** read open('r') L8; write open('w') + json.dump(nb, f) L563-564, same filename; grep '__main__' returns 0.
- **Fix:** Delete/archive. If ever adapted, write to a NEW path and never overwrite the source in place.
- **Fix at:** `code/modify_notebook.py:8-9`, `code/modify_notebook.py:563-564`
- _related: DEAD-012_

#### `DEAD-016` [HIGH] monitor_and_stop.py can send kill -15/-9 to an unrelated process (hardcoded PID 16672)
`code/monitor_and_stop.py:12-48` · safety · status:dead · effort:trivial · confidence:high
- **What:** PID is hardcoded to 16672 (L12). Inside an infinite 'while True' loop (L33) it watches a foreign log (L9, which never exists here, so get_latest_val_accuracies returns [] and the process loops forever). If the plateau condition were ever met it runs 'kill -15 16672' (L44) then 'kill -9 16672' (L48). On any machine PID 16672 is whatever process currently holds that id - potentially an unrelated user process - so running this can terminate the wrong process. PIDs are not stable identifiers.
- **Evidence:** PID=16672 L12; subprocess.run(['kill','-15',str(PID)]) L44; subprocess.run(['kill','-9',str(PID)]) L48; while True L33; foreign LOG_FILE L9.
- **Fix:** Delete. Never signal a process by a hardcoded literal PID.
- **Fix at:** `code/monitor_and_stop.py:12`, `code/monitor_and_stop.py:44`, `code/monitor_and_stop.py:48`
- _related: DEAD-015_

#### `CLE-013` [HIGH] Missing config_nmnist.json crashes main() (points at results/cl_confirmation)
`code/phase1_sanity_check.py:100-103` · safety · status:live · effort:small · confidence:high
- **What:** main() opens CKPT_DIR/config_nmnist.json unconditionally (100-101). CKPT_DIR = results/cl_confirmation/checkpoints/C_split_128_seed42 (note: cl_confirmation, not cl_confirmation_fixed) does not exist, so the first line of main() raises FileNotFoundError. All three torch.load calls at 128/139-140/161 would also fail, but the config open fails first.
- **Evidence:** Lines 100-103; results/cl_confirmation MISSING on disk. CKPT_DIR defined at line 32.
- **Fix:** Guard the config and each checkpoint path with existence checks and clear errors; note the directory name differs from eval_transfer_fixed.py (cl_confirmation vs cl_confirmation_fixed) -- confirm which run this diagnostic targets.
- **Fix at:** `code/phase1_sanity_check.py:32`, `code/phase1_sanity_check.py:100-103`
- _related: CLE-008_

#### `DEP-001` [HIGH] Future-dated version pins likely unresolvable on PyPI for CPython 3.11
`code/requirements.txt:1-12` · dependencies · status:live · effort:medium · confidence:high
- **What:** Several pins are dated ahead of the assistant knowledge cutoff (Jan 2026) and cannot be confirmed to exist on PyPI for the documented target interpreter python3.11: torch==2.11.0 (L1), torchaudio==2.11.0 (L2), torchvision==0.26.0 (L3), transformers==5.7.0 (L6), pandas==3.0.2 (L8), matplotlib==3.10.9 (L9), scipy==1.17.1 (L12). If any of these versions does not exist for cp311, `pip install -r code/requirements.txt` (the documented bootstrap, since the repo has no venv) fails to resolve and the whole environment cannot be built. These are pinned versions in the file, NOT measured results. numpy is covered separately (DEP-002), pytest separately (DEP-003).
- **Evidence:** code/requirements.txt lines 1-3,6,8,9,12: torch==2.11.0 / torchaudio==2.11.0 / torchvision==0.26.0 / transformers==5.7.0 / pandas==3.0.2 / matplotlib==3.10.9 / scipy==1.17.1
- **Fix:** For each pin, do a resolve dry-run for cp311 (pip index versions <pkg> or pip install --dry-run) and replace any nonexistent version with the newest release that supports Python 3.11; pin the torch/torchaudio/torchvision trio to a mutually compatible published triple. Do not assert a replacement number without confirming it resolves.
- **Fix at:** `code/requirements.txt:1-3`, `code/requirements.txt:6`, `code/requirements.txt:8`, `code/requirements.txt:9`, `code/requirements.txt:12`
- _related: DEP-002, DEP-003_

#### `DEP-002` [HIGH] numpy==1.26.4 is inconsistent with the rest of the stack and likely blocks resolution
`code/requirements.txt:7` · dependencies · status:live · effort:small · confidence:medium
- **What:** numpy==1.26.4 (L7) is a numpy 1.x release that predates the other, future-dated pins. Modern scipy (1.17.1, L12) and pandas (3.0.2, L8) almost certainly declare numpy>=2 as a floor, so a resolver will likely reject numpy 1.26.4 alongside them and the install fails. Even if it resolved, mixing a numpy 1.x ABI with 2026-era torch/scipy binaries risks a binary-incompatibility error at import. 1.26.4 is a value literally present in the file, not a measured result.
- **Evidence:** code/requirements.txt:7 numpy==1.26.4 ; alongside code/requirements.txt:8 pandas==3.0.2 and code/requirements.txt:12 scipy==1.17.1
- **Fix:** Cross-check numpy against the numpy floors declared by the chosen scipy/pandas/torch versions; most likely bump numpy to a 2.x that satisfies all three (verify it resolves for cp311), or downgrade the whole stack consistently. Do not pin numpy independently of scipy/pandas minimums.
- **Fix at:** `code/requirements.txt:7`
- _related: DEP-001_

#### `RA-005` [HIGH] 3 of 4 columns in the printed COMPARISON TABLE are hardcoded literals
`code/run_cl_rehearsal.py:435-444` · research-integrity · status:live · effort:small · confidence:high
- **What:** Only the first column (C_rehearsal) is live; the A_no_align, C_unfrozen and C_frozen columns are ~24 hardcoded literals (accuracies, forgetting, V2A/prototype transfer). Printed under a bare 'COMPARISON TABLE' header with no disclaimer, the table reads as four measured conditions when three are baked in.
- **Evidence:** lines 435-444; verdict UNSOURCED
- **Fix:** Load the comparison conditions from a CSV, or print only the live column and label the rest explicitly as 'prior-run literals, unsourced'.
- **Fix at:** `code/run_cl_rehearsal.py:435-444`
- _confirmed by SNN-012, CLE-025_

#### `CMP-001` [HIGH] Near-miss negatives leak held-out numbers into training
`code/run_compositional_crossmodal.py:538-540` · correctness · status:live · effort:small · confidence:high
- **What:** Phase-(b) near-miss augmentation draws d2p = rng.integers(0,10) and appends a composite with label d1*10+d2p, with NO exclusion of held_numbers. assert_split_coverage guarantees every pos1 digit appears in HELD, so near-miss composites can equal held-out numbers. With shuffle_pairs=False the visual and audio composites at the same batch index share that label, so the held-out number is trained as a labeled cross-modal SupCon POSITIVE pair. Held-30 metrics are therefore contaminated and a positive compositional-generalization result cannot be distinguished from memorization - the exact failure mode the docstring (16-18) claims to exclude.
- **Evidence:** d2p = int(rng.integers(0, 10)); specs.append((d1, d2p)); labels.append(d1 * 10 + d2p)   # shares pos1
- **Fix:** Reject-sample d2p so that d1*10+d2p is not in held_numbers (or give near-miss items a label outside the evaluated number space), and assert generated near-miss labels are disjoint from held before training.
- **Fix at:** `code/run_compositional_crossmodal.py:534-540`

#### `ALGN-002` [HIGH] CKPT_SOURCE points to a checkpoint that does not exist (no results/ tree)
`code/run_correspondence_anchor_v2.py:50` · dependencies · status:live · effort:medium · confidence:high
- **What:** CKPT_SOURCE = results/cl_confirmation/checkpoints/A_no_align_seed42/final_shd_shd.pt. There is no results/ directory in the repo, so run_one_seed's torch.load(CKPT_SOURCE) (line 533) raises FileNotFoundError before any work. That checkpoint is meant to be produced by the cl_confirmation pipeline, which is itself only runnable via an archived, broken import (see ALGN-027). The whole script is un-runnable in the current tree.
- **Evidence:** line 50 CKPT_SOURCE = REPO_ROOT/'results'/'cl_confirmation'/'checkpoints'/'A_no_align_seed42'/'final_shd_shd.pt'; line 533 ckpt = torch.load(CKPT_SOURCE, map_location=str(device)); verified: no results/ dir exists
- **Fix:** Document the required upstream artifact and guard with a clear error, or ship a small fixture; ultimately depends on repairing the cl_confirmation producer (ALGN-027).
- **Fix at:** `code/run_correspondence_anchor_v2.py:50`, `code/run_correspondence_anchor_v2.py:533`
- _related: ALGN-017, ALGN-027, ALGN-029_

#### `ALGN-003` [HIGH] _load_cached_features reads a feature cache that does not exist
`code/run_correspondence_anchor_v2.py:147-157` · dependencies · status:live · effort:medium · confidence:high
- **What:** _load_cached_features requires results/hypergraph_anchor/cache/{modality}_{split}_features.pt for nmnist/shd train+test and raises FileNotFoundError('Run v1 first or build cache separately') otherwise. results/ does not exist, and 'v1' is code/archive/run_correspondence_anchor.py (archived, off sys.path). So the documented 'run v1 first' recovery is itself broken. main() calls this at line 645 before any seed loop.
- **Evidence:** lines 153-154 if not cache_path.exists(): raise FileNotFoundError; line 49 CACHE_DIR = results/hypergraph_anchor/cache; line 645 cached=_load_cached_features(device); verified no results/ dir
- **Fix:** Provide a first-class cache-builder in a live module (the extractor already exists as dead code, see ALGN-006) instead of pointing at archived v1; or bundle the cache.
- **Fix at:** `code/run_correspondence_anchor_v2.py:49`, `code/run_correspondence_anchor_v2.py:147-157`
- _related: ALGN-006_

#### `ALGN-004` [HIGH] Hardcoded baseline numbers printed as a comparison table (RULE 1)
`code/run_correspondence_anchor_v2.py:716-718` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** The BASELINE COMPARISON block prints six literal numbers as if measured: A_no_align Retention ~97.0% V2A ~13.3%; Frozen-C Retention 13.3% V2A 14.8%; Unfrozen+rehearsal Retention 95.7% V2A 14.4%. They sit directly beneath the freshly-computed Correspondence rows (line 715), so a reader treats them as comparable results, but none trace to a CSV in this repo. This violates RULE 1 (never put a number anywhere that is not in a CSV).
- **Evidence:** lines 716-718 print(f'   A_no_align ... Retention ~97.0% | V2A ~13.3% | ...') and the two following lines with 13.3/14.8 and 95.7/14.4
- **Fix:** Remove the hardcoded rows or load them from an actual baseline CSV; do not print unsourced numbers next to measured ones.
- **Fix at:** `code/run_correspondence_anchor_v2.py:716-718`

#### `ALGN-021` [HIGH] Checkpoint-skip staleness trap on T1/T2/T3 (silent stale results)
`code/run_cross_modal_alignment_study.py:211-214` · safety · status:live · effort:medium · confidence:high
- **What:** Each of T1/T2/T3 skips training whenever {condition}_seed{seed}_T{n}.pt exists, loading it with only a '⏩ Loading checkpoint' print (212-214, 276-278, 323-325). This is the documented repo-wide trap: a checkpoint left over from older code silently yields stale results presented as a fresh run, with no code-version or hash guard. Combined with the rerun-append aggregation (ALGN-022), a stale run can quietly enter the summary.
- **Evidence:** lines 212-214 if ckpt_t1.exists(): print('   ⏩ Loading checkpoint'); model.load_state_dict(torch.load(ckpt_t1...)['model_state_dict']); same at 276-278 and 323-325
- **Fix:** Stamp checkpoints with a code/version hash and refuse or warn loudly on mismatch; at minimum log the checkpoint mtime and a hash.
- **Fix at:** `code/run_cross_modal_alignment_study.py:211-214`, `code/run_cross_modal_alignment_study.py:275-278`, `code/run_cross_modal_alignment_study.py:322-325`
- _related: ALGN-022_

#### `ALGN-027` [HIGH] --confirm does 'import run_cl_confirmation' but that module lives only in code/archive/
`code/run_digit_only_alignment.py:615-617` · dependencies · status:live · effort:small · confidence:high
- **What:** Under --confirm the entrypoint runs 'import run_cl_confirmation; run_cl_confirmation.main()' (616-617). run_cl_confirmation.py exists ONLY at code/archive/run_cl_confirmation.py; it was moved out of code/, and archive/ is not on sys.path (only REPO_ROOT/code and REPO_ROOT are inserted at 50-51), so this raises ModuleNotFoundError. A differently-named code/run_cl_confirmation_fixed.py exists but is not what the import targets, and the file's own run_full_cl_confirmation (461) is never wired to the flag. The documented --confirm CLI path is dead.
- **Evidence:** lines 615-617 if args.confirm: import run_cl_confirmation; run_cl_confirmation.main(); verified run_cl_confirmation.py only at code/archive/; sys.path inserts only code/ and repo root (50-51)
- **Fix:** Call the local run_full_cl_confirmation(base_cfg) instead of importing the archived module, or restore/relocate a live run_cl_confirmation module and add it to sys.path.
- **Fix at:** `code/run_digit_only_alignment.py:615-617`, `code/run_digit_only_alignment.py:461-505`
- _confirmed by ARCH-002, DEP-015; related: ALGN-029, ALGN-002_

#### `ALGN-028` [HIGH] SHD accuracy reported and Wilcoxon-tested on VALIDATION, never test (RULE 4)
`code/run_digit_only_alignment.py:318-333` · research-integrity · status:live · effort:small · confidence:high
- **What:** run_alignment_cell records SHD_val_acc = t2_result['best_val'] (318,333) and T1_val_acc = t1_result['best_val'] (277,331). trainer.evaluate on the test split is used ONLY for SHD zero-shot (308); after training, the SHD accuracy is never evaluated on test. SHD_val_acc then flows into summary.csv (aggregate metric) AND is one of the paired-Wilcoxon test metrics (line 416). Per RULE 4 the SHD validation/test gap is 15-19 pp, so the headline retention metric and the significance test are run on an inflated split. Columns are honestly named _val_ (mitigation), but validation is the only post-training accuracy this script produces. Contrast run_cross_modal_alignment_study.py, which evaluates on test (line 290).
- **Evidence:** line 277 t1_val_acc=t1_result['best_val']; line 318 shd_val_acc=t2_result['best_val']; provenance train_task returns {'best_val':...} (m7_trimodal_run_provenance.py:673) = validation; line 308 zero-shot uses ['test']; line 416 test_metrics=[...,'SHD_val_acc']
- **Fix:** Add trainer.evaluate(data_loaders[task]['test'], ...) after training and report/aggregate/test on the test accuracy; keep val only as a secondary column.
- **Fix at:** `code/run_digit_only_alignment.py:271-278`, `code/run_digit_only_alignment.py:312-319`, `code/run_digit_only_alignment.py:328-334`, `code/run_digit_only_alignment.py:416`

#### `JNT-001` [HIGH] Shuffled-label control does not shuffle labels (mislabeled null)
`code/run_joint_magnitude_ablation.py:809-817` · research-integrity · status:live · effort:small · confidence:high
- **What:** The seed-42 'shuffled-label control' is not a control: it trains a second normal model and reports it as the null. shuffle_v2a is therefore a duplicate normal run, not a chance baseline.
- **Evidence:** run_one_arm() 811-817 builds model_s and calls train_model(model_s, ...) with NO shuffling. train_model (457-551) has no shuffled_labels parameter and its loss uses the true labels throughout (494-506). The comment 'Permute audio labels within each batch during training' (815) is never realized. shuffle_ho / result['shuffle_v2a'] (817,842) is a second normal 40-epoch run.
- **Fix:** Add a shuffled_labels flag to train_model that permutes the labels feeding CrossModalSupCon (and per-modality CE), mirroring run_correspondence_anchor_v2.py:305-306 and run_joint_shared_encoder.py:512-514; until then do not emit shuffle_v2a as a control.
- **Fix at:** `code/run_joint_magnitude_ablation.py:809-817`, `code/run_joint_magnitude_ablation.py:457-467`
- _related: JNT-031_

#### `JNT-002` [HIGH] Docstring 'held-out 6-9 never receive a cross-modal signal' is false; held-out set is confounded
`code/run_joint_magnitude_ablation.py:11-12` · research-integrity · status:live · effort:medium · confidence:high
- **What:** The header claims digits 6-9 get no cross-modal signal in either arm, but a SHARED magnitude head is trained on 6-9 for both modalities (ON arm) and the SHARED core receives within-modality CE gradients from 6-9 (both arms). Held-out retrieval on 6-9 is therefore confounded, not evidence of emergent cross-modal generalization.
- **Evidence:** Docstring 11-12. magnitude_head is a single shared Linear (381). The within-modality pass regresses y/9.0 on ALL classes incl 6-9 through that shared head for both modalities (518-526), a shared supervised axis for held-out digits. shared_core (371-376) also gets within-modality CE gradients from 6-9 (518-523), co-shaping the z-space that held-out retrieval measures (794-797). The report itself concedes the ON side-channel (970-971,996-997), contradicting the docstring.
- **Fix:** Rewrite the docstring to state 6-9 DO receive (a) a shared magnitude signal in ON and (b) shared-core co-shaping via within-modality CE in both arms; frame OFF-arm held-out retrieval as potentially confounded. To isolate emergence, add an arm that excludes 6-9 from the within-modality pass entirely.
- **Fix at:** `code/run_joint_magnitude_ablation.py:11-15`, `code/run_joint_magnitude_ablation.py:381`, `code/run_joint_magnitude_ablation.py:518-526`
- _related: JNT-003_

#### `JNT-005` [HIGH] CachedNMNIST/CachedSHD pickled as __main__ couples the cache to run-as-main and forces alias hacks downstream
`code/run_joint_magnitude_ablation.py:85-129` · architecture · status:live · effort:medium · confidence:high
- **What:** Dataset classes defined in this run-as-__main__ script are pickled whole via torch.save; any other process must recreate them under __main__ to torch.load the cache. This coupling has already forced identical alias hacks in two downstream scripts. weights_only=False is also an unpickle code-exec surface.
- **Evidence:** CachedNMNIST/CachedSHD (85,105) and SubsampledByClass (132) are saved with torch.save(cache_obj, path) (184-185,197-198) and reloaded with torch.load(..., weights_only=False) (172-173,190-191). Pickled under 'python run_joint_magnitude_ablation.py' the class path is __main__.CachedNMNIST. Confirmed downstream aliasing: run_compositional_crossmodal.py:92-95 and run_joint_diagnostic.py:71-74 both do 'import run_joint_magnitude_ablation as _mag_module; for _cn in (...): setattr(sys.modules["__main__"], _cn, getattr(_mag_module,_cn))' with comments blaming this exact pickling.
- **Fix:** Move CachedNMNIST/CachedSHD/SubsampledByClass into an importable module (e.g. code/joint_datasets.py) so the pickle path is stable and importers need no __main__ aliasing; cache tensor state rather than whole Dataset objects and prefer weights_only=True on a tensor-only cache.
- **Fix at:** `code/run_joint_magnitude_ablation.py:85-129`, `code/run_joint_magnitude_ablation.py:172-198`, `code/run_compositional_crossmodal.py:92-95`, `code/run_joint_diagnostic.py:71-74`

#### `JNT-016` [HIGH] Hardcoded baseline numbers written into the report as if measured (RULE 1 violation)
`code/run_joint_shared_encoder.py:989-993` · research-integrity · status:live · effort:small · confidence:high
- **What:** The generated report embeds frozen-separate baseline figures as hardcoded literals, dressed with a plus/minus as if they came from a run, with no CSV backing.
- **Evidence:** report string 990-991: 'Frozen-separate held-out V2A:  ~0.22 ± 0.20 (chance = 0.25)' and 'Frozen-separate Procrustes:     0.00' are literals written to report.txt. There is no results CSV in-repo. (Same anti-pattern appears out-of-scope at run_correspondence_anchor_v2.py:716-718.)
- **Fix:** Read baseline numbers from their source CSV at runtime or omit them entirely; never embed plus/minus-formatted result literals in a generated report.
- **Fix at:** `code/run_joint_shared_encoder.py:989-993`

#### `ALGN-016` [HIGH] Invalid Mantel permutation null (permutes vectorized triangle, not objects)
`code/run_ordinal_anchor.py:147-168` · correctness · status:live · effort:small · confidence:medium
- **What:** _mantel_test permutes the vectorized upper triangle yv directly (line 161 perm = rng.permutation(len(yv)); yp = yv[perm]) and correlates. A valid Mantel test permutes the n underlying objects (rows AND columns of one RDM together) and recomputes the upper triangle; permuting the 45 distance entries independently breaks the distance-matrix dependency structure and yields an invalid permutation p-value. The observed r (line 157) is fine, but the reported vision_audio_rdm_p is not a valid Mantel p. This diagnostic underpins the 'RDMs uncorrelated (Mantel r, p)' claim.
- **Evidence:** lines 159-167 rng=np.random.default_rng(seed); for _ in range(permutations): perm=rng.permutation(len(yv)); yp=yv[perm]; rp=corrcoef(xv,yp); p_value=(count+1)/(permutations+1); called at line 197 on the 10x10 RDMs
- **Fix:** Permute the n=10 objects: draw a permutation of indices, apply it symmetrically to one RDM (rdm[np.ix_(perm,perm)]), then re-extract the upper triangle and correlate.
- **Fix at:** `code/run_ordinal_anchor.py:147-168`

#### `ALGN-017` [HIGH] CKPT_SOURCE and feature cache missing (script un-runnable)
`code/run_ordinal_anchor.py:63-70` · dependencies · status:live · effort:medium · confidence:high
- **What:** CKPT_SOURCE (63-70) is the same missing results/cl_confirmation/.../final_shd_shd.pt as ALGN-002; run_one_seed's torch.load(CKPT_SOURCE) at line 410 raises FileNotFoundError. main() also calls _load_cached_features (imported from v2, line 615) which needs the missing results/hypergraph_anchor/cache (ALGN-003); the docstring says 'from v2' but the cache is written by v1. Script cannot start in the current tree.
- **Evidence:** lines 63-70 CKPT_SOURCE path; line 410 ckpt=torch.load(CKPT_SOURCE, map_location=str(device)); line 615 cached=_load_cached_features(device); verified no results/ dir
- **Fix:** Same as ALGN-002/ALGN-003: repair the upstream producer and/or ship fixtures; correct the 'from v2' comment (cache is a v1 artifact).
- **Fix at:** `code/run_ordinal_anchor.py:63-70`, `code/run_ordinal_anchor.py:410`, `code/run_ordinal_anchor.py:614-615`
- _related: ALGN-002, ALGN-003, ALGN-027_

#### `CLE-001` [HIGH] No __main__ guard: entire script runs at import time
`code/run_rehearsal_transfer_eval_only.py:1-127` · safety · status:live · effort:small · confidence:high
- **What:** The file has no `if __name__ == "__main__"` and no main(); all work is module-level. Importing it instantiates the model (38), calls torch.load on a checkpoint (45), builds full N-MNIST + SHD test loaders and densifies the entire SHD test set into a list (54-77), runs the transfer eval (81), and writes a results CSV (120-123) as pure import side effects. Unlike m7_trimodal_run it does not launch training, so no 4-5h run, but it does perform a checkpoint load, dataset build, and a (fabricated) CSV write purely on import. Any static tool or downstream module that imports it triggers all of this.
- **Evidence:** No occurrence of `__main__` or `def main` in the file (grep count 0). Module-level statements at lines 37-127.
- **Fix:** Wrap lines 37-127 in `def main():` and gate with `if __name__ == "__main__": main()`, mirroring eval_transfer_fixed.py.
- **Fix at:** `code/run_rehearsal_transfer_eval_only.py:37-127`
- _related: CLE-002, CLE-004_

#### `CLE-004` [HIGH] torch.load of a non-existent checkpoint crashes the module on import
`code/run_rehearsal_transfer_eval_only.py:43-45` · safety · status:live · effort:small · confidence:high
- **What:** Loads results/cl_rehearsal/checkpoints/C_rehearsal_seed42/final_dvs_dvs.pt (43-45). results/ does not exist in the repo (confirmed: results/cl_rehearsal MISSING), so torch.load raises FileNotFoundError. Combined with the missing __main__ guard (CLE-001), merely importing this module raises today. The map_location=str(device) and the assumption the checkpoint is a dict with key 'model_state_dict' add further fragility.
- **Evidence:** Line 44-45 build ckpt_t3 and call `torch.load(ckpt_t3, ...)["model_state_dict"]`; results/cl_rehearsal does not exist on disk.
- **Fix:** After adding the main guard (CLE-001), check `ckpt_t3.exists()` and exit with a clear message if absent; validate the loaded object has 'model_state_dict'.
- **Fix at:** `code/run_rehearsal_transfer_eval_only.py:43-46`
- _related: CLE-001_

#### `TST-016` [HIGH] ImportError on import: _build_digit_loaders is not exported by run_ordinal_anchor
`code/test_ordinal_anchor_smoke.py:14` · correctness · status:live · effort:trivial · confidence:high
- **What:** The test does `from run_ordinal_anchor import (..., _build_digit_loaders, ...)` (line 14), but run_ordinal_anchor neither defines nor re-exports _build_digit_loaders. That symbol is defined in run_correspondence_anchor_v2.py (line 67); run_ordinal_anchor imports several helpers from that module (lines 51-58: _hash_params, _load_cached_features, build_paired_feature_dataset, compute_held_out_metrics, run_procrustes, train_correspondence) but NOT _build_digit_loaders. run_ordinal_anchor only defines a differently-named _build_test_loaders (line 75). Result: `ImportError: cannot import name '_build_digit_loaders' from 'run_ordinal_anchor'`. The module raises before main() ever runs, so the __main__ guard is moot and this is the actual, live crash — earlier than the missing-data/checkpoint issues.
- **Evidence:** code/test_ordinal_anchor_smoke.py:14 imports _build_digit_loaders from run_ordinal_anchor; run_ordinal_anchor.py:51-58 import list omits it; _build_digit_loaders defined in run_correspondence_anchor_v2.py:67; run_ordinal_anchor.py:75 defines only _build_test_loaders.
- **Fix:** Import _build_digit_loaders from run_correspondence_anchor_v2 (where it lives), or add it to run_ordinal_anchor's re-export block (run_ordinal_anchor.py:51-58). Confirm the intended loader: run_ordinal_anchor exposes _build_test_loaders, which may be the one actually wanted.
- **Fix at:** `code/test_ordinal_anchor_smoke.py:11-19`, `code/run_ordinal_anchor.py:51-58`
- _related: TST-017, TST-018_


---
## MED (112)

#### `ARCH-022` [MED] Stale hardcoded absolute sys.path to /Users/agent1/... (different machine/user)
`code/archive/debug_gradient.py:5` · dependencies · status:archive · effort:trivial · confidence:high
- **What:** Both legacy debug/benchmark scripts sys.path.insert a hardcoded absolute path '/Users/agent1/Documents/neuromorphic/m7-trimodal-cl' that does not exist on this machine (repo lives under /Users/zainhassan/...). The insert is dead and misleading; the subsequent 'from Models.registry import build_model' then only resolves if the CWD happens to be the repo root (Models/ at repo root on the implicit path). Portable only by accident.
- **Evidence:** debug_gradient.py:5 sys.path.insert(0,'/Users/agent1/Documents/neuromorphic/m7-trimodal-cl'); benchmark_mps.py:6 identical path; Models.registry import at debug_gradient.py:9 and benchmark_mps.py:9.
- **Fix:** None for retirement (both DELETE per ARCH-008/ARCH-009). If ever revived, replace with a REPO_ROOT-relative sys.path (Path(__file__).resolve().parents[...]).
- **Fix at:** `code/archive/debug_gradient.py:5`, `code/archive/benchmark_mps.py:6`
- _related: ARCH-008, ARCH-009_

#### `ARCH-018` [MED] forgetting_T1_T3 stores the incremental T2->T3 drop, not total T1->T3 forgetting
`code/archive/run_cl_confirmation.py:412` · correctness · status:archive · effort:trivial · confidence:medium
- **What:** forgetting_T1_T3 = acc_t1_after_t2 - acc_t1_after_t3 (line 412, self-labeled 'incremental T2->T3'). Its name implies total forgetting of T1 through T3, but total T1 forgetting is total_forgetting_T1_T3 = acc_t1_after - acc_t1_after_t3 (line 414). A downstream reader/aggregator that treats the column named forgetting_T1_T3 as total forgetting will understate/mislabel the headline forgetting metric — sensitive given RULE 5 (the forgetting claim must be reported honestly).
- **Evidence:** forgetting_T1_T3 = acc_t1_after_t2 - acc_t1_after_t3 line 412; total_forgetting_T1_T3 = acc_t1_after - acc_t1_after_t3 line 414; both written to CSV lines 437,439 and re-read by aggregate_confirmation_results metrics list lines 486-487.
- **Fix:** Rename the column to forgetting_T1_incremental_T2_T3 (or similar) so the incremental vs total distinction is unambiguous in the CSV consumed downstream. Retirement: note the ambiguity for anyone mining old cl_confirmation CSVs.
- **Fix at:** `code/archive/run_cl_confirmation.py:412`, `code/archive/run_cl_confirmation.py:437`
- _related: ARCH-019_

#### `DEP-013` [MED] Archive scripts compute REPO_ROOT one directory too shallow after relocation
`code/archive/run_cl_confirmation.py:49-51` · dependencies · status:live · effort:trivial · confidence:high
- **What:** REPO_ROOT = Path(__file__).resolve().parent.parent. For a file at code/archive/X.py this resolves to code/, not the repo root. So sys.path.insert(0, REPO_ROOT / 'code') points to code/code/ (nonexistent) and sys.path.insert(0, REPO_ROOT) adds code/ (not the repo root). The repo-local `from m7_trimodal_run_provenance import ...` still resolves (that module is in code/), but `from Dataloaders...` / `from utils.cross_modal_transfer...` cannot, because the legacy package lives at the repo root which is never added. These files worked before being moved into archive/; the parent.parent depth was not updated for the extra directory level.
- **Evidence:** code/archive/run_cl_confirmation.py:49 REPO_ROOT=Path(__file__).resolve().parent.parent ; :50-51 sys.path.insert(REPO_ROOT/'code'), sys.path.insert(REPO_ROOT) ; :204 from Dataloaders.paired_digit_loader import ... . Same shape in code/archive/run_correspondence_anchor.py:39-41 with :51 from utils.cross_modal_transfer import ...
- **Fix:** For archive-depth files use parent.parent.parent (or search upward for a repo-root marker) so REPO_ROOT is the actual repo root.
- **Fix at:** `code/archive/run_cl_confirmation.py:49-51`, `code/archive/run_correspondence_anchor.py:39-41`
- _related: DEP-011_

#### `ARCH-001` [MED] Superseded v1 of run_correspondence_anchor_v2.py
`code/archive/run_correspondence_anchor.py:1-767` · dead-code · status:archive · effort:trivial · confidence:high
- **What:** This is the v1 correspondence-anchor experiment. The live successor code/run_correspondence_anchor_v2.py (34 KB) exists and is the current version. v1 is import-safe (main() guarded, line 766) but dead: imported by nobody, and it also carries a real held-out-class bug (ARCH-015) that v2 presumably fixes. It cannot run today anyway because CKPT_SOURCE is missing (ARCH-005).
- **Evidence:** code/run_correspondence_anchor_v2.py present and live; docstring/design here matches the v2 lineage; no importer of this module.
- **Fix:** Retirement verdict DELETE: v2 is the record of this experiment. If retained for provenance, add a header comment marking it superseded and noting the held-out masking no-op (ARCH-015) so its printed held-out numbers are not trusted.
- **Fix at:** `code/archive/run_correspondence_anchor.py:1-767`
- _related: ARCH-005, ARCH-014, ARCH-015_

#### `ARCH-014` [MED] Hardcoded baseline numbers printed in a BASELINE COMPARISON table (RULE 1)
`code/archive/run_correspondence_anchor.py:757-761` · research-integrity · status:archive · effort:trivial · confidence:high
- **What:** main() prints a 'BASELINE COMPARISON' table whose baseline rows are hardcoded literals — A_no_align '~97.0% / ~13.3%', Frozen-C '13.3% / 14.8%', Unfrozen+rehearsal '95.7% / 14.4%' — interleaved with the run's own computed mean_retention/mean_v2a on the final row. Under RULE 1, hardcoded numbers presented next to measured ones as a results table is exactly the pattern that poisons trust; the '~' values read as estimates. These go to stdout only (not paper/CSV), and the file is archived, which bounds the harm.
- **Evidence:** literals hardcoded in code at lines 758-761 ('~97.0%','~13.3%','13.3%','14.8%','95.7%','14.4%'); computed values mean_v2a/mean_retention lines 755-756.
- **Fix:** If this experiment is ever revived (in v2), source every baseline number from its CSV or drop the row; never print hardcoded comparison numbers beside measured ones. For retirement, this is one more reason to DELETE v1 (ARCH-001).
- **Fix at:** `code/archive/run_correspondence_anchor.py:757-761`
- _related: ARCH-001_

#### `ARCH-013` [MED] Sim-vs-sim agreement is NOT SpiNNaker-hardware verification; no energy/latency producible
`code/archive/verify_brian2_raw.py:180` · research-integrity · status:archive · effort:trivial · confidence:medium
- **What:** The cluster checks that a PyTorch discrete-time simulator matches Brian2/PyNN IF_curr_exp and prints verdicts like 'VERIFIED: PyTorch discrete-time simulator matches Brian2/PyNN'. Brian2 is a SOFTWARE simulator; comments assert the equations are 'exact same as PyNN/SpiNNaker' (verify_brian2_raw.py:180; docstring line 6). Per project rules (mapping != inference != real hardware; nothing has run on a board), treating a 'VERIFIED' stdout line as evidence of on-chip correctness would be an overclaim. No energy or latency figure is or can be produced by these scripts — a simulator cannot yield real hardware energy. No hardcoded result numbers were found in the cluster itself (all counts/matches are computed).
- **Evidence:** verify_brian2_raw.py:180 comment 'exact same as PyNN/SpiNNaker'; verify_full.py:343 'VERIFIED ... matches Brian2/PyNN'; verify_simulators.py:348 same; no energy/accuracy literals presented as results in the cluster.
- **Fix:** Documentation-level: if any of this is cited, label it 'PyTorch-vs-Brian2 software-simulator numerical agreement', never 'SpiNNaker-verified'. Do not derive energy/latency from it. Keep the hardware claims in HARDWARE_ACCESS_CHECKLIST/SpiNNaker plan as unchecked hypotheses.
- **Fix at:** `code/archive/verify_brian2_raw.py:6`, `code/archive/verify_brian2_raw.py:180`, `code/archive/verify_full.py:342-349`, `code/archive/verify_simulators.py:347-348`
- _related: ARCH-003, ARCH-005_

#### `DEP-007` [MED] Third-party 'brian2' (Brian2) imported but absent from requirements.txt
`code/archive/verify_brian2_raw.py:20` · dependencies · status:live · effort:small · confidence:high
- **What:** brian2 is imported in five archive hardware-simulation scripts and is not declared. A clean install then running any of them raises ImportError. Archive-only, but these are the Brian2/SpiNNaker verification utilities, so they are not dead in intent.
- **Evidence:** code/archive/debug_brian2_exact.py:14, code/archive/verify_brian2_raw.py:20, code/archive/verify_full.py:21, code/archive/verify_simulators.py:22, code/archive/verify_tiny.py:9 (from brian2 import ...)
- **Fix:** If these archive scripts are to remain runnable, add Brian2 (pip name: Brian2) as an optional/extra dependency (e.g. a hardware extra), not the core runtime.
- **Fix at:** `code/requirements.txt:13`, `code/archive/debug_brian2_exact.py:14`, `code/archive/verify_brian2_raw.py:20`, `code/archive/verify_full.py:21`, `code/archive/verify_simulators.py:22`, `code/archive/verify_tiny.py:9`
- _related: DEP-008_

#### `ARCH-010` [MED] Brian2/PyNN verify cluster imports brian2 and/or pyNN — absent from requirements.txt
`code/archive/verify_full.py:20-21` · dependencies · status:archive · effort:small · confidence:high
- **What:** Every script in the verification cluster imports brian2 and/or pyNN.brian2, neither of which appears in code/requirements.txt (which pins torch/snntorch/tonic/transformers/numpy/pandas/matplotlib/tqdm/h5py/scipy/pytest only). In the declared environment these files raise ModuleNotFoundError at import, i.e. all five are un-runnable as documented. This is a whole-cluster dependency gap, not five separate bugs.
- **Evidence:** verify_full.py:20-21 (pyNN+brian2); verify_simulators.py:21-22; verify_brian2_raw.py:20-25 (brian2 only); verify_tiny.py:8-9; debug_brian2_exact.py:13-14. requirements.txt has no brian2/pyNN (grep: NONE).
- **Fix:** Do NOT add brian2/pyNN to requirements.txt just for archived scripts. If any verification is resurrected into live use, pin brian2 and pyNN (and sPyNNaker for a board) in a separate optional/dev requirements file and gate the import. Otherwise treat as documentation of an un-runnable historical env.
- **Fix at:** `code/archive/verify_full.py:20-21`, `code/archive/verify_simulators.py:21-22`, `code/archive/verify_brian2_raw.py:20-25`, `code/archive/verify_tiny.py:8-9`, `code/archive/debug_brian2_exact.py:13-14`
- _related: ARCH-003, ARCH-004, ARCH-005, ARCH-006, ARCH-007_

#### `ARCH-011` [MED] ConvFrontend class copy-pasted across 5 files (4 archived + 1 live)
`code/archive/verify_full.py:26-46` · duplication · status:archive · effort:medium · confidence:high
- **What:** An identical ConvFrontend nn.Module (conv1 2->32 k5, pool, LIF beta=0.9; conv2 32->64 k5, pool, LIF; flatten; sum over time) is duplicated verbatim in five places, including the LIVE code/final_evaluation.py. Five divergent copies of the visual backbone frontend are a maintenance hazard: a numerics fix to one (e.g. reset semantics, beta) will silently not propagate to the others.
- **Evidence:** verify_full.py:26-46; verify_simulators.py:27-47; verify_brian2_raw.py:30-50; debug_brian2_exact.py:19-39; code/final_evaluation.py:33-53 (LIVE). All byte-identical bodies.
- **Fix:** Extract ONE ConvFrontend into a shared, import-safe module (e.g. code/spinnaker_frontend.py) and import it in final_evaluation.py; retire the four archived copies with the files. Verify the extraction is numerically inert (same beta, same reset_hidden order) before touching the live final_evaluation.py.
- **Fix at:** `code/archive/verify_full.py:26-46`, `code/archive/verify_simulators.py:27-47`, `code/archive/verify_brian2_raw.py:30-50`, `code/archive/debug_brian2_exact.py:19-39`, `code/final_evaluation.py:33-53`
- _related: ARCH-003, ARCH-004, ARCH-005, ARCH-007_

#### `ARCH-012` [MED] Loads missing checkpoint ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt
`code/archive/verify_full.py:259` · dependencies · status:archive · effort:small · confidence:high
- **What:** Four cluster scripts torch.load a hard-coded checkpoint path ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt. ICONS_M7/ does not exist in the repo (no checkpoints anywhere), so each crashes with FileNotFoundError before any verification runs. The path is also CWD-relative (only resolves from code/). verify_tiny.py is exempt (synthetic, no checkpoint).
- **Evidence:** verify_full.py:259; verify_simulators.py:265; verify_brian2_raw.py:290; debug_brian2_exact.py:44. ICONS_M7 absent (ls: ABSENT).
- **Fix:** None for retirement — record that these are un-runnable without the (missing, gitignored) checkpoint. If resurrected, parameterize the path via argument/env and guard existence; also note torch.load without weights_only=True is an unsafe-deserialization pattern.
- **Fix at:** `code/archive/verify_full.py:259`, `code/archive/verify_simulators.py:265`, `code/archive/verify_brian2_raw.py:290`, `code/archive/debug_brian2_exact.py:44`
- _related: ARCH-010, ARCH-011_

#### `DEP-008` [MED] Third-party 'PyNN' imported but absent from requirements.txt
`code/archive/verify_full.py:20` · dependencies · status:live · effort:small · confidence:high
- **What:** `import pyNN.brian2 as sim` appears in four archive scripts; the pip package is PyNN. Not declared. Archive-only hardware-simulation path; ImportError on a clean install.
- **Evidence:** code/archive/debug_brian2_exact.py:13, code/archive/verify_full.py:20, code/archive/verify_simulators.py:21, code/archive/verify_tiny.py:8 (import pyNN.brian2 as sim)
- **Fix:** Add PyNN (pip name: PyNN) as an optional/hardware extra if these archive scripts are to remain runnable; it also pulls Brian2 (DEP-007).
- **Fix at:** `code/requirements.txt:13`, `code/archive/debug_brian2_exact.py:13`, `code/archive/verify_full.py:20`, `code/archive/verify_simulators.py:21`, `code/archive/verify_tiny.py:8`
- _related: DEP-007_

#### `DEAD-017` [MED] benchmark_mps_real.py is a dead one-off benchmark that imports the legacy bi-modal package
`code/benchmark_mps_real.py:1-54` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** One-off throughput benchmark. It imports the LEGACY package - Models.registry.build_model (L9) and Dataloaders.nmnist_loader.get_nmnist_loaders (L11) - not the M7 code, and runs the benchmark at module scope (L19-54) with no __main__ guard, so importing it would try to load N-MNIST and time training. Not part of the M7 pipeline and out of scope of the maintained code.
- **Evidence:** from Models.registry import build_model L9; from Dataloaders.nmnist_loader import get_nmnist_loaders L11; module-scope benchmark loop L19-54; grep '__main__' returns 0.
- **Fix:** Delete.
- **Fix at:** `code/benchmark_mps_real.py:1-54`
- _related: DEAD-018_

#### `DEP-012` [MED] Hardcoded foreign absolute path in sys.path.insert breaks repo-local imports off the origin machine
`code/benchmark_mps_real.py:6` · dependencies · status:live · effort:trivial · confidence:high
- **What:** sys.path.insert(0, '/Users/agent1/Documents/neuromorphic/m7-trimodal-cl') hardcodes another user's directory. That path does not exist on this machine (user zainhassan; repo is at /Users/zainhassan/Documents/cross-modal-neuromorphic), so the subsequent legacy imports `from Models.registry import build_model` (line 9) and `from training.pipeline import train_step` cannot resolve -> guaranteed ImportError whenever the script is run here. It also leaks a foreign filesystem layout. Live file code/benchmark_mps_real.py plus two archive files repeat the exact literal.
- **Evidence:** code/benchmark_mps_real.py:6 sys.path.insert(0, '/Users/agent1/Documents/neuromorphic/m7-trimodal-cl') ; code/archive/benchmark_mps.py:6 (same) ; code/archive/debug_gradient.py:5 (same). benchmark_mps_real.py:9 from Models.registry import build_model
- **Fix:** Replace the hardcoded path with a portable insert such as sys.path.insert(0, str(Path(__file__).resolve().parent.parent)), or adopt proper packaging (see DEP-011).
- **Fix at:** `code/benchmark_mps_real.py:6`, `code/archive/benchmark_mps.py:6`, `code/archive/debug_gradient.py:5`
- _related: DEP-011_

#### `DEAD-009` [MED] convert_mps.py is a dead one-off codegen whose only output (m7_mps.py) is also dead
`code/convert_mps.py:1-182` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** One-off script that string-transforms m7_raw.py into m7_mps.py. Both its input (m7_raw.py) and output (m7_mps.py) are retired notebook monoliths superseded by code/m7_trimodal_run_provenance.py, which already auto-selects MPS then CUDA then CPU. The tool has no remaining purpose.
- **Evidence:** reads m7_raw.py L7-8; writes m7_mps.py L173-174.
- **Fix:** Delete alongside m7_raw.py and m7_mps.py.
- **Fix at:** `code/convert_mps.py:1-182`
- _related: DEAD-010, DEAD-011_

#### `DEAD-011` [MED] convert_mps.py's exact-string .replace() transforms silently no-op when m7_raw.py drifts
`code/convert_mps.py:70-108` · correctness · status:dead · effort:small · confidence:high
- **What:** The port relies on brittle literal .replace()/regex edits with no verification that any substitution matched. Two are already stale versus current m7_raw.py: (a) the EmbeddingReplayBuffer target expects a 'device=None' third line (convert_mps L91-93) but m7_raw L1774-1776 has 'device: str = 'cuda'', and the regex at L70-84 only matches bare device='cuda' (not the annotated form), so neither the regex nor the exact block fires and auto-detect is never inserted; (b) the ContinualLearner target expects '... device=None, lr=1e-3, weight_decay=1e-4, gradient_clip=1.0' (L102-103) but m7_raw L1883-1884 is '... device='cuda', replay_batch=32, replay_weight=0.3'. A regenerated m7_mps.py would silently keep a CUDA default and break on MPS/CPU, with no error raised. This is why regeneration is unsafe (DEAD-004).
- **Evidence:** convert_mps L70-84 regex; L91-99 old_er_init; L102-108 old_cl_init; vs m7_raw L1774-1776 (device: str = 'cuda') and L1883-1884 (device='cuda', replay_batch=32, replay_weight=0.3).
- **Fix:** Delete. If a similar tool is ever needed, assert each replacement changed the text (count matches) and fail loudly on zero matches; never patch a moving source with literal strings.
- **Fix at:** `code/convert_mps.py:70-108`
- _related: DEAD-004, DEAD-009_

#### `CLE-010` [MED] Misleading 'transfer did not survive' verdict when post_T2 is simply absent
`code/eval_transfer_fixed.py:126-137` · correctness · status:live · effort:small · confidence:high
- **What:** When df is non-empty (e.g. only post_T3 present) but has no post_T2 row, the ternary falls to `else 0`, so post_t2_v2a=0 and the code prints the failure branch: 'Transfer DID NOT survive. V2A = 0.0% (near chance).' This reports a fabricated 0.0% and a scientific conclusion for a condition that was never evaluated -- absence of data is silently rendered as a negative result.
- **Evidence:** Line 126 `else 0`; lines 134-137 print the failure verdict using post_t2_v2a.
- **Fix:** Distinguish 'not evaluated' (None) from a measured low value; only emit a verdict when post_T2 was actually scored.
- **Fix at:** `code/eval_transfer_fixed.py:126-137`
- _related: CLE-009_

#### `CLE-011` [MED] Magic verdict thresholds >20 and >12 for V2A survival
`code/eval_transfer_fixed.py:127-131` · numerics · status:live · effort:small · confidence:medium
- **What:** The verdict tiers hardcode V2A>20 (SURVIVED) and V2A>12 (PARTIALLY) with no derivation or link to chance. Chance here is 100/num_classes = 10% (num_classes=10 at line 95), so 20 and 12 are ~2x and ~1.2x chance, but that rationale is undocumented and the constants are unnamed, making the pass/fail scientific call opaque and easy to drift.
- **Evidence:** Lines 127 `if post_t2_v2a > 20:` and 131 `elif post_t2_v2a > 12:`.
- **Fix:** Define named thresholds relative to the eval's own chance (e.g. chance = 100/num_classes; survived if V2A > chance + margin) and document the margin choice.
- **Fix at:** `code/eval_transfer_fixed.py:127`, `code/eval_transfer_fixed.py:131`

#### `FIG-022` [MED] torch.load of a missing checkpoint crashes the only run path; CWD-relative path; name lacks seed
`code/final_evaluation.py:207` · correctness · status:live · effort:small · confidence:high
- **What:** main() calls torch.load('ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt', map_location='cpu') (207). The repo has NO checkpoints, so the script raises FileNotFoundError immediately - the entire evaluation depends on this file. The path is CWD-relative (no './'; resolves only from code/, and note it uses 'ICONS_M7/...' whereas the figure scripts use './ICONS_M7'). The filename 'final_nmnist_nmnist_replay.pt' also lacks the seed field of the documented pattern final_{task}_{task}_seed{seed}_{condition}.pt. Unlike generate_figures_3seed.py (which sys.exits with guidance on a missing CSV), this gives a bare traceback. Minor: torch.load is called without weights_only=; torch 2.11 (pinned) defaults weights_only=True (secure-by-default), which also means a non-tensor-only checkpoint would fail here.
- **Evidence:** torch.load('ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt', map_location='cpu')
- **Fix:** Check the path exists and emit an informative message (mirror the CSV-missing pattern); make the path './ICONS_M7/...'; confirm the intended checkpoint filename/seed convention; pass weights_only explicitly.
- **Fix at:** `code/final_evaluation.py:207`
- _related: FIG-009_

#### `FIG-023` [MED] collect_stratified silently under-collects if a class has <200 samples; assumes batch_size=1; KeyError if label>=10
`code/final_evaluation.py:88-99` · correctness · status:live · effort:small · confidence:high
- **What:** collect_stratified (88-99) fills up to per_class (200) samples/class, breaking early once all classes are satisfied (97-98). If any class has fewer than per_class samples in the loader, the all(...) condition never becomes True and the loop simply exhausts the finite loader and returns fewer samples than requested - with NO error - while main() still prints '200/class' (200/223). It also assumes batch_size=1 via int(y.item()) (92) - y.item() raises if a batch has >1 element (batch_size=1 is set at 221, so this is a coupled assumption). counts is keyed 0..9 (90); a label >=10 would KeyError. Not an infinite loop (the for-loop is bounded by the finite loader).
- **Evidence:** counts = {i: 0 for i in range(10)}; break when all(c >= per_class ...)
- **Fix:** After the loop, assert each class reached per_class (or report actual per-class counts); derive the class set from the data rather than hardcoding range(10); document the batch_size=1 requirement.
- **Fix at:** `code/final_evaluation.py:88-99`

#### `FIG-024` [MED] Headline accuracy depends on hand-tuned SCALE=7.0 and other magic scale factors whose calibration is not in-repo
`code/final_evaluation.py:27-29` · research-integrity · status:live · effort:medium · confidence:medium
- **What:** The reported sim accuracy is sensitive to several hand-set scalars: SCALE=7.0 (27, applied to weights at 120), RATE_SCALE=10.0 (28, applied at 111), bias*10.0 (123), and probs=rates/1000.0 (112). The docstring/statement assert SCALE 'was calibrated once and never adjusted' (8/304), but the calibration procedure and its target are not present in this file or (per repo context) in any CSV, so the choice of 7.0 is an undocumented researcher degree-of-freedom that directly moves the headline number. These are also undocumented magic constants (with DURATION_MS=100 at 29, refrac_steps=2 at 134, thresholds -50/-65, tau 20/5 at 130-133).
- **Evidence:** SCALE = 7.0 (locked, no re-calibration)
- **Fix:** Document how SCALE=7.0 (and the *10 bias / rate scaling) were calibrated, against what metric and data, ideally with a sourced artifact; name the LIF magic constants and cite build_spinnaker_network() as their single source.
- **Fix at:** `code/final_evaluation.py:27-29`, `code/final_evaluation.py:111-123`, `code/final_evaluation.py:130-134`
- _related: FIG-020_

#### `RA-011` [MED] '95% CI' is actually mean +/- 1.96*std (missing /sqrt(n))
`code/final_evaluation.py:289` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** The printed interval is [mean-1.96*std, mean+1.96*std] labelled '95% CI'. A 95% CI of the mean is mean +/- 1.96*std/sqrt(n); what is computed is a ~95% normal spread of individual seeds, ~sqrt(n) wider. The label contradicts the statistic. The script also cannot currently run (loads a nonexistent checkpoint at line 207), so this is a latent mislabel, not a produced number.
- **Evidence:** line 289; verdict CONTRADICTED
- **Fix:** Either add /np.sqrt(len(SEEDS)) and keep the '95% CI' label, or rename the output to 'mean +/- 1.96 sigma (spread)'.
- **Fix at:** `code/final_evaluation.py:289`
- _confirmed by FIG-018; related: RA-013_

#### `RA-006` [MED] Hardcoded replay-memory bars and compression factors
`code/generate_figures.py:157-176` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** memory=[80,150,4] MB (line 157) and the '20x smaller'/'37x smaller' annotations (173-176) are literals with no derivation. generate_figures_3seed.py:283-284 documents this exact [80,150,4] as the removed hardcode.
- **Evidence:** lines 157, 173-176; verdict UNSOURCED
- **Fix:** Delete; the 3seed version derives this from CONFIG with the arithmetic printed.
- **Fix at:** `code/generate_figures.py:157`, `code/generate_figures.py:173-176`
- _related: RA-002, RA-014_

#### `FIG-009` [MED] No __main__ guard: module body (incl. load_summary) runs at import
`code/generate_figures_3seed.py:93-94` · safety · status:live · effort:small · confidence:high
- **What:** generate_figures_3seed.py has no `if __name__=='__main__'` guard (grep: 0). The whole analysis executes at import: csv_path assignment and load_summary(csv_path) at 93-94, aggregation at 96, and all five figures at 136-378. Because the results CSV does not exist, importing calls sys.exit() inside load_summary (58-62) - a clean, informative exit, so this is far less dangerous than generate_figures.py. But if the CSV DID exist, a mere import would render and write five PNGs (and it always creates ./ICONS_M7/results/plots via the module-level mkdir at 30). Static inspection therefore cannot import this module; it must be read as text.
- **Evidence:** grep __main__ -> 0 hits; module-level load_summary at 94
- **Fix:** Wrap the module body in main() behind `if __name__=='__main__'` so importing is side-effect-free; keep the informative missing-CSV exit inside main().
- **Fix at:** `code/generate_figures_3seed.py:93-94`, `code/generate_figures_3seed.py:136-378`
- _related: FIG-029_

#### `DEAD-004` [MED] m7_mps.py has drifted from its generator m7_raw.py and is no longer reproducible by convert_mps.py
`code/m7_mps.py:2-23` · architecture · status:dead · effort:trivial · confidence:high
- **What:** m7_mps.py is meant to be a pure machine translation of m7_raw.py via convert_mps.py, but the two have diverged: the !pip lines are COMMENTED in m7_mps.py (L2-4/L17-23) yet UNCOMMENTED in m7_raw.py, and convert_mps.py never edits pip lines - so the current m7_mps.py cannot have been produced by running convert_mps.py on the current m7_raw.py. Regenerating is also unsafe because convert_mps.py's exact-string targets are themselves stale (see DEAD-011). Treat m7_mps.py as an orphaned snapshot, not a derived file that can be rebuilt.
- **Evidence:** m7_raw.py L2 '!pip...' vs m7_mps.py L2 '# !pip...'; convert_mps.py performs no pip edits (only device/seed/cache/print blocks, L14-167).
- **Fix:** Delete m7_raw.py and m7_mps.py together; do not attempt to reconcile or regenerate. Maintained source is m7_trimodal_run_provenance.py.
- **Fix at:** `code/m7_mps.py:1-3900`, `code/m7_raw.py:1-3877`
- _related: DEAD-011_

#### `RA-007` [MED] Hardcoded Paper-2 M1-M5 reference block under a 'FINAL NUMBERS FOR PAPER' header
`code/m7_mps.py:2284-2288` · research-integrity · status:dead · effort:small · confidence:high
- **What:** Printed literals for N-MNIST/SHD M1-M5 plus 'Best N-MNIST 97.68%', 'Best SHD 82.16%', 'Cross-modal gap 21.53pp'. Disclosed as 'Paper 2 reference' so not mislabelled as M7, but hardcoded with no CSV, and one block sits under a 'FINAL NUMBERS FOR PAPER - NeurIPS 2026' header (3839). Near-identical copies in m7_raw.py and m7_trimodal_cl.py.
- **Evidence:** m7_mps.py:2284-2288 and 3845-3846; copies m7_raw.py:2261-2264,3822-3823; m7_trimodal_cl.py:2211-2214; verdict UNSOURCED
- **Fix:** Archive/delete these three superseded two-task notebook exports; or load the reference row from a CSV.
- **Fix at:** `code/m7_mps.py:2284-2288`, `code/m7_mps.py:3845-3846`, `code/m7_raw.py:2261-2264`, `code/m7_trimodal_cl.py:2211-2214`
- _related: DEAD_

#### `RA-008` [MED] Hardcoded 'Independent M7' reference lines 97.70 / 82.21
`code/m7_mps.py:3716-3727` · research-integrity · status:dead · effort:trivial · confidence:high
- **What:** axhline(97.70,'Independent M7') and axhline(82.21,'Independent M7') draw single-task M7 baselines as hardcoded literals (82.21 matches the two-task onboarding-brief SHD number). No CSV backs them. Copies in m7_raw.py:3704 and m7_trimodal_cl.py:3654.
- **Evidence:** m7_mps.py:3716,3727; verdict UNSOURCED
- **Fix:** Delete the axhlines or source the baseline from a CSV.
- **Fix at:** `code/m7_mps.py:3716`, `code/m7_mps.py:3727`, `code/m7_raw.py:3704`, `code/m7_trimodal_cl.py:3654`
- _related: RA-007_

#### `RA-009` [MED] Energy-efficiency claim rests on an assumed 5% spike rate
`code/m7_mps.py:3305-3434` · research-integrity · status:dead · effort:medium · confidence:high
- **What:** estimate_synops_analytical hardcodes SPIKE_RATE=0.05 ('estimated from Paper 2') and computes snn/ann energy and an efficiency_x fold-factor written to energy_results.json. The spike rate that must accompany accuracy has to be measured from the trained model, not assumed; the result is an analytical estimate on an unmeasured input and is not a hardware energy result. Copies in m7_raw.py:3290 and m7_trimodal_cl.py:3240.
- **Evidence:** const at m7_mps.py:3313; cell 3305-3434; verdict UNSOURCED
- **Fix:** Measure spike rate from a real forward pass into a CSV and drive the estimate from that; label output 'analytical estimate, not hardware-measured'.
- **Fix at:** `code/m7_mps.py:3313`, `code/m7_mps.py:3305-3434`
- _related: RA-010_

#### `RA-010` [MED] SynOps energy uses Hopfield 256 patterns, but M7 config is 100
`code/m7_mps.py:3346` · research-integrity · status:dead · effort:trivial · confidence:high
- **What:** The shared-component term 512*256*SPIKE_RATE uses 256 patterns. M7's call site sets num_patterns=100; 256 is the class default / legacy bi-modal value. The energy estimate counts a component the actual M7 model does not have. Copies m7_raw.py:3323, m7_trimodal_cl.py:3273.
- **Evidence:** m7_mps.py:3346; verdict CONTRADICTED
- **Fix:** Replace 256 with the config value (read CONFIG['num_patterns']=100); do not hardcode.
- **Fix at:** `code/m7_mps.py:3346`, `code/m7_raw.py:3323`, `code/m7_trimodal_cl.py:3273`
- _related: RA-009_

#### `DEAD-001` [MED] m7_raw.py is a dead Kaggle/Jupyter notebook dump superseded by m7_trimodal_run_provenance.py
`code/m7_raw.py:1-3877` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** Raw export of the Era-1 notebook (retains '# %% CELL N' cell markers). Re-defines the 7 core M7 classes as copy-paste clones: SupervisedContrastiveLoss (L614), ModernHopfieldLayer (L692), ImprovedHGRNGate (L774), SNN_Backbone (L868), M7_ContinualAdaptiveModel (L1615), EmbeddingReplayBuffer (L1761), ContinualLearner (L1867); plus the entire legacy bi-modal model zoo (Model_1..5_SHD L1068-1158, Model_1..5 baselines L1158-1275) and ablation classes (EWC L2502, M7_NoSCL/M7_FixedGate/M7_SharedHead L2893-2921). The maintained implementation of the same classes is code/m7_trimodal_run_provenance.py (guarded, import-safe, L151-527). Imported by nobody.
- **Evidence:** class defs at listed lines; same class names present in code/m7_trimodal_run_provenance.py L151-527; no other module imports m7_raw.
- **Fix:** Delete, or move to code/archive/ if a provenance snapshot is wanted. Do not import (also blocked by DEAD-002 SyntaxError).
- **Fix at:** `code/m7_raw.py:1-3877`
- _related: DEAD-002, DEAD-003, DEAD-004_

#### `DEAD-006` [MED] m7_trimodal_cl.py manufactures paper_tables.tex and 'N x more efficient' energy figures from an analytical estimate
`code/m7_trimodal_cl.py:3329-3577` · research-integrity · status:dead · effort:trivial · confidence:medium
- **What:** On import it computes an analytical SynOps energy estimate, renders it as per-inference pJ and a figure titled '... x more efficient' (fig_energy.pdf, last ~25 lines), and writes LaTeX paper_tables.tex (L3465-3577). These are analytical/simulator numbers, not measured hardware energy (contested-facts item 6: a simulator cannot produce energy figures; RULE 1: no number without a CSV). If this dead script is re-run and its output lifted into the paper, it injects unsourced energy/efficiency claims. Retirement should remove this number-manufacturing path, not just the clutter.
- **Evidence:** L3329 'Analytical SynOps Energy Analysis'; energy_out dict L3354; paper_tables.tex write L3569; figure title includes 'x ... more efficient' near L3745.
- **Fix:** Delete the file; never reuse its energy/table output. Any energy number must originate from a CSV per RULE 1.
- **Fix at:** `code/m7_trimodal_cl.py:3329-3361`, `code/m7_trimodal_cl.py:3465-3577`
- _related: DEAD-005_

#### `RUN-015` [MED] Collate truncates every sample to the first 25 frames, discarding most of each recording
`code/m7_trimodal_run.py:77-97` · numerics · status:live · effort:small · confidence:medium
- **What:** Both nmnist_collate_fn and dvs_collate_fn keep only e[:25] frames and zero-pad shorter ones. With ToFrame time_windows of 1000 us for N-MNIST (117) and 50000 us for DVS (165), each raw recording produces many more than 25 frames, so the truncation silently drops the later part of every sample. The retained window is fixed by a magic 25 rather than derived from CONFIG['time_steps']. This is a modeling decision that materially affects both accuracy and spike cost and is undocumented.
- **Evidence:** line 78 'e = e[:25]' (nmnist); line 95 'e = e[:25]' (dvs); ToFrame time_window=1000 at 117 and 50000 at 165
- **Fix:** Choose the ToFrame time_window and the kept-frame count together so the 25 frames cover the intended fraction of each recording, source the count from config, and document the temporal coverage per modality.
- **Fix at:** `code/m7_trimodal_run.py:77-80`, `code/m7_trimodal_run.py:94-97`, `code/m7_trimodal_run.py:115-118`, `code/m7_trimodal_run.py:164-166`
- _related: RUN-014_

#### `RUN-016` [MED] SupervisedContrastiveLoss includes the self/diagonal term in both the positive mask and the denominator
`code/m7_trimodal_run.py:190-199` · correctness · status:live · effort:small · confidence:medium
- **What:** The positive mask is 'labels==labels' which is 1 on the diagonal, and the log-prob denominator sums over all j including j==i, with no logits_mask to remove self-contrast. Standard SupCon (Khosla et al.) excludes the anchor itself from both positives and denominator. Including self gives every anchor a trivial always-satisfied positive (its own normalized similarity, the largest term after max-subtraction), diluting the contrastive gradient and biasing the loss, most visibly for classes appearing once per batch.
- **Evidence:** line 193 'mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()'; lines 196-198 build exp_logits and mean_log_prob without masking the diagonal
- **Fix:** Build a logits_mask that removes self from the denominator and from mask before the mean, matching the canonical SupCon formulation; this will move the loss and thus results, so flag it as a deliberate numeric change.
- **Fix at:** `code/m7_trimodal_run.py:190-199`

#### `RUN-017` [MED] Entire SHD train+test set is densified into a Python list of [100,1,1,700] tensors at import
`code/m7_trimodal_run.py:149-150` · performance · status:live · effort:medium · confidence:high
- **What:** shd_train_data and shd_test_data are built by list comprehensions that convert every SHD sample to a dense [100,700] float tensor (unsqueezed to [100,1,1,700]) and hold them all in memory before training. events_to_dense also fills the dense tensor with a per-event Python for-loop (145-147) rather than vectorized indexing. This is eager full-dataset materialization at import (compounding RUN-001) and is both memory-heavy and slow.
- **Evidence:** lines 149-150 'shd_train_data = [events_to_dense(e, l) for e, l in tqdm(shd_train...)]' and same for shd_test; per-event loop 145-147
- **Fix:** Densify lazily inside a Dataset/transform (as N-MNIST and DVS do via tonic) or cache to disk, and vectorize the t/x index assignment instead of the Python loop.
- **Fix at:** `code/m7_trimodal_run.py:138-160`
- _related: RUN-001, RUN-033_

#### `RUN-018` [MED] Backbone fc1=Linear(64*5*5,1024) hardcodes a 34x34 input, forcing DVS to be pooled to 34x34 in collate
`code/m7_trimodal_run.py:242` · architecture · status:live · effort:medium · confidence:high
- **What:** SNN_Backbone.fc1 assumes the post-conv spatial size that results only from a 34x34 input (conv5->30, pool->15, conv5->11, pool->5 => 64*5*5=1600). To satisfy this fixed fan-in, dvs_collate_fn adaptively pools native 128x128 DVS frames down to 34x34 (98-102), a large downsampling for a visual-motion task. The coupling is implicit: no assertion, nothing derives the fan-in from input size, so any input-resolution change silently breaks the layer. (This 1600x1024 fan-in is also the basis of the SpiNNaker core budget.)
- **Evidence:** line 242 'self.fc1 = nn.Linear(64 * 5 * 5, 1024)'; dvs collate pooling to (34,34) at lines 98-102
- **Fix:** Derive the conv output size dynamically (lazy linear or a shape probe) or assert the expected input size; document why DVS is pooled to 34x34 and the information loss it implies.
- **Fix at:** `code/m7_trimodal_run.py:242`, `code/m7_trimodal_run.py:88-104`

#### `RUN-019` [MED] Determinism only partially configured: torch+numpy global seeds set, but not deterministic algorithms or per-loader RNG
`code/m7_trimodal_run.py:522-524` · numerics · status:live · effort:small · confidence:medium
- **What:** run_one_seed sets torch.manual_seed(seed) and np.random.seed(seed) only. It does not call torch.use_deterministic_algorithms(True), does not configure MPS/CUDA determinism, and does not pin the shuffle generator of the module-level train DataLoaders (created once at import with shuffle=True and no generator). Reproducibility therefore depends on exact global-RNG consumption order across the six sequential runs sharing those loaders, which is fragile. (Python's 'random' is not imported, so that gap is moot.)
- **Evidence:** lines 523-524 'torch.manual_seed(seed)' and 'np.random.seed(seed)'; train loaders 127/155/173 use shuffle=True with no generator
- **Fix:** Set torch.use_deterministic_algorithms where supported, pass an explicitly seeded generator to each training DataLoader, and document the MPS determinism caveat so seed-to-seed variation is attributable.
- **Fix at:** `code/m7_trimodal_run.py:522-524`, `code/m7_trimodal_run.py:127-132`, `code/m7_trimodal_run.py:155-160`, `code/m7_trimodal_run.py:173-178`
- _related: RUN-034_

#### `RUN-020` [MED] Real training uses hardcoded 15 epochs / patience 5 / lr 1e-3 / wd 1e-4, overriding CONFIG and the defaults
`code/m7_trimodal_run.py:422-424` · numerics · status:live · effort:small · confidence:high
- **What:** train_task defaults to num_epochs=50, patience=12, lr=1e-3 and hardcodes weight_decay=1e-4 in the optimizer, but every call site passes num_epochs=15 and patience=5 (542, 567-570, 594-597). CONFIG['max_epochs']=50 and CONFIG['patience']=12 are therefore never used and disagree with the actual run, and weight_decay ignores CONFIG['weight_decay']. Anyone reading CONFIG or the signature to learn the training budget gets the wrong numbers.
- **Evidence:** line 422 defaults 'num_epochs=50, patience=12, lr=1e-3'; line 424 'weight_decay=1e-4'; call sites 542, 567-570, 594-597 pass num_epochs=15, patience=5
- **Fix:** Single-source the schedule from CONFIG (max_epochs, patience, learning_rate, weight_decay) and pass those values at the call sites; remove the divergent literals.
- **Fix at:** `code/m7_trimodal_run.py:422-425`, `code/m7_trimodal_run.py:542`, `code/m7_trimodal_run.py:567-570`, `code/m7_trimodal_run.py:594-597`
- _related: RUN-021_

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

#### `RUN-023` [MED] _get_features and forward duplicate the whole backbone feature-extraction, including the audio unroll verbatim
`code/m7_trimodal_run.py:302-342` · duplication · status:live · effort:small · confidence:high
- **What:** The audio-backbone temporal unroll in _get_features (307-318) is a verbatim copy of the same block in forward (326-337), and the visual branch '_, features = bb(x)' is duplicated too. Any change to feature extraction (e.g. adding spike-rate logging per RUN-011, or fixing timesteps per RUN-014) must be made in two places, which is exactly how numeric drift between the replay path and the training path is introduced.
- **Evidence:** lines 302-319 (_get_features) vs 321-342 (forward) contain the same isinstance branch and audio for-loop
- **Fix:** Extract a single _extract_features(x, modality) helper and call it from both _get_features and forward.
- **Fix at:** `code/m7_trimodal_run.py:302-342`
- _related: RUN-022_

#### `SNN-013` [MED] Two divergent M7 entrypoints produce non-comparable numbers
`code/m7_trimodal_run.py:1` · architecture · status:live · effort:large · confidence:high
- **What:** m7_trimodal_run.py vs m7_trimodal_run_provenance.py differ in RNG order (SNN-001), fwt/forgetting definitions (SNN-002/010), CSV schema (SNN-003), and seed count (SNN-011). CLAUDE.md names m7_trimodal_run.py the entrypoint and the canonical loader matches its schema, yet every run_* script imports the provenance hub - so it is ambiguous which file produced any published number, and the two are not numerically interchangeable.
- **Evidence:** schema/metric/seed differences enumerated across SNN-001/002/003/010/011
- **Fix:** Declare one source-of-record, have the other import from it (no duplicated run_one_seed/model code), and reconcile metric definitions.
- **Fix at:** `code/m7_trimodal_run.py:1`, `code/m7_trimodal_run_provenance.py:1`
- _related: SNN-002, SNN-003, SNN-011_

#### `PROV-009` [MED] '95% CI' uses the normal quantile 1.96 for tiny seed counts and returns a zero-width CI at n=1
`code/m7_trimodal_run_provenance.py:896-905` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** aggregate_results computes ci95 = 1.96 * std / sqrt(n) if n>1 else 0.0 (896). With the small seed counts this project uses (3, or the 5 in the default seeds list), the correct multiplier is the Student-t quantile (t.975 = 4.30 at n=3, 2.78 at n=5), not 1.96, so the interval is materially too narrow and mislabeled as a 95% CI. When n==1 the code emits ci95=0.0, so ci95_lower==ci95_upper==mean (903-904) — a fabricated zero-uncertainty interval rather than an undefined one. std uses ddof=1 (894), which is NaN at n=1.
- **Evidence:** ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0 (896); ci95_lower/upper = mean -/+ ci95 (903-904); std = sub.std(ddof=1) (894).
- **Fix:** Use scipy.stats.t.ppf(0.975, df=n-1) for the multiplier (scipy is already imported, 23), and emit NaN (not 0) for n<2. Label it a t-based CI.
- **Fix at:** `code/m7_trimodal_run_provenance.py:893-905`
- _related: PROV-010_

#### `PROV-010` [MED] Wilcoxon signed-rank on 3-5 paired seeds is underpowered and cannot reach significance; also a dead 'seeds' variable
`code/m7_trimodal_run_provenance.py:909-938` · research-integrity · status:live · effort:small · confidence:medium
- **What:** aggregate_results runs stats.wilcoxon(diff) on per-seed paired differences (920) where the number of pairs equals the seed count (3, or up to 5). At n=3 the smallest achievable two-sided p is ~0.25, so no result can be significant and any reported p-value is misleading; even n=5 is far below the sample size where Wilcoxon is meaningful. The guard only checks that at least one difference is nonzero (919), not that n is adequate. Separately, seeds = df['seed'].unique() (911) is assigned and never used.
- **Evidence:** stat, pval = stats.wilcoxon(diff) (920) with len(paired) potentially 3-5 (916); seeds = df['seed'].unique() unused (911).
- **Fix:** Do not report a Wilcoxon p-value at these sample sizes; report the paired mean difference and its spread, or gather many more seeds before any significance test. Remove the unused seeds variable.
- **Fix at:** `code/m7_trimodal_run_provenance.py:909-938`
- _related: PROV-009_

#### `PROV-011` [MED] random_init_baseline reseeds the global torch RNG and leaves it altered before training starts, coupling training determinism to the baseline procedure
`code/m7_trimodal_run_provenance.py:972-985` · numerics · status:live · effort:small · confidence:medium
- **What:** run_one_seed sets torch.manual_seed(cfg.seed) and np.random.seed(cfg.seed) (964-965) and constructs the real model at 971 (so model init is deterministic). It then calls random_init_baseline for each task (982-984), which internally does torch.manual_seed(cfg.seed + 1000 + s) three times per task (812) and builds throwaway models. After the loop the global torch RNG is left at the last baseline seed's consumed state, NOT at manual_seed(cfg.seed). Training then begins (997) drawing DataLoader shuffling and any stochastic ops from that leftover state. Consequently the training RNG stream depends on how many baseline seeds/tasks were run: changing n_seeds (currently 3, hardcoded at 983) or task_order silently changes training results, so the run is not numerically inert to a change that ought to be inert. The reported cfg.seed also does not cleanly map to the training RNG state.
- **Evidence:** torch.manual_seed(cfg.seed) (964); model built (971); random_init_baseline(...) loop (982-984) calls torch.manual_seed(cfg.seed + 1000 + s) (812) and builds models (813); training starts at 997 with no reseed.
- **Fix:** Isolate baseline RNG (use a local torch.Generator, or save/restore torch.get_rng_state() around random_init_baseline), then reseed to cfg.seed immediately before training so the training stream is a deterministic function of cfg.seed alone.
- **Fix at:** `code/m7_trimodal_run_provenance.py:804-823`, `code/m7_trimodal_run_provenance.py:979-985`

#### `PROV-012` [MED] Default seeds list is five seeds, deviating from the documented 3-seed policy [42,123,456] and from the orphan
`code/m7_trimodal_run_provenance.py:63` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** ExperimentConfig.seeds defaults to [42, 123, 456, 789, 101112] (63) and main() sets the same (1320), so the default run is 5 seeds x 2 conditions = 10 runs. The documented policy (CLAUDE.md, python-numerics rule 4, load_results.py DOCUMENTED_SEEDS) is [42, 123, 456], and the orphan uses SEEDS=[42,123,456] (m7_trimodal_run.py:516). More seeds is not itself wrong, but the undocumented deviation means the hub and the orphan produce different-n results, the canonical loader's seed-policy gate is calibrated to three seeds, and any 'n-seed' label must be checked against which pipeline ran.
- **Evidence:** seeds default [42,123,456,789,101112] (63); main base_cfg seeds=[42,123,456,789,101112] (1320); orphan SEEDS=[42,123,456] (m7_trimodal_run.py:516); DOCUMENTED_SEEDS={42,123,456} (load_results.py:49).
- **Fix:** Align the default seed list with the documented policy, or document and justify the 5-seed choice in CLAUDE.md/EXPERIMENTAL_MANIFEST and update the canonical loader's expected-seeds gate accordingly.
- **Fix at:** `code/m7_trimodal_run_provenance.py:63`, `code/m7_trimodal_run_provenance.py:1320`
- _related: PROV-007_

#### `PROV-013` [MED] The 'per-seed' CSV actually stores the entire accumulated master table, not that seed's row
`code/m7_trimodal_run_provenance.py:858-860` · correctness · status:live · effort:trivial · confidence:high
- **What:** save_seed_result reassigns df to the full master (existing rows + new row) at 855, writes the master (856), then writes df again to cl_results_seed{seed}_{condition}.csv (859-860). Because df is the concatenated master, the file labelled per-seed contains EVERY seed/condition accumulated so far, not one seed. The comment 'Also save per-seed CSV for clean aggregation' (858) is therefore false, and downstream code expecting one row per file gets a growing full history mislabeled by whichever seed happened to be written last.
- **Evidence:** df = pd.concat([existing, df], ignore_index=True) (855); df.to_csv(csv_path) master (856); seed_csv = ... (859); df.to_csv(seed_csv) writes the SAME master df (860).
- **Fix:** Write only the current seed's row to the per-seed file: build a fresh single-row frame (e.g. pd.DataFrame([result])) for seed_csv instead of reusing the concatenated df.
- **Fix at:** `code/m7_trimodal_run_provenance.py:849-860`
- _related: PROV-006_

#### `PROV-014` [MED] SupCon losses count the anchor itself as a positive (diagonal not masked); the CrossModal comment claiming self contributes zero is wrong
`code/m7_trimodal_run_provenance.py:156-203` · correctness · status:live · effort:small · confidence:medium
- **What:** SupervisedContrastiveLoss.forward builds mask = eq(labels_i, labels_j) which is True on the diagonal (159), and mean_log_prob = (mask*log_prob).sum(1)/(mask.sum(1)+eps) (164), so each anchor's own similarity is counted in both numerator and denominator of the positive average. Standard SupCon (Khosla et al. 2020) excludes the anchor from its positive set P(i); including it biases the loss toward the trivially-satisfied self term and dilutes the true-positive gradient. This loss is ACTIVE in the default run (scl_weight=0.1, use_scl=True, 608-609). CrossModalSupConLoss repeats the same self-as-positive construction (192,198,202) and its comment (201) asserts 'self gives 0 contribution since mask*log_prob on diagonal is log(1)=0' — that is incorrect: the diagonal log_prob is log-softmax of the self position (<=0), not 0. The CrossModal loss only runs under align_cross_modal (default off), but the misleading comment and the shared bug pattern should be fixed.
- **Evidence:** SCL: mask includes diagonal (159); mean over positives incl. self (164); used in train_task (585,609). XMSCL: pos_mask=mask (198), denom incl. self (200), wrong comment (201).
- **Fix:** Mask the diagonal out of the positive set (set self-similarity to -inf before softmax and zero it in pos_mask/pos-count) to match Khosla SupCon; correct or delete the false comment at 201. Numerics-affecting: flag before/after if applied.
- **Fix at:** `code/m7_trimodal_run_provenance.py:156-165`, `code/m7_trimodal_run_provenance.py:181-203`, `code/m7_trimodal_run.py:186-199`

#### `PROV-015` [MED] Many ExperimentConfig fields are never consumed; some advertise values the run silently ignores (max_epochs, patience, dropout)
`code/m7_trimodal_run_provenance.py:33-74` · dead-code · status:live · effort:medium · confidence:high
- **What:** Grep confirms zero cfg.-qualified reads for optimizer (37), max_epochs (41), patience (42), dropout (45), hidden_dim (46), num_gru_layers (50), num_workers (51), pin_memory (53), time_steps (52); and align_joint_training (66), align_mixed_batches (67), nmnist_train_per_class (74) are referenced nowhere. Several are actively misleading: max_epochs=50 and patience=12 are ignored because train_task is called with num_epochs=15, patience=5 hardcoded (999,1048,1100); dropout=0.2 is advertised but the model contains no dropout layer; num_workers is hardcoded to 0 in every DataLoader; time_steps=25 is hardcoded in the collate fns (see PROV-018); optimizer='AdamW' is hardcoded (580). A reader trusting the config for reproducibility gets the wrong epoch/patience/dropout.
- **Evidence:** cfg.max_epochs/patience/dropout/hidden_dim/num_gru_layers/num_workers/pin_memory/time_steps/optimizer: 0 cfg-qualified references (grep); train_task called num_epochs=15 patience=5 (999,1048,1100); AdamW hardcoded (580); num_workers=0 literal (e.g. 1187-1193).
- **Fix:** Either wire these fields into the code (pass cfg.max_epochs/cfg.patience/cfg.num_workers/cfg.time_steps/cfg.optimizer through, add the dropout layer) or delete them from ExperimentConfig so the config cannot misrepresent the run.
- **Fix at:** `code/m7_trimodal_run_provenance.py:33-74`, `code/m7_trimodal_run_provenance.py:999`, `code/m7_trimodal_run_provenance.py:1048`, `code/m7_trimodal_run_provenance.py:1100`
- _related: PROV-018_

#### `PROV-016` [MED] build_data_loaders unconditionally imports and builds the paired cross-modal loader from the legacy root package, coupling the hub to Dataloaders/ even when alignment is off
`code/m7_trimodal_run_provenance.py:1251-1271` · dependencies · status:live · effort:small · confidence:high
- **What:** build_data_loaders does 'from Dataloaders.paired_digit_loader import get_paired_digit_loaders' (1251) and then constructs paired_loader (1262-1271) UNCONDITIONALLY, even though the paired loader is only consumed when cfg.align_cross_modal is True (1017). The hub has no sys.path insertion, so this import requires each of the 19 importing callers to have already put the repo root on sys.path; if not, build_data_loaders raises ModuleNotFoundError. It also does real work (materialising paired samples) in the common case where it is never used. This couples the shared hub to the legacy bi-modal package that CLAUDE.md marks as separate from M7.
- **Evidence:** from Dataloaders.paired_digit_loader import get_paired_digit_loaders (1251); paired_loader built unconditionally (1262-1271); only used under cfg.align_cross_modal (1017); no sys.path insert in this file (grep).
- **Fix:** Build the paired loader only when cfg.align_cross_modal, and move/guard the import behind that flag. If cross-modal alignment is a first-class hub feature, vendor the needed loader into code/ so the hub does not depend on the legacy root package or on caller sys.path setup.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1250-1271`, `code/m7_trimodal_run_provenance.py:1017-1027`

#### `PROV-017` [MED] Entire SHD train and test sets are densified into Python lists in memory before any training
`code/m7_trimodal_run_provenance.py:1212-1213` · performance · status:live · effort:medium · confidence:medium
- **What:** build_data_loaders materialises every SHD sample into a dense (100,1,1,700) float tensor via list comprehensions over the full datasets (1212-1213) before constructing loaders. Each dense sample is 100*700=70000 floats (~0.27 MB); across the full SHD train+test this is on the order of gigabytes held resident, which is wasteful on MPS/limited-RAM machines and duplicates the same eager pattern as the orphan (m7_trimodal_run.py:149-150). It also runs at build time for every invocation regardless of whether SHD is the current task.
- **Evidence:** shd_train_data = [_events_to_dense(e,l) for e,l in tqdm(shd_train,...)] (1212); shd_test_data likewise (1213); dense zeros(100,700) per sample (1201-1202).
- **Fix:** Wrap SHD in a lazy Dataset that densifies per-item in __getitem__ (or a DiskCachedDataset like N-MNIST) instead of building full in-memory lists, so memory scales with batch size, not dataset size.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1198-1227`

#### `PROV-018` [MED] Timestep count is inconsistent: cfg.time_steps=25 is dead, visual/DVS are cropped to 25 frames, but SHD runs at 100 timesteps
`code/m7_trimodal_run_provenance.py:121-123` · numerics · status:live · effort:medium · confidence:high
- **What:** cfg.time_steps=25 (52) is never read (PROV-015). The visual and DVS collate fns hard-crop/pad to 25 frames (113-116, 133-140), but SHD is densified to 100 time bins (_events_to_dense time_bins=100, 1201) and shd_collate_fn does no crop/pad (121-123), so SHD tensors reach the model with T=100. Since every backbone loops over T=x.size(1) (275, 353), N-MNIST/DVS run 25 steps while SHD runs 100 — a 4x difference in temporal depth. This makes cfg.time_steps=25 doubly misleading and complicates the mandatory 'accuracy travels with timestep count' reporting, because the per-task timestep cost is not the single documented value.
- **Evidence:** shd_collate_fn stacks without crop (121-123); _events_to_dense time_bins=100 (1201); nmnist/dvs cropped to 25 (113-116,133-140); model loops T=x.size(1) (275,353); cfg.time_steps unread (grep).
- **Fix:** Make the timestep count explicit and consistent: either drive all collate/densify paths from cfg.time_steps, or document per-modality timestep counts and report accuracy alongside the actual per-task T (25 visual / 100 audio). Remove or wire cfg.time_steps.
- **Fix at:** `code/m7_trimodal_run_provenance.py:121-123`, `code/m7_trimodal_run_provenance.py:1200-1210`, `code/m7_trimodal_run_provenance.py:52`
- _related: PROV-015_

#### `PROV-022` [MED] Output/data paths are CWD-relative ('./ICONS_M7'), so a hub used by 19 callers writes wherever each caller's CWD happens to be
`code/m7_trimodal_run_provenance.py:1285-1290` · quality · status:live · effort:small · confidence:high
- **What:** main() sets base_dir=Path('./ICONS_M7') (1285) and build_data_loaders uses Path('./ICONS_M7/datasets') (1169); both are relative to the current working directory, resolving as documented only when CWD is code/ (CLAUDE.md, python-numerics rule 6). Because this module is imported by 19 files that may run from different directories, results, checkpoints, and datasets can silently land in, or be read from, the wrong place — including the checkpoint-skip trap loading a stale final_*.pt from an unexpected directory.
- **Evidence:** base_dir = Path('./ICONS_M7') (1285); data_dir = Path('./ICONS_M7/datasets') (1169); checkpoints under results_dir (1287).
- **Fix:** Anchor paths to a stable base (e.g. Path(__file__).resolve().parent or an explicit argument/env var) rather than CWD, or require callers to pass an absolute results/ckpt dir (run_one_seed already accepts results_dir/ckpt_dir — main and build_data_loaders should too).
- **Fix at:** `code/m7_trimodal_run_provenance.py:1168-1170`, `code/m7_trimodal_run_provenance.py:1284-1290`

#### `SNN-011` [MED] Provenance runs 5 seeds (incl. 101112), not the policy [42,123,456]; three inconsistent seed sets across scripts
`code/m7_trimodal_run_provenance.py:63` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** provenance seeds=[42,123,456,789,101112] (L63, used L1320) vs m7 SEEDS=[42,123,456] (L516). Joint scripts use ...,789,1011 (run_joint_magnitude_ablation L62; run_joint_shared_encoder L85). The two 'M7' entrypoints aggregate over different n (5 vs 3) so their mean+/-std are not comparable, and provenance violates the documented seed policy the loader enforces (load_results.py L156-163). 101112 vs 1011 looks like a typo that fragments cross-script comparability.
- **Evidence:** provenance L63 seeds list; m7 L516
- **Fix:** Set provenance seeds=[42,123,456]; reconcile the joint scripts' fifth seed with the intended value.
- **Fix at:** `code/m7_trimodal_run_provenance.py:63`, `code/m7_trimodal_run_provenance.py:1320`
- _related: SNN-013_

#### `DEAD-014` [MED] modify_notebook.py sets a cell's source to a list of single characters (list(str))
`code/modify_notebook.py:558` · correctness · status:dead · effort:trivial · confidence:high
- **What:** src_text is a STRING (built at L207 as ''.join(src) and mutated via .replace()). At L558 the modified source is stored as cells[cl_cell_idx]['source'] = list(src_text), which splits the string into a list of individual CHARACTERS, not lines. The resulting notebook cell 'source' would be thousands of one-character strings - malformed for nbformat and unusable. Unreachable today because L9 crashes first (DEAD-012), but a genuine logic defect unique to this file.
- **Evidence:** L207 src_text = ''.join(src); L558 cells[cl_cell_idx]['source'] = list(src_text).
- **Fix:** Use src_text.splitlines(keepends=True) (or store the string directly) instead of list(...). Moot if the file is deleted.
- **Fix at:** `code/modify_notebook.py:558`
- _related: DEAD-012, DEAD-013_

#### `DEAD-015` [MED] monitor_and_stop.py is a dead one-off log-watcher tied to a machine that no longer exists
`code/monitor_and_stop.py:1-61` · dead-code · status:dead · effort:trivial · confidence:high
- **What:** A throwaway watcher that tailed one training log and killed one PID during an Era-1 run. It references a foreign absolute log path (/Users/agent1/...) and a fixed PID that have no meaning in this repo, and there is no train.log here. No reusable value.
- **Evidence:** LOG_FILE=/Users/agent1/.../train.log L9; PID=16672 L12; regex parses 'Val: NN.N%' L25.
- **Fix:** Delete.
- **Fix at:** `code/monitor_and_stop.py:1-61`
- _related: DEAD-016_

#### `DEP-003` [MED] pytest>=9.0 floor may be unsatisfiable, and pytest is never imported under code/
`code/requirements.txt:13` · dependencies · status:live · effort:trivial · confidence:high
- **What:** pytest>=9.0 (L13). At the Jan 2026 cutoff pytest was on the 8.x series; a >=9.0 floor may not be satisfiable on PyPI, which would break the install. Separately, pytest is never imported anywhere under code/ (0 occurrences in a full grep of code/ incl. archive/); the actual pytest suites live in the repo-root tests/ directory, out of scope for code/requirements.txt. A runtime requirements file is the wrong home for a test-only tool.
- **Evidence:** code/requirements.txt:13 pytest>=9.0 ; grep for 'pytest' across code/**.py returns 0 matches
- **Fix:** Verify a pytest 9.x exists for py3.11; if not, relax to an existing floor. Consider moving pytest to a dev/test extra (e.g. requirements-dev.txt) rather than the runtime requirements.
- **Fix at:** `code/requirements.txt:13`

#### `DEP-010` [MED] Declared dependencies never imported anywhere under code/
`code/requirements.txt:6` · dependencies · status:live · effort:small · confidence:high
- **What:** Four pins have zero import occurrences in a full grep of code/ (incl. archive/). transformers==5.7.0 (L6): appears only in stale notebook cruft (m7_raw.py:22 `!pip install -q transformers`; m7_mps.py:27/m7_raw.py:27/etc print 'Newly installed: ... transformers, datasets') and is never imported -> safe to drop, and it is a heavy dependency that adds resolver risk for nothing. torchaudio==2.11.0 (L2) and torchvision==0.26.0 (L3): not imported directly anywhere in code/, but are commonly transitive deps of the torch/tonic stack -> verify before removing. h5py==3.16.0 (L11): not imported directly, but tonic reads SHD from HDF5 and likely needs h5py transitively -> verify before removing. Only transformers is clearly removable.
- **Evidence:** grep across code/**.py: torchaudio 0 hits, torchvision 0 hits, h5py 0 hits, transformers only in comments/prints (m7_raw.py:22,27 ; m7_mps.py:22,27 ; m7_smoke_test.py:22,27 ; m7_trimodal_cl.py:22,27)
- **Fix:** Remove transformers unless something outside code/ needs it. For torchaudio/torchvision/h5py, confirm whether they are pulled transitively (they most likely are) and keep them if so; otherwise drop. Do not blindly delete the three transitive candidates.
- **Fix at:** `code/requirements.txt:2`, `code/requirements.txt:3`, `code/requirements.txt:6`, `code/requirements.txt:11`
- _related: DEP-018_

#### `ABL-001` [MED] _save_csv opens in 'w' mode, silently discarding prior ablation rows on a narrower re-run
`code/run_ablation.py:461-470` · correctness · status:live · effort:small · confidence:medium
- **What:** run_ablation.py writes results with _save_csv opening the output CSV in write ('w') mode, so re-running with a narrower --datasets/--models/--seeds selection overwrites and discards all previously computed ablation rows instead of merging. This is the inverse of the append-without-dedup hazard elsewhere: here legitimate prior results are silently lost. (Added by the completeness-critic pass. run_ablation.py is otherwise the one script that seeds correctly with [42,123,456], measures real spike metrics via SpikeRecorder/measure_spike_metrics, and has a __main__ guard - it wires to the legacy bi-modal package, not M7.)
- **Evidence:** _save_csv opens the CSV in 'w' (overwrite) mode at ~461-470; no merge with existing rows keyed by (dataset,model,seed)
- **Fix:** Merge new rows with any existing CSV keyed by (dataset,model,seed) before writing (upsert), or append then drop_duplicates on that key; do not blindly overwrite.
- **Fix at:** `code/run_ablation.py:461-470`

#### `DEP-011` [MED] Legacy repo-root package imports resolve only via runtime sys.path hacks plus correct CWD
`code/run_ablation.py:35-45` · dependencies · status:live · effort:medium · confidence:high
- **What:** code/ has no __init__.py, so scripts reach the repo-root legacy package (Models/, Dataloaders/, training/, utils/) by doing sys.path.insert(0, REPO_ROOT) at runtime before importing. All referenced modules exist (Models/registry.py, training/pipeline.py, training/settings.py, Dataloaders/*_loader.py, utils/cross_modal_transfer.py), so these resolve WHEN the insert runs, but the pattern is fragile: imports must physically follow the insert (hence the noqa: E402 markers scattered through the run_* scripts), and any tool that imports the module without first running the insert (linters, IDEs, test collectors, a different CWD) fails to resolve the legacy package.
- **Evidence:** Importers of legacy packages: code/run_ablation.py:35-45 (Models.registry, training.pipeline, training.settings, Dataloaders.*) ; utils.cross_modal_transfer at code/eval_transfer_fixed.py:29, code/run_cl_rehearsal.py:44, code/run_correspondence_anchor_v2.py:45, code/run_cross_modal_alignment_study.py:61, code/run_digit_only_alignment.py:64, code/run_ordinal_anchor.py:48, code/run_rehearsal_transfer_eval_only.py:19 ; sys.path.insert(0, str(REPO_ROOT)) at code/run_ablation.py and ~15 other code/ files
- **Fix:** Make the legacy package importable as a real package (pip install -e a package at repo root, or a conftest.py/path shim, or convert code/ into a package) so imports do not depend on runtime sys.path mutation and CWD.
- **Fix at:** `code/run_ablation.py:35-45`, `code/eval_transfer_fixed.py:29`, `code/run_cl_rehearsal.py:44`, `code/run_correspondence_anchor_v2.py:45`, `code/run_cross_modal_alignment_study.py:61`, `code/run_digit_only_alignment.py:64`, `code/run_ordinal_anchor.py:48`, `code/run_rehearsal_transfer_eval_only.py:19`
- _related: DEP-012, DEP-013, DEP-015_

#### `CLE-020` [MED] Saved config misrepresents actual training length (cfg 50/12 vs hardcoded 15/5)
`code/run_cl_confirmation_fixed.py:224-231` · research-integrity · status:live · effort:medium · confidence:high
- **What:** cfg sets max_epochs=50, patience=12 (368) but every train_task call hardcodes num_epochs=15, patience=5 (226,266,302), and align uses num_epochs=5 (241). The config saved next to each checkpoint via cfg.to_json (231,271,307) and results/master_config.json (402) therefore records 50/12, not the 15/5 the run truly used. Anyone reproducing from the saved config would train 50 epochs and not reproduce the checkpoints -- a provenance/reproducibility defect. Identical pattern in run_cl_rehearsal.py (201/249/291 vs 381).
- **Evidence:** cfg max_epochs=50, patience=12 at line 368; train_task hardcodes num_epochs=15, patience=5 at 226/266/302; cfg.to_json at 231/271/307/402.
- **Fix:** Drive num_epochs/patience from cfg (add per-phase fields if 15 vs 50 is intentional) so the persisted config equals what actually ran. Apply the same fix in run_cl_rehearsal.py.
- **Fix at:** `code/run_cl_confirmation_fixed.py:224-231`, `code/run_cl_confirmation_fixed.py:264-271`, `code/run_cl_confirmation_fixed.py:300-307`, `code/run_cl_rehearsal.py:199-207`, `code/run_cl_rehearsal.py:247-257`, `code/run_cl_rehearsal.py:289-299`
- _related: CLE-019_

#### `CLE-021` [MED] forgetting_T1_T3 is the incremental post-T2->post-T3 drop, not total T1 forgetting the name implies
`code/run_cl_confirmation_fixed.py:327` · correctness · status:live · effort:small · confidence:high
- **What:** forgetting_T1_T3 = acc_t1_after_t2 - acc_t1_after_t3 (327) measures only the T1 drop that occurs during T3 (between post-T2 and post-T3). The name reads as total T1->T3 forgetting, which is actually a separate column total_forgetting_T1_T3 = acc_t1_after - acc_t1_after_t3 (329). This is a real mislabel risk: a summary or paper could quote 'T1_T3 forgetting' and use the incremental value. Identical construction in run_cl_rehearsal.py:336.
- **Evidence:** Line 327 forgetting_t1_t3 = acc_t1_after_t2 - acc_t1_after_t3; line 329 total_forgetting_t1_t3 = acc_t1_after - acc_t1_after_t3.
- **Fix:** Rename to forgetting_T1_during_T3 (incremental) or make the pair unambiguous in a data dictionary; ensure downstream analysis and paper text pick the intended one. Mirror in run_cl_rehearsal.py.
- **Fix at:** `code/run_cl_confirmation_fixed.py:326-329`, `code/run_cl_rehearsal.py:335-338`
- _related: CLE-022_

#### `CLE-024` [MED] Results CSV stores accuracy/forgetting with no spike-rate/sparsity cost metric
`code/run_cl_confirmation_fixed.py:332-347` · research-integrity · status:live · effort:medium · confidence:medium
- **What:** The persisted result dict (332-347) carries T1/T2/T3 test accuracy, FWT and forgetting but no spike rate or sparsity; only the timestep count survives, and only indirectly via the separately saved config (time_steps=25). Per the 'accuracy never travels alone' rule, an SNN accuracy in a results CSV that downstream figures/analysis read must ship with its spike-rate/sparsity and timestep cost. Same for run_cl_rehearsal.py:341-356.
- **Evidence:** result dict keys at 332-347 contain no spike/sparsity field; time_steps only in cfg.to_json output.
- **Fix:** Add measured mean spike rate / sparsity (and time_steps) columns to the result row, or document that cost metrics live in a companion file. Mirror in run_cl_rehearsal.py.
- **Fix at:** `code/run_cl_confirmation_fixed.py:332-347`, `code/run_cl_rehearsal.py:341-356`
- _related: CLE-016_

#### `CLE-029` [MED] Per-task checkpoint-skip silently loads stale weights instead of retraining
`code/run_cl_confirmation_fixed.py:220-222` · safety · status:live · effort:medium · confidence:high
- **What:** For each task, `if ckpt.exists(): print skip; model.load_state_dict(torch.load(ckpt))` (220-222, 260-262, 296-298) skips training whenever a matching final_{task}_{task}.pt is present. This is the documented repo trap (training silently skipped on existing checkpoint) replicated per task: a stale checkpoint from older code yields stale results that look freshly computed, with only a transient stdout note. Same in run_cl_rehearsal.py (195-197, 243-245, 285-287).
- **Evidence:** Lines 220-222, 260-262, 296-298 (confirmation); 195-197, 243-245, 285-287 (rehearsal).
- **Fix:** Record and check a code/version/config hash against the checkpoint before skipping, or require an explicit --resume flag; at minimum warn loudly (not just print) and log the checkpoint mtime.
- **Fix at:** `code/run_cl_confirmation_fixed.py:220-222`, `code/run_cl_confirmation_fixed.py:260-262`, `code/run_cl_confirmation_fixed.py:296-298`, `code/run_cl_rehearsal.py:195-197`, `code/run_cl_rehearsal.py:243-245`, `code/run_cl_rehearsal.py:285-287`
- _related: CLE-018_

#### `CLE-030` [MED] build_full_loaders and SHD/N-MNIST loader helpers are duplicated across the cluster
`code/run_cl_confirmation_fixed.py:69-179` · duplication · status:live · effort:large · confidence:high
- **What:** build_full_loaders (69-179) is nearly identical between run_cl_confirmation_fixed.py and run_cl_rehearsal.py (47-157). The _events_to_dense SHD-densify helper is copy-pasted into 4 files (run_cl_confirmation_fixed 101-111, run_cl_rehearsal 79-89, eval_transfer_fixed 46-56, run_rehearsal_transfer_eval_only 65-75), and build_nmnist_test_loader/build_shd_test_loader are duplicated between eval_transfer_fixed and phase1_sanity_check. Divergent edits to any copy (transform windows, split seed, densify logic) silently produce mismatched preprocessing across scripts that are supposed to be comparable.
- **Evidence:** build_full_loaders present in two files with matching bodies; _events_to_dense present in 4 files; build_*_test_loader present in eval_transfer_fixed.py:33-58 and phase1_sanity_check.py:34-42.
- **Fix:** Extract the loaders and _events_to_dense into a shared, import-safe module (alongside m7_trimodal_run_provenance) and import them; do not re-declare per script.
- **Fix at:** `code/run_cl_confirmation_fixed.py:69-179`, `code/run_cl_rehearsal.py:47-157`, `code/eval_transfer_fixed.py:33-58`, `code/run_rehearsal_transfer_eval_only.py:49-77`, `code/phase1_sanity_check.py:34-42`
- _related: CLE-031, CLE-032_

#### `CMP-002` [MED] held-30 whole-number chance level mislabeled (actual 1%, documented 3.33%)
`code/run_compositional_crossmodal.py:40` · research-integrity · status:live · effort:small · confidence:high
- **What:** The docstring states 'chance 1.0% all-100 / 3.33% held-30', but run_one_model builds number prototypes over ALL present numbers (Pa/ua = number_protos over train+held) and retrieves held-mask queries against that full 100-number pool. Chance for held-30 is therefore 1/100 = 1%, not 1/30 = 3.33%. The verdict threshold wn_held>2.0 is interpreted against the wrong baseline, and the empirical perm null (~1%) contradicts the documented 3.33%.
- **Evidence:** Pa, ua = number_protos(aZ, a_lab)  # all present numbers; r = whole_number_retrieval(vZ[mask], v_lab[mask], Pa, ua, ...)
- **Fix:** Either restrict the held-30 retrieval pool to the 30 held prototypes (then 3.33% is correct) or fix the docstring/threshold to 1% chance; make the retrieval pool explicit in code and report.
- **Fix at:** `code/run_compositional_crossmodal.py:40`, `code/run_compositional_crossmodal.py:895-916`, `code/run_compositional_crossmodal.py:1017`

#### `CMP-003` [MED] Visual 'spike_rate'/'sparsity' is a post-conv2+ReLU activation rate, not a spike rate
`code/run_compositional_crossmodal.py:794-809` · research-integrity · status:live · effort:small · confidence:high
- **What:** _report_spiking_stats treats the stem output as summed spikes, but VisualMapStem.forward returns F.relu(conv2(time-summed LIF spikes)) - conv2 is a non-spiking real-valued refiner, so its tokens are ReLU activations, not spikes. counts=tokens.sum(-1)/T thus reports an activation magnitude mislabeled as spike_rate/sparsity for the visual modality. AudioMapStem returns genuine LIF spikes (s3), so the audio channel is correct; only visual is mislabeled. CLAUDE.md requires accuracy to travel with a genuine spike-rate/sparsity + timestep cost, so a mislabeled cost metric violates the integrity rule.
- **Evidence:** VisualMapStem.forward: m = F.relu(self.conv2(acc)); return m.flatten(2).transpose(1,2)  -> not spikes
- **Fix:** Measure the visual spike rate at the LIF output (s1 inside VisualMapStem) rather than after conv2/ReLU, or rename the visual column to conv2_activation_rate and stop reporting it as spike_rate/sparsity.
- **Fix at:** `code/run_compositional_crossmodal.py:794-809`, `code/run_compositional_crossmodal.py:304-315`

#### `CMP-004` [MED] Per-digit horizontal jitter is inert for the right digit and boundary-violating for the left
`code/run_compositional_crossmodal.py:206-225` · correctness · status:live · effort:small · confidence:medium
- **What:** Per-digit jitter is far weaker than the boundary comment implies. The RIGHT digit is fully inert: with RIGHT_X0=40 and the min/max clamp, xr==40 for every jitter in [-3,3]. The LEFT digit does jitter but only over xl in {0,1,2,3} (LEFT_X0=0, JITTER=3), spanning cols [3,37); it clips ~3px into the 6px inter-digit buffer (cols 34-39) but never reaches the right region (col>=40). The code's boundary comment is technically violated, but by ~3px, not by a large cross-region slide.
- **Evidence:** RIGHT_X0=40 + clamp => xr==40 for all jitter; LEFT_X0=0, JITTER=3 => xl in {0..3}, span [3,37), max 3px into the 6px buffer
- **Fix:** Make the augmentation actually jitter both digits within intended bounds (the right digit is pinned; the left moves <=3px); if a buffer-crossing near-miss is intended, widen the left range deliberately and assert it.
- **Fix at:** `code/run_compositional_crossmodal.py:206-215`, `code/run_compositional_crossmodal.py:217-225`

#### `CMP-005` [MED] 'shuffled' negative control is defined inconsistently across phases
`code/run_compositional_crossmodal.py:511-556` · correctness · status:live · effort:small · confidence:medium
- **What:** Phase (a) applies ONE permutation to a single label vector reused for both modalities (lab -> cat([lab,lab,lab,lab])), preserving cross-modal pairing while breaking the label-to-data correspondence. Phase (b) shuffles the AUDIO labels with an INDEPENDENT permutation (lab_a) while visual keeps lab, which breaks the cross-modal pairing itself. The negative control therefore means two different things across the curriculum, muddying its interpretation as a baseline.
- **Evidence:** phase a: if shuffle_pairs: lab = lab[torch.randperm(...)]  vs  phase b: lab_a = lab[torch.randperm(...)]
- **Fix:** Choose a single shuffle semantics (e.g. permute the cross-modal correspondence, or permute labels-vs-data) and apply it identically in both phases.
- **Fix at:** `code/run_compositional_crossmodal.py:511-519`, `code/run_compositional_crossmodal.py:547-556`

#### `CMP-006` [MED] Aggregation double-counts a seed when a timestamped rerun CSV coexists with its base
`code/run_compositional_crossmodal.py:953-997` · research-integrity · status:live · effort:small · confidence:medium
- **What:** load_rows_from_disk globs model_*_seed*.csv, which matches both model_X_seedN.csv and the model_X_seedN__<ts>.csv variant that _model_csv_path writes when the base already exists. write_summary_and_report means over ALL rows with no collapse to unique (model,seed), so a rerun inflates n_seeds and corrupts mean/std - the same near-duplicate inflation documented for generate_figures_3seed in CLAUDE.md.
- **Evidence:** for p in sorted(RESULTS_DIR.glob('model_*_seed*.csv')): ... ; vals = [r[key] for r in sub ...]  # no unique-seed collapse
- **Fix:** Collapse to one row per (model, seed) (keep the latest timestamp) before aggregating, and warn when duplicates are found.
- **Fix at:** `code/run_compositional_crossmodal.py:953-966`, `code/run_compositional_crossmodal.py:984-997`, `code/run_compositional_crossmodal.py:817-822`
- _related: CMP-022_

#### `CMP-007` [MED] assert_split_coverage can fail for arbitrary seeds and is then silently swallowed per-seed
`code/run_compositional_crossmodal.py:265-281` · correctness · status:live · effort:small · confidence:medium
- **What:** assert_split_coverage requires all ten digits to appear in pos1 AND pos2 within BOTH the 70-number train set and the 30-number held set. A random 30/70 permutation split does not guarantee full per-position coverage for every seed. On failure the assert raises, and main's broad try/except prints a traceback and drops that seed, so a nominal 3-seed run can silently become fewer seeds.
- **Evidence:** assert p1 == set(range(10)), f'{name} missing pos1 digits: ...'  (raised inside run_one_model, caught by main)
- **Fix:** Resample the split in make_split until coverage holds (loop), or restrict to seeds verified to satisfy coverage; do not rely on the broad except to hide coverage failures.
- **Fix at:** `code/run_compositional_crossmodal.py:265-281`, `code/run_compositional_crossmodal.py:256-262`, `code/run_compositional_crossmodal.py:1195-1201`

#### `CMP-008` [MED] Module-level pickle-alias hack mutates sys.modules['__main__'] at import time
`code/run_compositional_crossmodal.py:87-95` · safety · status:live · effort:medium · confidence:high
- **What:** At import, the module injects CachedNMNIST/CachedSHD/SubsampledByClass into sys.modules['__main__'] so torch.load(weights_only=False) of caches pickled under __main__ resolves. This is an import-time side effect (namespace pollution of whatever program imports this file) duplicated verbatim from run_joint_diagnostic.py:67-74. It only helps if the cache was pickled while run_joint_magnitude_ablation ran as __main__; no cache exists in-repo, so it currently guards a state that cannot occur until the magnitude script is first run as main.
- **Evidence:** for _cn in ('CachedNMNIST','CachedSHD','SubsampledByClass'): setattr(sys.modules['__main__'], _cn, getattr(_mag_module,_cn))
- **Fix:** Rebuild caches so the classes resolve to the importable run_joint_magnitude_ablation module, or load via a custom Unpickler that maps __main__.<cls> -> the module class; remove the __main__ mutation and factor the shared logic into one importable helper.
- **Fix at:** `code/run_compositional_crossmodal.py:87-95`, `code/run_joint_diagnostic.py:67-74`
- _related: CMP-020_

#### `CMP-009` [MED] Running with no cache silently triggers a ~23 GB tonic render/download
`code/run_compositional_crossmodal.py:1177-1186` · safety · status:live · effort:small · confidence:high
- **What:** _build_modality_loaders renders N-MNIST and SHD from tonic when the cache is absent; no results/ cache exists in this repo. The module comment claims it 'loads the existing 4.5 GB cache without a re-render', which is false in current repo state. The __main__ guard means import is safe, but an unsuspecting explicit run starts the full multi-hour render/download rather than a quick diagnostic.
- **Evidence:** loaders, cached = _build_modality_loaders(cfg)  # renders from tonic when cache missing (magnitude:174-198)
- **Fix:** Fail fast with a clear message if the cache directory is missing rather than silently rendering, and document the dataset prerequisite in the docstring.
- **Fix at:** `code/run_compositional_crossmodal.py:1177-1186`, `code/run_compositional_crossmodal.py:87-91`
- _related: CMP-021_

#### `CMP-013` [MED] File is a structural clone of run_joint_diagnostic.py and the magnitude harness
`code/run_compositional_crossmodal.py:1` · architecture · status:live · effort:large · confidence:high
- **What:** set_seed, _build_cfg, _mean/_std, _binom_z, load_rows_from_disk, write_summary_and_report, _report_spiking_stats, and the main/aggregate scaffolding are near-duplicates of run_joint_diagnostic.py and run_joint_magnitude_ablation.py. The two audited files together are ~95 KB of largely copy-pasted experiment harness, so a fix to one (e.g. the dedup or the spike-metric bug) must be applied in several places.
- **Evidence:** set_seed(146), _build_cfg(153), _mean/_std(969), _binom_z(979), load_rows_from_disk(953), write_summary_and_report(984) duplicate the sibling
- **Fix:** Extract a shared, importable, __main__-guarded harness module (seed, cfg, aggregation, spiking-stats) and import it from both experiment scripts.
- **Fix at:** `code/run_compositional_crossmodal.py:146-165`, `code/run_compositional_crossmodal.py:969-997`, `code/run_compositional_crossmodal.py:795-809`, `code/run_joint_diagnostic.py:130-135`, `code/run_joint_diagnostic.py:441-505`
- _related: CMP-024_

#### `ALGN-006` [MED] _extract_backbone_features defined but never called (dead clone of archived v1)
`code/run_correspondence_anchor_v2.py:120-144` · dead-code · status:live · effort:trivial · confidence:high
- **What:** _extract_backbone_features is a module-level function that is never invoked anywhere in the file (feature extraction goes through _load_cached_features and the model's own _get_backbone_features method). It is a verbatim clone of code/archive/run_correspondence_anchor.py:139. Dead code that also re-implements the audio backbone forward (drift risk, see ALGN-007).
- **Evidence:** grep shows the only occurrence of _extract_backbone_features in the file is its def at line 120; clone at code/archive/run_correspondence_anchor.py:139
- **Fix:** Delete the dead function; if a cache-builder is needed, expose one live helper and call the model's feature method.
- **Fix at:** `code/run_correspondence_anchor_v2.py:120-144`
- _related: ALGN-003, ALGN-007, ALGN-008_

#### `ALGN-007` [MED] Audio backbone forward hand-reimplemented, duplicating the model (silent-drift risk)
`code/run_correspondence_anchor_v2.py:190-207` · duplication · status:live · effort:small · confidence:high
- **What:** CorrespondenceModelV2._get_backbone_features (and the dead _extract_backbone_features) manually re-run the SHD audio ModuleDict forward (bb['lif1'](bb['fc1'](xt)) ... fc3), duplicating M7_ContinualAdaptiveModel._get_features/forward_with_shared in m7_trimodal_run_provenance.py:343-399. It is numerically consistent with the model TODAY (verified: audio ModuleDict is fc1->1024,fc2->1024,fc3->512 with lif1-3), but if the provenance backbone changes, this copy silently produces wrong features with no error.
- **Evidence:** lines 190-207 mem1=bb['lif1'].init_leaky(); s3=bb['lif3'](bb['fc3'](s2)); provenance _get_features m7_trimodal_run_provenance.py:343-360 is identical logic
- **Fix:** Call the base model's feature extractor (e.g. a public _get_features) instead of re-deriving the forward pass in each script.
- **Fix at:** `code/run_correspondence_anchor_v2.py:190-207`, `code/run_correspondence_anchor_v2.py:120-144`, `code/run_ordinal_anchor.py:282-299`
- _related: ALGN-006, ALGN-018_

#### `ALGN-008` [MED] Cluster-wide helper duplication: no shared alignment library
`code/run_correspondence_anchor_v2.py:1` · architecture · status:live · effort:large · confidence:high
- **What:** Core alignment helpers are copy-pasted across the run_* fleet instead of centralized. compute_held_out_metrics has 3 definitions (run_correspondence_anchor_v2.py:372, run_joint_magnitude_ablation.py:572, run_joint_shared_encoder.py:653 [different signature]); _hash_params has 2 (run_correspondence_anchor_v2.py:59, code/archive/run_correspondence_anchor.py:65); _extract_backbone_features 2 (v2:120, archive:139); build_paired_feature_dataset and train_correspondence in both v2 and archive; run_procrustes in v2:450 and run_joint_shared_encoder.py:704 plus run_procrustes_z in run_joint_magnitude_ablation.py:648; _events_to_dense re-defined in ~8 files. Divergent copies will silently disagree.
- **Evidence:** grep -rn 'def compute_held_out_metrics|def _hash_params|def run_procrustes|def _extract_backbone_features|def build_paired_feature_dataset|def _events_to_dense' code/ (see counts above)
- **Fix:** Extract shared alignment/metric helpers into one module (e.g. code/alignment_common.py or utils/) and import everywhere; delete archived duplicates from the live import surface.
- **Fix at:** `code/run_correspondence_anchor_v2.py:59-64`, `code/run_correspondence_anchor_v2.py:120-144`, `code/run_correspondence_anchor_v2.py:241-275`, `code/run_correspondence_anchor_v2.py:278-365`, `code/run_correspondence_anchor_v2.py:372-443`, `code/run_correspondence_anchor_v2.py:450-517`, `code/run_ordinal_anchor.py:92-102`, `code/run_correspondence_anchor_v2.py:93-103`, `code/run_digit_only_alignment.py:122-132`
- _related: ALGN-006, ALGN-007, ALGN-009, ALGN-018_

#### `ALGN-009` [MED] run_one_seed and compute_held_out_metrics collide with other definitions (3 signatures each)
`code/run_correspondence_anchor_v2.py:524-525` · naming · status:live · effort:small · confidence:high
- **What:** This file defines module-level run_one_seed(seed,cfg,loaders,cached,device,head_type) at line 524, while m7_trimodal_run_provenance.py:962 defines run_one_seed(cfg,data_loaders,results_dir,ckpt_dir) and run_ordinal_anchor.py:395 defines yet another run_one_seed. Likewise compute_held_out_metrics has three incompatible signatures across the fleet (ALGN-008). Any future 'from run_correspondence_anchor_v2 import run_one_seed' would silently bind the wrong function; run_digit_only_alignment already imports the provenance one. Same-name/different-contract functions are an error waiting to happen.
- **Evidence:** run_correspondence_anchor_v2.py:524 def run_one_seed(seed, cfg, loaders, cached, device, head_type); m7_trimodal_run_provenance.py:962 def run_one_seed(cfg, data_loaders, results_dir, ckpt_dir); run_ordinal_anchor.py:395 def run_one_seed(seed, cfg, loaders, cached, device, head_type)
- **Fix:** Rename the per-script runners (e.g. run_one_seed_correspondence) or namespace them; keep one canonical run_one_seed.
- **Fix at:** `code/run_correspondence_anchor_v2.py:524`, `code/run_ordinal_anchor.py:395`, `code/m7_trimodal_run_provenance.py:962`
- _related: ALGN-008_

#### `ALGN-022` [MED] master_results.csv accumulates duplicate rows on rerun; summary n/std/CI corrupted
`code/run_cross_modal_alignment_study.py:68-118` · research-integrity · status:live · effort:small · confidence:high
- **What:** _save_seed_result appends each row to master_results.csv (read existing, concat, rewrite; lines 72-75) with no dedup on (seed,condition). _aggregate_alignment_results computes n=len(sub), mean, std(ddof=1) and 95% CI from all rows (99-118), so reruns inflate n and shrink/skew the CI - the same class of bug called out for generate_figures_3seed. Asymmetric: the Wilcoxon pivot uses aggfunc='first' (line 131) so the significance test dedups, but the summary statistics do not.
- **Evidence:** lines 72-75 existing=pd.read_csv(master_csv); df=pd.concat([existing,df]); df.to_csv(master_csv); lines 107-109 n=len(sub); ci95=1.96*std/sqrt(n); line 131 pivot_table(..., aggfunc='first')
- **Fix:** drop_duplicates on (seed,condition) (keep last) before computing summary stats, mirroring the pivot's dedup.
- **Fix at:** `code/run_cross_modal_alignment_study.py:68-83`, `code/run_cross_modal_alignment_study.py:96-118`
- _related: ALGN-021_

#### `ALGN-023` [MED] Condition A silhouettes hardcoded to 0.0 and fed into the summary
`code/run_cross_modal_alignment_study.py:260-265` · research-integrity · status:live · effort:small · confidence:high
- **What:** For the no-alignment branch, transfer_after_align is fabricated: every key is copied from transfer_after_t1 except sil_visual/sil_audio, which are forced to 0.0 (260-265). result['ALIGN_sil_vis']/['ALIGN_sil_aud'] (388-389) are therefore 0.0 for every A seed, and ALIGN_sil_vis/ALIGN_sil_aud are in the aggregated metrics list (522), so condition A summarizes as 'after-alignment silhouette = 0.0 +/- 0.0'. That reads as a measured zero cluster-separation but is a placeholder (A has no alignment phase). The V2A/A2V/proto ALIGN_* values for A are meanwhile copies of the T1 values, not independent measurements.
- **Evidence:** lines 261-265 transfer_after_align = {k:(0.0 if k.startswith('sil_') else transfer_after_t1.get(k,0.0)) for k in [...]}; lines 388-389 ALIGN_sil_vis/aud; line 522 lists ALIGN_sil_vis, ALIGN_sil_aud in aggregation
- **Fix:** Either actually measure post-'alignment' silhouettes for A (they equal the T1 values) or exclude ALIGN_* for condition A from the summary and mark them N/A rather than 0.0.
- **Fix at:** `code/run_cross_modal_alignment_study.py:260-265`, `code/run_cross_modal_alignment_study.py:384-389`, `code/run_cross_modal_alignment_study.py:520-524`

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

#### `ALGN-030` [MED] Digit-only script couples to the legacy bi-modal Dataloaders package (boundary)
`code/run_digit_only_alignment.py:152` · architecture · status:live · effort:medium · confidence:medium
- **What:** build_digit_loaders does 'from Dataloaders.paired_digit_loader import get_paired_digit_loaders' (line 152). Dataloaders/ is the legacy bi-modal (Paper 2) package that CLAUDE.md and python-numerics.md place OUTSIDE the M7 reproducibility boundary (m7_trimodal_run.py must not import it). The import resolves (the module exists), but it couples an M7-era alignment study to legacy code, so a change there silently affects these results.
- **Evidence:** line 152 from Dataloaders.paired_digit_loader import get_paired_digit_loaders; Dataloaders/paired_digit_loader.py exists at repo root (legacy package)
- **Fix:** Vendor the paired-loader logic into the M7 code/ surface or into the shared provenance module so the alignment studies do not depend on the legacy package.
- **Fix at:** `code/run_digit_only_alignment.py:151-172`
- _related: ALGN-008_

#### `DEP-014` [MED] `import run_cl_confirmation` references a module that exists only in archive/
`code/run_digit_only_alignment.py:616-617` · dependencies · status:live · effort:trivial · confidence:high
- **What:** Under the --confirm CLI branch the script does `import run_cl_confirmation; run_cl_confirmation.main()`. There is no run_cl_confirmation.py in code/ (only code/run_cl_confirmation_fixed.py); the sole run_cl_confirmation.py lives in code/archive/, which is never added to sys.path (the script inserts only REPO_ROOT/'code' and REPO_ROOT). Running with --confirm therefore raises ImportError at call time. Almost certainly a stale reference left after the file was renamed/superseded by run_cl_confirmation_fixed.py.
- **Evidence:** code/run_digit_only_alignment.py:616 import run_cl_confirmation ; :617 run_cl_confirmation.main() . find shows run_cl_confirmation.py only under code/archive/ (code/ has run_cl_confirmation_fixed.py). sys.path inserts at code/run_digit_only_alignment.py:50-51
- **Fix:** Point the import at run_cl_confirmation_fixed (verify it exposes main()), or restore/relocate the intended module onto the import path.
- **Fix at:** `code/run_digit_only_alignment.py:616-617`
- _related: DEP-013_

#### `CMP-018` [MED] 'Above chance' verdict rests on a binomial z that treats correlated per-sample retrievals as independent
`code/run_joint_diagnostic.py:451-466` · research-integrity · status:live · effort:medium · confidence:medium
- **What:** _binom_z uses n = n_vis_heldout (all N-MNIST test 6-9 samples, on the order of thousands) as independent Bernoulli trials, and _above_chance thresholds z>6. But every query is scored against the same set of prototypes with the same argmax structure, so the trials are strongly correlated (pseudo-replication); the binomial SE is understated and z is inflated. With n in the thousands, an accuracy only slightly above 10% (roughly 13%) already clears z>6, so weak structure reads as 'clearly above chance'. This statistic directly drives the written verdicts about a classifier-leak mechanism and the absence of emergent cross-modal generalization.
- **Evidence:** se = (p0*(1-p0)/max(n,1))**0.5; return (p_hat-p0)/(se+1e-12);  _above_chance: n=int(r.get('n_vis_heldout',0)); z>z_min(6.0)
- **Fix:** Use a test that accounts for correlation across shared prototypes (per-class or per-prototype resampling, or the permutation null with a robust spread) instead of equating sample count with independent trials; at minimum report the effective n and stop treating n_vis_heldout as the trial count.
- **Fix at:** `code/run_joint_diagnostic.py:451-466`, `code/run_joint_diagnostic.py:544-582`

#### `CMP-019` [MED] legacy_untrained_v2a is computed via a full test-set forward but never analyzed; legacy_proto_a2v is written but never read
`code/run_joint_diagnostic.py:374` · research-integrity · status:live · effort:small · confidence:high
- **What:** run_one_diagnostic_cell computes legacy_untrained = compute_held_out_metrics(untrained, ...), a full two-modality embedding pass over the N-MNIST and SHD test sets, solely to populate the legacy_untrained_v2a column, which no summary or report consumes. legacy_proto_a2v is likewise written to every CSV row but never read. Only legacy_proto_v2a is actually used (the artifact-collapse cross-check). So part of the legacy_* surface is dead code plus a wasted expensive forward pass per cell.
- **Evidence:** legacy_untrained = compute_held_out_metrics(untrained, loaders, device=device)  -> stored at 416, never read; legacy_proto_a2v stored at 415, never read
- **Fix:** Drop legacy_untrained_v2a and its forward pass and legacy_proto_a2v (or actually consume them); add a comment that legacy_proto_v2a is the degenerate 4-class metric kept only as an artifact cross-check, not a result.
- **Fix at:** `code/run_joint_diagnostic.py:374`, `code/run_joint_diagnostic.py:414-416`, `code/run_joint_diagnostic.py:92-93`, `code/run_joint_diagnostic.py:487-505`

#### `CMP-020` [MED] Module-level pickle-alias hack mutates sys.modules['__main__'] at import time
`code/run_joint_diagnostic.py:67-74` · safety · status:live · effort:medium · confidence:high
- **What:** Identical to the compositional file: at import this injects CachedNMNIST/CachedSHD/SubsampledByClass into sys.modules['__main__'] so a torch.load(weights_only=False) of a cache pickled under __main__ resolves. Import-time side effect / __main__ namespace pollution, duplicated across both files, and the torch.load it enables executes arbitrary pickle. It only helps when the cache was pickled while run_joint_magnitude_ablation ran as __main__.
- **Evidence:** for _cn in ('CachedNMNIST','CachedSHD','SubsampledByClass'): setattr(sys.modules['__main__'], _cn, getattr(_mag_module,_cn))
- **Fix:** Same as CMP-008: rebuild caches under an importable module or use a custom Unpickler mapping; remove the __main__ mutation and share one helper across both scripts.
- **Fix at:** `code/run_joint_diagnostic.py:67-74`, `code/run_compositional_crossmodal.py:87-95`
- _related: CMP-008_

#### `CMP-021` [MED] No cache in-repo means running triggers a ~23 GB tonic render; the 'avoids re-render' comment is false here
`code/run_joint_diagnostic.py:659-668` · safety · status:live · effort:small · confidence:high
- **What:** main calls _build_modality_loaders(cfg), which renders N-MNIST and SHD from tonic when the cache is absent. No results/ cache exists in this repo, yet the module comment says the alias exists so 'the existing cache loads without a 23 GB re-render'. The __main__ guard keeps import safe, but explicit execution silently starts the full render/download.
- **Evidence:** loaders, cached = _build_modality_loaders(cfg)  # comment claims 'without a 23 GB re-render' but no cache present
- **Fix:** Fail fast with a clear message when the cache directory is missing, and document the dataset prerequisite. (Same class as CMP-009.)
- **Fix at:** `code/run_joint_diagnostic.py:659-668`, `code/run_joint_diagnostic.py:66-70`
- _related: CMP-009_

#### `CMP-022` [MED] Aggregation double-counts a seed when a timestamped rerun CSV coexists with its base
`code/run_joint_diagnostic.py:469-505` · research-integrity · status:live · effort:small · confidence:medium
- **What:** load_rows_from_disk globs cell_*_seed*.csv, which matches both cell_<kind>_supcon_<x>_seedN.csv and the __<ts>.csv variant _cell_csv_path writes when the base exists. write_summary_and_report means over all gate-passing rows per (kind,use_supcon) with no collapse to unique seed, so a rerun inflates n_seeds and corrupts the mean/std - the same near-duplicate inflation documented for generate_figures_3seed.
- **Evidence:** for p in sorted(RESULTS_DIR.glob('cell_*_seed*.csv')) ... vals = [r[m] for r in sub if r.get('gate_passed')]  # no unique-seed collapse
- **Fix:** Collapse to one row per (model_kind, use_supcon, seed) (latest timestamp) before aggregating; warn on duplicates.
- **Fix at:** `code/run_joint_diagnostic.py:469-484`, `code/run_joint_diagnostic.py:487-505`, `code/run_joint_diagnostic.py:357-362`
- _related: CMP-006_

#### `CMP-024` [MED] File is a structural clone of run_compositional_crossmodal.py and the magnitude harness
`code/run_joint_diagnostic.py:1` · architecture · status:live · effort:large · confidence:high
- **What:** set_seed, _build_cfg, _mean/_std, _binom_z, load_rows_from_disk, write_summary_and_report, _report_spiking_stats and the main/aggregate-only scaffolding are near-duplicates of the sibling file and run_joint_magnitude_ablation.py. Shared bugs (dedup, spike-metric semantics, dead-config) must be fixed in multiple copies.
- **Evidence:** set_seed(130), _build_cfg(624), _mean/_std(441), _binom_z(451), load_rows_from_disk(469), write_summary_and_report(487) mirror the sibling
- **Fix:** Extract a shared importable harness module and import it from both scripts.
- **Fix at:** `code/run_joint_diagnostic.py:130-135`, `code/run_joint_diagnostic.py:441-505`, `code/run_joint_diagnostic.py:336-349`, `code/run_compositional_crossmodal.py:146-165`
- _related: CMP-013_

#### `JNT-003` [MED] Silent outcome-dependent seed exclusion via probe>40 assert inside a swallowing try/except
`code/run_joint_magnitude_ablation.py:800-801` · research-integrity · status:live · effort:small · confidence:medium
- **What:** A per-seed probe-accuracy assertion, combined with the main-loop try/except, silently drops any seed whose within-modality probe falls below 40% from the aggregate, biasing mean/std.
- **Evidence:** run_one_arm asserts probe['visual']>40 and probe['audio']>40 (800-801). main wraps each arm/seed in try/except that prints a traceback and continues (917-928), so a failed assert removes that seed from all_results; aggregates then run over the survivors with n=len(sub) (940,948-951). Excluding seeds on an outcome threshold is outcome-dependent filtering.
- **Fix:** Do not gate inclusion on a performance threshold. Record the probe value and report it; if the gate is a genuine validity check, fail the whole sweep loudly rather than dropping one seed. Distinguish real exceptions (OOM) from the assert.
- **Fix at:** `code/run_joint_magnitude_ablation.py:800-801`, `code/run_joint_magnitude_ablation.py:917-928`
- _related: JNT-002_

#### `JNT-004` [MED] Leakage assert uses a non-element-wise idiom that does not mean what it reads
`code/run_joint_magnitude_ablation.py:488` · correctness · status:live · effort:trivial · confidence:high
- **What:** The leakage guard checks 'any element >=6 AND any element <=9' rather than 'any element in [6,9]'. It happens to catch a leak for the 0-9 label range but is the wrong idiom and can false-pass other ranges.
- **Evidence:** assert not ((labels >= 6).any() and (labels <= 9).any()) at 488 (duplicated at 776, 778). Because (labels>=6).any() and (labels<=9).any() are evaluated independently, the intended element-wise membership test ((labels>=6)&(labels<=9)).any() is not what runs. For labels in 0-9 a label 8 trips both terms so a leak is caught, but the idiom is incorrect and fragile.
- **Fix:** Replace all three with element-wise: assert not ((labels >= 6) & (labels <= 9)).any().
- **Fix at:** `code/run_joint_magnitude_ablation.py:488`, `code/run_joint_magnitude_ablation.py:776`, `code/run_joint_magnitude_ablation.py:778`
- _related: JNT-032_

#### `JNT-006` [MED] _magnitude_sanity reports a mislabeled, tie-broken Spearman that ignores the magnitude head
`code/run_joint_magnitude_ablation.py:723-753` · research-integrity · status:live · effort:small · confidence:high
- **What:** The function's docstring promises Spearman rho between magnitude and digit value, but it correlates the first PCA direction of z with digit value using the no-ties Spearman formula on heavily tied data, and discards the magnitude-head output.
- **Evidence:** Docstring 724 says 'Spearman rho between |z| and digit value, and MSE of magnitude head'. Code computes mag (733) then never uses it for rho; instead it PCA-projects z (743-746) and applies rho = 1 - 6*d2/(n*(n^2-1)) (751-752), the tie-free closed form, to values with ~n/10 ties per digit. Result fields mag_rho_visual/audio (845-846) are summarized as head sanity (986-987).
- **Fix:** If head sanity is intended, correlate the magnitude-head output with digit value using scipy.stats.spearmanr (tie-corrected). Otherwise rename the metric to reflect PC1-vs-value; do not label a PC1 correlation as |z| or magnitude-head rho.
- **Fix at:** `code/run_joint_magnitude_ablation.py:723-753`

#### `JNT-007` [MED] SNN accuracies reported without spike-rate/sparsity or timestep cost
`code/run_joint_magnitude_ablation.py:786-788` · research-integrity · status:live · effort:medium · confidence:high
- **What:** Every accuracy in this experiment comes from spiking stems that time-sum spikes, but no spike rate/sparsity or timestep count is computed or reported alongside, violating the 'accuracy never travels alone' rule.
- **Evidence:** acc_visual_all/acc_audio_all (786-787), probe_visual/audio_6_9 (788,829-830), and the printed/CSV/report figures (819-843) are outputs of VisualStem/AudioStem which sum spikes over T (324,352). No sparsity or T is recorded anywhere.
- **Fix:** Compute and log mean spike rate/sparsity per stem plus the timestep counts (T=25 visual, T=100 audio) in the result dict, CSV and report next to every accuracy.
- **Fix at:** `code/run_joint_magnitude_ablation.py:786-788`, `code/run_joint_magnitude_ablation.py:822-843`
- _related: JNT-022_

#### `JNT-008` [MED] VisualStem/AudioStem re-implemented independently of the provenance backbone
`code/run_joint_magnitude_ablation.py:288-352` · duplication · status:live · effort:medium · confidence:high
- **What:** The visual and audio SNN stems are hand-copied with divergent final dimensions instead of reusing the provenance backbone, creating a third drifting copy of the same architecture across the repo.
- **Evidence:** VisualStem (288-324) and AudioStem (327-352) duplicate the provenance canonical stems (m7_trimodal_run_provenance.py:258-297 visual SNN_Backbone; 307-317 audio ModuleDict) but end at 256-d with 4 LIFs. run_joint_shared_encoder.py:280-352 is a third variant (adds fc2 1024->512 and fc_proj, 5 LIFs). No shared source of truth.
- **Fix:** Extract one parameterized visual conv stem and audio MLP stem into an importable module and consume it from provenance and both joint scripts; if the joint experiments deliberately use a smaller stem, document that deviation in one place.
- **Fix at:** `code/run_joint_magnitude_ablation.py:288-352`, `code/m7_trimodal_run_provenance.py:258-297`, `code/m7_trimodal_run_provenance.py:307-317`, `code/run_joint_shared_encoder.py:280-352`
- _related: JNT-018_

#### `JNT-009` [MED] Dead imports from run_correspondence_anchor_v2 pull in a heavy transitive dependency
`code/run_joint_magnitude_ablation.py:56-59` · dependencies · status:live · effort:trivial · confidence:high
- **What:** Two symbols imported from run_correspondence_anchor_v2 are never used, yet importing that module transitively imports utils.cross_modal_transfer, so an unrelated breakage there fails this script at import for functionality it does not use.
- **Evidence:** Imports 'compute_held_out_metrics as _compute_held_out_metrics_v2' and 'run_procrustes' (56-59). Neither is called: the file defines its own compute_held_out_metrics (572) and run_procrustes_z (648, called at 795). Importing run_correspondence_anchor_v2 runs 'from utils.cross_modal_transfer import evaluate_cross_modal_transfer' (run_correspondence_anchor_v2.py:45).
- **Fix:** Delete the run_correspondence_anchor_v2 import block (56-59); the file is self-sufficient.
- **Fix at:** `code/run_joint_magnitude_ablation.py:56-59`
- _related: JNT-010_

#### `JNT-017` [MED] Parallel Config dataclass re-declares ExperimentConfig hyperparameters (config-drift hazard)
`code/run_joint_shared_encoder.py:72-88` · architecture · status:live · effort:medium · confidence:high
- **What:** A second config dataclass re-declares shared hyperparameters with its own defaults, so values like beta or lr can silently diverge between M7 and this experiment.
- **Evidence:** Config (72-88) re-declares lr=1e-3, weight_decay=1e-4, beta=0.9, batch_size, num_workers and seeds, parallel to ExperimentConfig (m7_trimodal_run_provenance.py:33-74). Two independent sources of truth for the same knobs.
- **Fix:** Compose or derive from ExperimentConfig for the shared hyperparameters, keeping only experiment-unique fields (d_stem, d_embed, supcon_temperature, samples_per_class, epochs) local, so one edit governs both.
- **Fix at:** `code/run_joint_shared_encoder.py:72-88`, `code/m7_trimodal_run_provenance.py:33-74`
- _related: JNT-013_

#### `JNT-018` [MED] Own VisualStem/AudioStem duplicate the provenance backbone and diverge from the ablation's stems
`code/run_joint_shared_encoder.py:280-352` · duplication · status:live · effort:medium · confidence:high
- **What:** A third copy of the visual/audio SNN stems, structurally different from both the provenance backbone and the magnitude ablation's stems.
- **Evidence:** VisualStem (280-320) and AudioStem (323-352) duplicate provenance (m7_trimodal_run_provenance.py:258-297,307-317) and differ from run_joint_magnitude_ablation.py:288-352 (this variant adds fc2 1024->512 and fc_proj 512->d_stem, 5 LIFs vs the ablation's 4).
- **Fix:** Consume one shared parameterized stem module (see JNT-008) instead of a per-script copy.
- **Fix at:** `code/run_joint_shared_encoder.py:280-352`, `code/m7_trimodal_run_provenance.py:258-297`, `code/run_joint_magnitude_ablation.py:288-352`
- _related: JNT-008_

#### `JNT-019` [MED] Import-time filesystem side effect: RESULTS_DIR.mkdir at module top level
`code/run_joint_shared_encoder.py:56-57` · safety · status:live · effort:trivial · confidence:high
- **What:** Merely importing the module creates a results directory on disk, a side effect that static analysis and tooling should not trigger.
- **Evidence:** RESULTS_DIR = Path('results/joint_shared_encoder'); RESULTS_DIR.mkdir(parents=True, exist_ok=True) at module scope (56-57). The magnitude ablation correctly defers its mkdir into main() (run_joint_magnitude_ablation.py:866).
- **Fix:** Move the mkdir into main()/run_one_seed; module import must have no filesystem side effects.
- **Fix at:** `code/run_joint_shared_encoder.py:56-57`

#### `JNT-020` [MED] SHD train cache uses identity labels (%10) but test cache stores raw 0-19; reconciled only post-hoc
`code/run_joint_shared_encoder.py:414-472` · correctness · status:live · effort:small · confidence:high
- **What:** The train and test SHD caches use different label conventions under the same filename convention; correctness depends on a single post-hoc %10 that is easy to remove or overlook.
- **Evidence:** Train cache stores int(l) % 10 (414); test cache stores raw int(l) (472). run_one_seed fixes only the test side later: aud_lbl_all = aud_lbl_all % 10 (848). Without that %10 the HELD_OUT mask (6,7,8,9) drops German 6-9 (labels 16-19) and TRAIN mask drops German 0-5; it is load-bearing and asymmetric with the train convention.
- **Fix:** Apply %10 at cache-build time for BOTH train and test (line 472) so the on-disk convention is uniform; then the 848 %10 becomes a harmless no-op that can be removed.
- **Fix at:** `code/run_joint_shared_encoder.py:414`, `code/run_joint_shared_encoder.py:472`, `code/run_joint_shared_encoder.py:848`

#### `JNT-021` [MED] Meaningless 'held-out classifier accuracy' recorded and summarized as a metric
`code/run_joint_shared_encoder.py:874-880` · research-integrity · status:live · effort:small · confidence:high
- **What:** A 6-way classifier's argmax is compared against an arbitrary relabeling of held-out digits 6-9, and the resulting number is stored and averaged as an accuracy.
- **Evidence:** classifier has NUM_TRAIN_CLASSES=6 outputs (373). For 6-9 the code computes argmax == (lbl-6).clamp(0,5) (879-880), mapping 6->0..9->3; the comment says 'map held-out labels 6-9 to random guesses'. Fields held_out_classifier_acc_vis/aud (891-892) go into the per-seed CSV and summaries.
- **Fix:** Remove these fields or replace with a defensible metric (the embedding-prototype retrieval already computed). Do not emit an 'accuracy' defined against arbitrary relabeling.
- **Fix at:** `code/run_joint_shared_encoder.py:874-880`, `code/run_joint_shared_encoder.py:891-892`

#### `JNT-022` [MED] SNN accuracies/retrieval reported without spike-rate/sparsity or timestep cost
`code/run_joint_shared_encoder.py:570-580` · research-integrity · status:live · effort:medium · confidence:high
- **What:** Validation accuracies and retrieval come from spiking stems that time-sum spikes, but no spike rate/sparsity or timestep count is reported alongside.
- **Evidence:** acc_vis/acc_aud (578-579) logged each epoch (823-825); stems sum spikes over T (320,352). No sparsity or T recorded.
- **Fix:** Record per-stem spike rate/sparsity and timestep counts (T=25 visual, T=100 audio) next to every accuracy/retrieval number.
- **Fix at:** `code/run_joint_shared_encoder.py:570-580`, `code/run_joint_shared_encoder.py:823-825`
- _related: JNT-007_

#### `JNT-023` [MED] Eager full-dataset materialization to read labels (N-MNIST decoded multiple times)
`code/run_joint_shared_encoder.py:104-107` · performance · status:live · effort:medium · confidence:high
- **What:** Label-based filtering indexes the entire transformed dataset just to read labels, decoding every sample; the paired dataset then re-indexes the full datasets again.
- **Evidence:** ClassFilteredDataset.__init__ loops base_dataset[i] over the whole dataset discarding data (104-107); for cached_nmnist (DiskCachedDataset over ToFrame) this decodes every N-MNIST sample. PairedDigitDatasetFiltered.__init__ re-indexes the full visual+audio datasets (143-155). N-MNIST train is decoded several times per seed before training.
- **Fix:** Expose a labels-only accessor or cache label arrays once; avoid indexing the transformed dataset purely to filter by label.
- **Fix at:** `code/run_joint_shared_encoder.py:104-107`, `code/run_joint_shared_encoder.py:143-155`

#### `JNT-024` [MED] PairedDigitDatasetFiltered encodes flat index by class VALUE but decodes by positional INDEX
`code/run_joint_shared_encoder.py:182-196` · correctness · status:live · effort:small · confidence:medium
- **What:** The flat-index encode/decode is asymmetric: it is only correct when the class list is exactly 0..N-1, and silently mis-decodes for any other class list.
- **Evidence:** Encode uses the class value: int(self.classes[c] + s*self.num_classes) (186). Decode uses the position: c = flat_idx % self.num_classes; s = flat_idx // self.num_classes (193-196), then indexes self.visual_indices[c] keyed by class value (203). Correct only for classes==(0,1,2,3,4,5); always called with TRAIN_CLASSES so it works today but is a latent bug for non-zero-based/non-contiguous class lists.
- **Fix:** Encode with the positional index (c_index + s*num_classes) and map to the class value on decode via self.classes[c_index]; or assert self.classes == tuple(range(num_classes)).
- **Fix at:** `code/run_joint_shared_encoder.py:182-196`

#### `ALGN-018` [MED] OrdinalCorrespondenceModel duplicates CorrespondenceModelV2 except for anchors
`code/run_ordinal_anchor.py:247-326` · duplication · status:live · effort:small · confidence:high
- **What:** OrdinalCorrespondenceModel (247-326) is a near-verbatim copy of CorrespondenceModelV2 (run_correspondence_anchor_v2.py:164-234): identical head construction, _get_backbone_features (282-299), project (301-306), forward_with_shared (308-314), task_accuracy (316-326). The only real difference is anchors = build_ordinal_anchors(...) instead of the QR construction. Two full copies of the audio-forward re-implementation (ALGN-007) live here too.
- **Evidence:** lines 282-326 mirror run_correspondence_anchor_v2.py:190-234; only the register_buffer('anchors', ...) source differs (line 277 vs v2:182-186)
- **Fix:** Make OrdinalCorrespondenceModel subclass CorrespondenceModelV2 and override only anchor construction, or parameterize the anchor builder.
- **Fix at:** `code/run_ordinal_anchor.py:247-326`, `code/run_correspondence_anchor_v2.py:164-234`
- _related: ALGN-007, ALGN-008_

#### `CLE-005` [MED] cfg is a hand-copied duplicate of run_cl_rehearsal.py's config (drift hazard)
`code/run_rehearsal_transfer_eval_only.py:21-35` · duplication · status:live · effort:small · confidence:high
- **What:** The ExperimentConfig block (22-35) is a manual copy of the config in run_cl_rehearsal.py, flagged by its own comment `# --- Reconstruct config (must match run_cl_rehearsal.py) ---` (21). A hand-maintained 'must match' duplicate is a classic drift hazard, and it has already drifted in effect (see the replay_weight discrepancy, CLE-003). If the source config changes, this eval silently evaluates under a different config than the run that produced the checkpoint.
- **Evidence:** Line 21 comment plus the literal block 22-35, which mirrors run_cl_rehearsal.py:377-390.
- **Fix:** Load the config that was saved alongside the checkpoint (the run writes config_*.json / master_config.json) via ExperimentConfig(**json.load(...)) rather than re-declaring it, as eval_transfer_fixed.py and phase1_sanity_check.py already do.
- **Fix at:** `code/run_rehearsal_transfer_eval_only.py:21-35`
- _related: CLE-032_

#### `TST-009` [MED] Entire alignment smoke test executes at import time (no __main__ guard, no functions)
`code/test_cross_modal_alignment.py:1-169` · safety · status:live · effort:small · confidence:high
- **What:** Like the provenance smoke test, there is no __main__ guard and no functions; the body (model build line 75, forward passes lines 86-96, orthogonal/SupCon checks lines 103-115, a 2-epoch alignment run lines 130-138, and 3 optimizer steps lines 152-157) all execute on import. Import==execution. VERIFIED SAFE from the 4-5h trap: device='cpu' (line 58), FakePairedDataset random tensors (lines 26-40), and it imports only m7_trimodal_run_provenance (guarded) plus Dataloaders.paired_digit_loader — not the unguarded m7_trimodal_run. But collecting/importing it still runs real backprop and can error during collection.
- **Evidence:** No `if __name__`/`def ` (only class FakePairedDataset and methods); executable top-level statements lines 67-169; device cpu at line 58; align_cross_modal 2-epoch run at lines 130-138.
- **Fix:** Move the body into test_* functions with a __main__ guard so import/collection does not run training.
- **Fix at:** `code/test_cross_modal_alignment.py:67-169`
- _related: TST-001, TST-002_

#### `TST-010` [MED] Flaky assertion: cross-modal loss must strictly decrease over just 2 epochs on near-noise fake data
`code/test_cross_modal_alignment.py:142-143` · correctness · status:live · effort:small · confidence:high
- **What:** Line 142 asserts history['xmscl_loss'][1] < history['xmscl_loss'][0] — i.e. the CrossModalSupCon loss must strictly drop between epoch 1 and epoch 2. The data is FakePairedDataset, where per-class signal is c*0.05 added onto randn*0.1 noise (lines 37-39), i.e. mostly noise; optimization uses AdamW with CosineAnnealingLR(T_max=num_epochs=2) (m7_trimodal_run_provenance.py:702). Over only two epochs on near-noise data a strict monotonic decrease is not guaranteed, so this test can fail spuriously on some seeds/runs — a false alarm rather than a real regression signal.
- **Evidence:** code/test_cross_modal_alignment.py:142-143 strict `<` assertion; FakePairedDataset signal c*0.05 over 0.1 noise at lines 37-39; num_epochs=2 at line 132; CosineAnnealingLR(T_max=num_epochs) in align_cross_modal (m7_trimodal_run_provenance.py:702).
- **Fix:** Make the check robust: assert loss[-1] <= loss[0] with a tolerance, or assert a downward trend over more epochs, or compare mean of last-half vs first-half. Seed the run and/or increase epochs so the trend is reliable.
- **Fix at:** `code/test_cross_modal_alignment.py:130-143`

#### `TST-017` [MED] Requires N-MNIST/SHD datasets not in the repo — becomes a large tonic download once the import is fixed
`code/test_ordinal_anchor_smoke.py:61` · dependencies · status:live · effort:medium · confidence:high
- **What:** main() calls _build_digit_loaders(cfg) (line 61). That builder (run_correspondence_anchor_v2.py:67-91) constructs tonic.datasets.NMNIST and SHD with save_to='./ICONS_M7/datasets' and mkdir(parents=True) — the repo has no ICONS_M7 datasets (per CLAUDE.md/AUDIT_SPEC). So once TST-016 is fixed, running this file will trigger a multi-GB tonic dataset DOWNLOAD (or crash if offline), and the path is CWD-relative so it only resolves as intended when run from code/. Today this is masked by the ImportError (TST-016), which is why severity is medium rather than a live download; but any naive 'fix the import and run it' will hit the download.
- **Evidence:** code/test_ordinal_anchor_smoke.py:61 `loaders = _build_digit_loaders(cfg)`; run_correspondence_anchor_v2.py:67-91 build NMNIST/SHD via tonic with save_to='./ICONS_M7/datasets' and data_dir.mkdir; repo has no ICONS_M7 datasets.
- **Fix:** For a true smoke test, feed a tiny fake/dummy paired loader (as test_cross_modal_alignment.py does with FakePairedDataset) instead of _build_digit_loaders, or gate the real-data path behind an explicit env flag / pytest skip so it never downloads unexpectedly.
- **Fix at:** `code/test_ordinal_anchor_smoke.py:60-61`
- _related: TST-016, TST-018_

#### `TST-018` [MED] Requires a cached-feature artifact and a checkpoint that do not exist in the repo
`code/test_ordinal_anchor_smoke.py:63-73` · dependencies · status:live · effort:medium · confidence:high
- **What:** main() calls _load_cached_features(device) (line 64) and run_one_seed(42, cfg, loaders, cached, device, head_type='mlp') (line 73). run_one_seed (run_ordinal_anchor.py:395-411) does torch.load(CKPT_SOURCE) where CKPT_SOURCE = REPO_ROOT/results/cl_confirmation/checkpoints/A_no_align_seed42/final_shd_shd.pt (run_ordinal_anchor.py:63-70). The repo has no results/ directory and no *.pt checkpoints, and _load_cached_features expects a cached feature file. So even past TST-016 and TST-017, this crashes with FileNotFoundError on the missing checkpoint/cache. Masked today by TST-016.
- **Evidence:** code/test_ordinal_anchor_smoke.py:64 `_load_cached_features(device)`; line 73 run_one_seed(...); torch.load(CKPT_SOURCE) at run_ordinal_anchor.py:410; CKPT_SOURCE path at run_ordinal_anchor.py:63-70; verified no results/ dir and no *.pt in repo.
- **Fix:** For a smoke test, provide a tiny synthetic cached-feature dict and either construct a fresh model instead of loading CKPT_SOURCE or point at a small fixture checkpoint; otherwise mark the test skip when the artifacts are absent so it degrades gracefully.
- **Fix at:** `code/test_ordinal_anchor_smoke.py:63-73`, `code/run_ordinal_anchor.py:63-70`
- _related: TST-016, TST-017_

#### `TST-001` [MED] None of the three code/ smoke tests are real pytest tests, and none are in the validation loop
`code/test_provenance_smoke.py:1` · quality · status:live · effort:medium · confidence:high
- **What:** All three files are named test_*.py (which signals a pytest suite) but define ZERO test_-prefixed functions: test_provenance_smoke.py and test_cross_modal_alignment.py put their whole body at module top level, and test_ordinal_anchor_smoke.py wraps everything in main(). pytest collects 0 test items from any of them. Worse, pytest.ini pins testpaths=tests (pytest.ini:2), so the canonical validation loop `pytest tests/` never even looks at code/. These files therefore contribute nothing to the safe validation loop and give a false impression of test coverage for the M7 provenance/alignment/ordinal code paths (which the repo-root tests/ suite does NOT exercise — it only tests the legacy bi-modal package).
- **Evidence:** pytest.ini:2 `testpaths = tests`; grep for `def test_` across the three files returns nothing; test_provenance_smoke.py has statements at module scope lines 17-110; test_cross_modal_alignment.py lines 67-169; test_ordinal_anchor_smoke.py logic is inside main() (line 22) under a __main__ guard (line 83).
- **Fix:** Convert each smoke script into pytest test_* functions (assertions inside functions, no work at import), add a __main__ guard, and relocate the dataset-free ones (provenance + alignment, both CPU/dummy-data) into tests/ so `pytest tests/` actually runs them. Keep dataset/checkpoint-dependent ordinal logic out of the default collection or mark it skip/xfail with a clear reason.
- **Fix at:** `code/test_provenance_smoke.py:17-110`, `code/test_cross_modal_alignment.py:67-169`, `code/test_ordinal_anchor_smoke.py:22-84`, `pytest.ini:2`
- _related: TST-002, TST-009, TST-019_

#### `TST-002` [MED] Entire smoke test executes at import time (no __main__ guard, no functions)
`code/test_provenance_smoke.py:1-110` · safety · status:live · effort:small · confidence:high
- **What:** There is no `if __name__ == '__main__'` guard and no functions; every statement (lines 17-110) runs the moment the module is imported. That includes constructing the model and running one epoch of ContinualLearner.train_task plus several evaluate() calls. Importing the module IS running the smoke test, so any pytest collection or plain `import` executes training, and a failed assertion surfaces as an import/collection error rather than a clean test failure. VERIFIED SAFE regarding the 4-5h trap: it forces device='cpu' (line 31) and uses DummyLoader random tensors (lines 36-54), and its only heavy import m7_trimodal_run_provenance is __main__-guarded (that module, line 1378) and does not import the unguarded m7_trimodal_run — so importing this file does NOT download datasets or start the long run. The problem is purely that import==execution.
- **Evidence:** No `if __name__` / `def ` in file; top-level executable statements lines 17-34 and 52-110; device forced cpu at line 31; DummyLoader defined lines 36-54.
- **Fix:** Wrap the body in test functions (e.g. test_random_init_baseline, test_zero_shot_near_chance) and a __main__ guard so importing/collecting the module does not execute training.
- **Fix at:** `code/test_provenance_smoke.py:17-110`
- _related: TST-001, TST-009_

#### `TST-003` [MED] Assertion tolerance (<8.0 pp) contradicts the success message it prints (~3pp)
`code/test_provenance_smoke.py:62-63` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** Line 62 asserts `abs(ri - chance) < 8.0` but line 63 prints `All random-init baselines within ~3pp of chance`. The asserted bound (8.0 pp) is nearly 3x the claimed bound (3 pp), so the printed claim is not what the code checks. The 8.0 is also a magic tolerance with no derivation, and it is statistically under-powered: each random_init_baseline evaluates on DummyLoader(10, ...) which yields 10 batches x 4 = ~40 samples (lines 52-54, 42-48), averaged over n_seeds=2 (line 59). At that sample size an accuracy estimate has a large spread, so the 8.0 bound is arbitrary and can occasionally fail spuriously. Under the repo's RULE-1 culture, printing 'within ~3pp' while enforcing 8pp is a mislabeled claim.
- **Evidence:** code/test_provenance_smoke.py:62 `assert abs(ri - chance) < 8.0`; line 63 message `within ~3pp of chance`; sample count from DummyLoader(10,10) at lines 52-54 and n_seeds=2 at line 59.
- **Fix:** Pick one tolerance, justify it from the sample size, and make the printed message match the asserted bound. Either raise the message to ~8pp or tighten the assert to ~3pp AND increase the DummyLoader sample count so ~3pp is statistically defensible.
- **Fix at:** `code/test_provenance_smoke.py:62-63`
- _related: TST-004_

#### `TST-004` [MED] The 'key assertion' is not an assertion — it is a non-enforcing if/else that always passes
`code/test_provenance_smoke.py:89-93` · correctness · status:live · effort:trivial · confidence:high
- **What:** The comment on lines 87-88 calls the near-chance zero-shot check 'The key assertion', but lines 89-93 implement it as an `if abs(...) < 3.0: print(ok) else: print(warning)` — it never fails the test. The module still prints 'SMOKE TEST PASSED' (line 109) regardless of whether SHD zero-shot is near chance. So the single most important property this smoke test claims to verify (no spurious cross-modal signal / no data leakage) is not enforced. Compounding it, the 3.0 pp band is tighter than the noise floor of a ~40-sample evaluate() (DummyLoader(10,20), line 53), so the warning branch fires often yet nothing fails.
- **Evidence:** code/test_provenance_smoke.py:87-88 comment 'The key assertion'; lines 89-93 if/else that only prints; line 109 prints PASSED unconditionally; evaluate() sample count ~40 from dummy_shd = DummyLoader(10, 20) at line 53.
- **Fix:** Promote the check to a real assertion (assert abs(acc_shd_zero - chance_shd) < TOL) with a defensible TOL and larger sample count, or explicitly document why it is only a warning. Do not label a warning as the 'key assertion'.
- **Fix at:** `code/test_provenance_smoke.py:87-93`
- _related: TST-003_


---
## LOW (105)

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

#### `ARCH-020` [LOW] 95% CI uses normal z=1.96 with n=2-3 seeds
`code/archive/run_cl_confirmation.py:497` · research-integrity · status:archive · effort:small · confidence:medium
- **What:** aggregate_confirmation_results computes ci95 = 1.96 * std / sqrt(n) (line 497) and writes ci95_lower/ci95_upper labeled as 95% CI. With the seed policy [42,123,456] n is 2-3, where the normal approximation is inappropriate — a t-multiplier (e.g. t_.975,df=2 ~ 4.30) would be far wider. Labeling a z-based interval '95% CI' at n<=3 overstates precision.
- **Evidence:** ci95 = 1.96 * std / np.sqrt(n) line 497; std uses ddof=1 line 495; seeds=[42,123,456] line 539.
- **Fix:** Use scipy.stats.t.ppf for small-n intervals, or report mean +/- std with explicit n rather than a CI, and label accordingly. Retirement: any 'CI' from this script should be treated as under-wide.
- **Fix at:** `code/archive/run_cl_confirmation.py:495-497`

#### `ARCH-016` [LOW] Dead vis_indices/aud_indices loop that still consumes the RNG
`code/archive/run_correspondence_anchor.py:313-322` · dead-code · status:archive · effort:trivial · confidence:high
- **What:** In build_paired_feature_dataset the first per-class loop builds vis_indices/aud_indices via rng.choice (lines 313-322) and they are never read — a follow-up comment 'Actually we need PAIRED samples' recomputes everything in a second loop (lines 326-343). The dead loop is not inert: it advances the same rng (created line 312), so the real paired selection at lines 335-336 draws different samples than it would without the dead loop. Removing it would change which samples train the projection heads, i.e. it is dead code with a numeric side effect.
- **Evidence:** unused vis_indices/aud_indices lines 313-322 vs paired recompute lines 326-343; shared rng line 312 consumed by both.
- **Fix:** Delete the dead loop, but note it will change results (RNG stream shifts); acceptable only because the file is being retired. If preserving v1 outputs bit-for-bit, leave it. Prefer DELETE the whole file (ARCH-001).
- **Fix at:** `code/archive/run_correspondence_anchor.py:313-322`
- _related: ARCH-001_

#### `ARCH-017` [LOW] Null baseline shuffles only 2 held-out classes -> 50% chance it equals same-class
`code/archive/run_correspondence_anchor.py:524-531` · research-integrity · status:archive · effort:small · confidence:medium
- **What:** compute_held_out_alignment builds its null baseline by np.random.shuffle of the held-out class list (line 526), but held_out_classes=(8,9) has just two elements, so the shuffle is the identity with probability 0.5. When identity, null_sims compares each class prototype with itself, making null_baseline equal to same-class similarity and the verdict test (same_class > null_baseline + 0.1, line 651) meaningless. Uses the global numpy RNG (seeded line 550), not the local rng. Compounds ARCH-015.
- **Evidence:** np.random.shuffle(shuffled_classes) line 526 with held_out_classes length 2; verdict comparison line 651.
- **Fix:** Use a permutation that excludes fixed points, or average many derangements, or use off_diag as the control; and use a local Generator. Retirement: another reason v1 held-out numbers are untrustworthy.
- **Fix at:** `code/archive/run_correspondence_anchor.py:524-531`, `code/archive/run_correspondence_anchor.py:651`
- _related: ARCH-015, ARCH-001_

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

#### `FIG-025` [LOW] argmax over all-zero head spike counts silently predicts class 0
`code/final_evaluation.py:183-184` · correctness · status:live · effort:small · confidence:high
- **What:** preds = np.argmax(head_counts, axis=1) (183). For any sample with zero total head spikes, head_counts row is all zeros and np.argmax returns index 0 - the sample is silently classified as class 0, biasing accuracy/confusion toward class 0. zero_count (182) records how many such samples occur and the report prints the total (291), but the predictions still default to 0 rather than being marked no-prediction. The HONEST STATEMENT relies on 'Confirms 0 all-zero' - if that count is ever nonzero, accuracy already silently absorbed those as class-0 guesses.
- **Evidence:** preds = np.argmax(head_counts, axis=1) with zero_count tracked separately
- **Fix:** Mark all-zero samples as an explicit no-prediction sentinel (e.g. -1) and exclude/flag them in accuracy and confusion, instead of letting argmax default to class 0.
- **Fix at:** `code/final_evaluation.py:181-184`
- _related: FIG-026_

#### `FIG-026` [LOW] mean_confusion_matrix computes a single (non-mean) CM; dead 'p>=0' guard; transposed cm[pred,true] orientation
`code/final_evaluation.py:187-192` · quality · status:live · effort:trivial · confidence:high
- **What:** mean_confusion_matrix (187-192) computes counts for ONE seed's predictions, not a mean - the averaging happens later at np.mean(spinnaker_cms, axis=0) (276) - so the name is misleading. The guard `if p >= 0` (190) is dead: p comes from np.argmax and is always >=0 (see FIG-025 - all-zero rows still yield 0, never negative). cm[p, t] += 1 (191) stores [predicted, true], the transpose of the conventional cm[true, pred]; the print loop labels rows r and columns by class index (294-297), so a reader assuming sklearn convention would misread rows/columns.
- **Evidence:** def mean_confusion_matrix(...): if p >= 0: cm[p, t] += 1
- **Fix:** Rename to confusion_matrix; drop the dead p>=0 guard (or make it meaningful once no-prediction sentinels exist, FIG-025); document the [pred,true] orientation or switch to [true,pred].
- **Fix at:** `code/final_evaluation.py:187-192`, `code/final_evaluation.py:294-297`
- _related: FIG-025_

#### `FIG-027` [LOW] final_evaluation.py is misfiled: a hardware-verify script in code/ root, not with the archive cluster
`code/final_evaluation.py:1` · architecture · status:live · effort:trivial · confidence:high
- **What:** final_evaluation.py is a SpiNNaker discrete-time-simulator verification script, but it lives in code/ root while its peer hardware-verify scripts are in code/archive/: verify_brian2_raw.py, verify_simulators.py, verify_full.py, verify_tiny.py, debug_brian2_exact.py (confirmed by listing). It even references 'raw Brian2' verification (307) that corresponds to code/archive/verify_brian2_raw.py. Its presence in the main code/ directory, named 'final', implies it is a current M7 deliverable rather than an archived hardware-verification experiment.
- **Evidence:** code/archive/ contains verify_brian2_raw.py, verify_simulators.py, debug_brian2_exact.py; final_evaluation.py is in code/ root
- **Fix:** Move code/final_evaluation.py into code/archive/ alongside the other verify_*/debug_brian2 scripts (or into a dedicated hardware-verify directory).
- **Fix at:** `code/final_evaluation.py:1`
- _related: FIG-020_

#### `FIG-028` [LOW] ConvFrontend duplicates FullModel's conv layers, and both re-implement m7's SNN_Backbone
`code/final_evaluation.py:33-85` · architecture · status:live · effort:small · confidence:high
- **What:** ConvFrontend (33-53) redefines conv1/pool1/lif1/conv2/pool2/lif2 identical to the first half of FullModel (59-64), and both re-implement the backbone that m7_trimodal_run.py defines as SNN_Backbone (conv1 2->32 k5, conv2 32->64 k5, fc1 1600->1024, fc2 1024->512). The architecture is stated three times; a backbone change (e.g. the 2-vs-3 conv issue resolved in contested-facts.md) must be mirrored here or these silently diverge from the trained model.
- **Evidence:** ConvFrontend and FullModel both define conv1/conv2/lif1/lif2 with the same params
- **Fix:** Have ConvFrontend reuse FullModel's sub-modules (or share a single backbone definition) instead of re-declaring the conv stack; ideally import the canonical backbone rather than copying it.
- **Fix at:** `code/final_evaluation.py:33-53`, `code/final_evaluation.py:56-85`
- _related: FIG-017_

#### `FIG-030` [LOW] code/ script reaches into the legacy bi-modal package via sys.path hack
`code/final_evaluation.py:219-220` · dependencies · status:live · effort:trivial · confidence:high
- **What:** final_evaluation.py inserts the repo root on sys.path (219) to import Dataloaders.nmnist_loader.get_nmnist_loaders (220). Verified: Dataloaders/nmnist_loader.py exists and defines get_nmnist_loaders (line 13), so the import resolves - it is not broken. But it couples a code/ script to the legacy bi-modal package (Models/Dataloaders/training/utils) that CLAUDE.md keeps separate from the M7 entry point, and depends on CWD/path assumptions. code/ has no __init__.py, hence the sys.path.insert.
- **Evidence:** sys.path.insert(0, repo_root); from Dataloaders.nmnist_loader import get_nmnist_loaders
- **Fix:** Acceptable for an archived verify script; if kept active, document the dependency on the legacy Dataloaders package and the repo-root sys.path requirement.
- **Fix at:** `code/final_evaluation.py:219-220`
- _related: FIG-027_

#### `RA-013` [LOW] Non-standard 5-seed set (disclosed, distinct purpose)
`code/final_evaluation.py:26` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** SEEDS=[42,123,456,789,999] differs from the project policy [42,123,456]. It is disclosed in the docstring as '5 random seeds for Poisson input encoding' variability, a legitimate distinct purpose, so not a violation - but easily conflated with the CL seed set.
- **Evidence:** line 26; docstring 8,310-311; verdict DISCLOSED
- **Fix:** No code change required; ensure any reported number states '5 Poisson-input seeds', never '5 CL seeds'.
- **Fix at:** `code/final_evaluation.py:26`
- _confirmed by FIG-021; related: RA-011_

#### `FIG-017` [LOW] box() helper and the entire Figure 5 architecture schematic are duplicated across both figure scripts
`code/generate_figures.py:185-252` · duplication · status:live · effort:small · confidence:high
- **What:** The box() helper (generate_figures.py 192-197 vs generate_figures_3seed.py 339-343) and the whole Figure-5 schematic block (generate_figures.py 185-252 vs generate_figures_3seed.py 330-378) are near-identical copies. This is dead duplication maintained in two places and part of why the two scripts collide (FIG-001).
- **Evidence:** def box(...) and Figure 5 present verbatim in both files
- **Fix:** After retiring generate_figures.py (FIG-008) the duplication disappears. If both must exist, factor box()/the schematic into a shared module.
- **Fix at:** `code/generate_figures.py:185-252`, `code/generate_figures_3seed.py:330-378`
- _related: FIG-008_

#### `FIG-010` [LOW] Import-time filesystem side effect: mkdir creates ICONS_M7/results/plots on import
`code/generate_figures_3seed.py:30` · safety · status:live · effort:trivial · confidence:high
- **What:** PLOT_DIR.mkdir(parents=True, exist_ok=True) runs at module level (line 30), so merely importing the module creates ./ICONS_M7/results/plots/ (CWD-relative; only lands where docs expect when CWD is code/). Same pattern in generate_figures.py line 12. Non-destructive, but a side effect that should live inside main().
- **Evidence:** PLOT_DIR.mkdir(parents=True, exist_ok=True) at module scope
- **Fix:** Move the mkdir into main()/the figure-writing path so import has no filesystem effect.
- **Fix at:** `code/generate_figures_3seed.py:30`, `code/generate_figures.py:12`
- _related: FIG-009_

#### `FIG-011` [LOW] POSITIVE: dedup is correct (one row per seed+condition); minor: .first() is first-non-null-per-column, not 'first row'
`code/generate_figures_3seed.py:65-77` · quality · status:live · effort:trivial · confidence:high
- **What:** load_summary correctly (a) filters to summary rows via df['step'].isna() OR blank-step (65) - more robust than the documented isna()-only ritual, (b) exact-dedups (66), (c) warns on near-duplicate seed+condition rows (68-75), and (d) collapses to one row per (seed,condition) with groupby(...).first() (77). This is the CORRECT population for mean/std and avoids the n-inflation trap flagged in CLAUDE.md. Minor precision note: GroupBy.first() returns the first NON-NULL value per column, not the first physical row, so the comment 'Keeping the FIRST row' (74) is slightly inaccurate; for near-duplicate rows with scattered NaNs it could assemble a column-wise mix. Harmless for complete summary rows.
- **Evidence:** summary.groupby(['seed','condition'], as_index=False).first()
- **Fix:** No action required for the dedup. Optionally use .head(1)/.nth(0) if 'first physical row' semantics are intended, and reword the comment at 74.
- **Fix at:** `code/generate_figures_3seed.py:74`, `code/generate_figures_3seed.py:77`

#### `FIG-012` [LOW] Figure 4 hardcodes CONFIG constants (currently correct, disclosed) - single-source-of-truth drift hazard
`code/generate_figures_3seed.py:285-299` · research-integrity · status:live · effort:small · confidence:high
- **What:** SAMPLES_PER_CLASS=50, FEATURE_DIM=512, BYTES=4, CLASSES={nmnist:10,shd:20,dvs:11} and FRAME_ELEMS (285-295) are hardcoded. Verified against source: m7_trimodal_run.py:530 constructs EmbeddingReplayBuffer(feature_dim=512, samples_per_class=50), and time_steps=25 (line 58) matches the nmnist/dvs frame time dims - so the constants are CURRENTLY ACCURATE. The figure title says 'derived from CONFIG, not measured' (322-323) and the arithmetic is printed for audit (304-310), so this is disclosed derivation, NOT fabrication (contrast the legacy [80,150,4], FIG-007). Residual risk: these are hand-copied duplicates of m7 CONFIG; if CONFIG changes they silently go stale while the figure still claims CONFIG-derivation.
- **Evidence:** SAMPLES_PER_CLASS=50 vs m7_trimodal_run.py:530 samples_per_class=50
- **Fix:** Import/read the constants from a single source (the CONFIG dict) rather than re-declaring them, or comment-pin them to m7_trimodal_run.py:530/58 with an update-together note.
- **Fix at:** `code/generate_figures_3seed.py:285-299`
- _related: FIG-007_

#### `FIG-013` [LOW] Figure 4 'Raw dense frames' bar uses DVS pooled 34x34, not raw 128x128
`code/generate_figures_3seed.py:294` · correctness · status:live · effort:trivial · confidence:medium
- **What:** FRAME_ELEMS['dvs'] = 25*2*34*34 (294) uses the POOLED 34x34 resolution (comment: 'pooled 128->34 to match the visual backbone'), but the bar it feeds is labeled 'Raw dense frames (50/class)' (313). DVS-Gesture raw frames are 2x128x128 (per CLAUDE.md), ~14x more elements. So the 'raw' baseline for DVS is understated, which understates the raw total and hence the compression ratio (322) - conservative/anti-flattering, not an inflating error - but internally inconsistent with the 'Raw dense frames' label.
- **Evidence:** 'dvs': 25 * 2 * 34 * 34 under a bar labeled 'Raw dense frames'
- **Fix:** Either compute the DVS raw-frame term at native 128x128 to match the 'raw' label, or relabel the bar to clarify it uses the pooled resolution the model consumes.
- **Fix at:** `code/generate_figures_3seed.py:294`, `code/generate_figures_3seed.py:313`
- _related: FIG-012_

#### `FIG-014` [LOW] aggregate() computes mean/std over ALL numeric columns (incl. seed and all-NaN step); re-validate under pinned pandas 3.0.2
`code/generate_figures_3seed.py:88-89` · numerics · status:live · effort:small · confidence:low
- **What:** sub.mean(numeric_only=True) (88) and sub.std(numeric_only=True, ddof=1).fillna(0.0) (89) reduce over every numeric column, which includes 'seed' (mean-of-seeds is meaningless) and the all-NaN 'step' column (mean/std -> NaN, std filled to 0). Currently harmless because m()/s() only read named accuracy/forgetting columns, but it is sloppy and could surface if a numeric column is later treated as a metric. requirements.txt pins pandas==3.0.2; pandas 3.0 changed several reduction/string-dtype defaults, so this reduction path (and ddof=1 single-row -> NaN -> fillna(0) for single-seed) should be re-validated on the pinned version. Low confidence on any specific 3.0 breakage; flagged for verification.
- **Evidence:** sub.mean(numeric_only=True) / sub.std(numeric_only=True, ddof=1)
- **Fix:** Select the metric columns explicitly before mean/std (drop 'seed'/'step'), and add a smoke test under pandas 3.0.2.
- **Fix at:** `code/generate_figures_3seed.py:88-89`
- _related: FIG-016_

#### `FIG-015` [LOW] Expected seed set {42,123,456} hardcoded in the plotting script
`code/generate_figures_3seed.py:120` · quality · status:live · effort:trivial · confidence:high
- **What:** expected = {42, 123, 456} (120) duplicates the seed policy also hardcoded at m7_trimodal_run.py:516. It is used only to emit a 'not a 3-seed result' warning (121-124), so it is a policy check, not a plotted result number - benign. But it is a second copy of the seed policy that can drift from the training script.
- **Evidence:** expected = {42, 123, 456}
- **Fix:** Optional: source the expected seeds from a shared config/constant rather than re-hardcoding.
- **Fix at:** `code/generate_figures_3seed.py:120`
- _related: FIG-021_

#### `FIG-016` [LOW] Summary-row mask at line 65 is fragile: KeyError if 'step' absent; nullable-boolean mask risk under pandas 3.0 string dtype
`code/generate_figures_3seed.py:65` · correctness · status:live · effort:small · confidence:low
- **What:** summary = df[df['step'].isna() | (df['step'].astype(str).str.strip()=='')].copy() (65). Two fragilities: (1) if the CSV lacks a 'step' column, df['step'] raises KeyError with no friendly message - the dual-schema contract is assumed, not checked. (2) Under pandas 3.0 (pinned 3.0.2), if 'step' is inferred as the new string dtype rather than float, str.strip()=='' yields a nullable BooleanArray; OR-ing/masking that with possible <NA> can raise 'Cannot mask with non-boolean array containing NA'. In practice blank CSV cells usually parse to float NaN (safe path), so this is conditional - confidence low - but worth a guard.
- **Evidence:** df[df['step'].isna() | (df['step'].astype(str).str.strip() == '')]
- **Fix:** Guard for the 'step' column's presence and coerce the mask to plain numpy bool (fillna(False) / .to_numpy(dtype=bool)) before indexing.
- **Fix at:** `code/generate_figures_3seed.py:65`
- _related: FIG-014_

#### `DEP-009` [LOW] Third-party 'wandb' imported but absent from requirements.txt (optional, guarded)
`code/m7_mps.py:84` · dependencies · status:live · effort:trivial · confidence:high
- **What:** `import wandb` runs inside try/except ImportError with a WANDB_AVAILABLE flag in four live scripts and degrades gracefully with a printed hint. Not in the seed list; adds it. No crash, but W&B logging is silently unavailable on any clean install because the dependency is never declared.
- **Evidence:** code/m7_mps.py:83-88 try: import wandb ... except ImportError: WANDB_AVAILABLE=False ; same at code/m7_raw.py:76, code/m7_smoke_test.py:84, code/m7_trimodal_cl.py:84
- **Fix:** Declare wandb as an optional extra (e.g. a logging extra) so it is installable when W&B logging is wanted; leaving it undeclared is the reason logging never activates.
- **Fix at:** `code/requirements.txt:13`, `code/m7_mps.py:84`, `code/m7_raw.py:76`, `code/m7_smoke_test.py:84`, `code/m7_trimodal_cl.py:84`

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

#### `RUN-033` [LOW] events_to_dense: binary occupancy, redundant max_time guard, and divide-by-zero when all timestamps are equal
`code/m7_trimodal_run.py:138-148` · correctness · status:live · effort:trivial · confidence:medium
- **What:** The dense encoding sets dense[t,c]=1.0, so repeated events at the same (t,c) do not accumulate (occupancy, not count). max_time is guarded by 'if len(events)>0 else 1' but the real risk is max_time==0 (all events at t=0), which makes 'events[t]/max_time' divide by zero and yields invalid indices after astype(int). The guard checks length, not max_time.
- **Evidence:** line 142 'max_time = events["t"].max() if len(events) > 0 else 1'; line 143 divides by max_time; line 147 'dense[t, c] = 1.0'
- **Fix:** Guard max_time with 'max(events["t"].max(), 1)', and decide deliberately between occupancy and event-count encoding.
- **Fix at:** `code/m7_trimodal_run.py:138-148`
- _related: RUN-017_

#### `RUN-035` [LOW] Derived metrics survive only in the end-of-seed summary row; an interrupted seed loses forgetting/fwt
`code/m7_trimodal_run.py:549-604` · quality · status:live · effort:small · confidence:medium
- **What:** Only the raw T*_after_T* accuracies are persisted incrementally after each task (549/577/604). All derived metrics (forgetting_T1_T2, forgetting_T1_T3, forgetting_T2_T3, fwt) are computed and written only in the final summary row (612-640). If a seed is interrupted after some tasks, the incremental rows remain but no derived metrics are recoverable, and the checkpoints alone do not contain them.
- **Evidence:** incremental saves 549, 577, 604 store only accuracies; derived metrics computed 613-616 and saved only in the summary row 618-640
- **Fix:** Persist derived metrics incrementally as they become computable (e.g. after T2 write forgetting_T1_T2 and fwt) so an interrupted run is analyzable.
- **Fix at:** `code/m7_trimodal_run.py:549`, `code/m7_trimodal_run.py:577`, `code/m7_trimodal_run.py:604`, `code/m7_trimodal_run.py:612-640`
- _related: RUN-010_

#### `SNN-014` [LOW] torch.mps.manual_seed set in joint scripts but not in entrypoint/hub; dropout=0.2 declared but unused
`code/m7_trimodal_run.py:523-524` · numerics · status:live · effort:trivial · confidence:medium
- **What:** No MPS seed in m7 (L523-524) or provenance (L964-965); the joint scripts do call torch.mps.manual_seed (run_joint_shared_encoder L767-768; run_joint_magnitude_ablation L763-764). dropout=0.2 (m7 L53 / provenance L45) is never used. Currently inert for the entrypoint (init is on CPU before .to(device), no MPS stochastic op), so results are reproducible today - but wiring the declared dropout would introduce forward-time MPS RNG the entrypoint does not seed, silently breaking reproducibility.
- **Evidence:** joint scripts seed MPS; entrypoints do not; dropout unused
- **Fix:** Add torch.mps.manual_seed(seed) in both entrypoints for parity, or remove the unused dropout key so it cannot be wired without review.
- **Fix at:** `code/m7_trimodal_run.py:523-524`, `code/m7_trimodal_run_provenance.py:964-965`
- _related: SNN-015_

#### `SNN-015` [LOW] Python random never seeded; joint-encoder split policy diverges
`code/m7_trimodal_run.py:523-524` · numerics · status:live · effort:trivial · confidence:high
- **What:** Python's random is never seeded (m7 L523-524; provenance L964-965) - inert only because random is not imported/used, but a gap the moment any stdlib-random call is added. Separately run_joint_shared_encoder.py pins its train/val split with manual_seed(seed) (L435) so the split moves with seed - divergent from the M7 policy of manual_seed(42) (defensible for that standalone experiment, but worth stating).
- **Evidence:** no random.seed; joint split manual_seed(seed) L435
- **Fix:** Add random.seed(seed) alongside torch/numpy seeds; document the joint-encoder split policy.
- **Fix at:** `code/m7_trimodal_run.py:523-524`, `code/run_joint_shared_encoder.py:435`
- _related: SNN-014_

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

#### `PROV-021` [LOW] EmbeddingReplayBuffer default device picks mps/cpu only, ignoring CUDA
`code/m7_trimodal_run_provenance.py:418-423` · correctness · status:live · effort:trivial · confidence:medium
- **What:** When device is None the buffer defaults to mps if available else cpu (419-420), never cuda. run_one_seed passes an explicit device (975) so the default is not hit there, but any of the 19 importers that construct EmbeddingReplayBuffer without a device on a CUDA host would place replay embeddings on CPU while the model is on CUDA, causing a device-mismatch error in forward_from_embeddings.
- **Evidence:** device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') (420); explicit device passed at 975.
- **Fix:** Mirror get_device precedence (mps -> cuda -> cpu) in the buffer default, or make device a required argument.
- **Fix at:** `code/m7_trimodal_run_provenance.py:418-423`

#### `PROV-023` [LOW] N-MNIST collate silently crops each sample to the first 25 time-windows, discarding most of the temporal window
`code/m7_trimodal_run_provenance.py:113-116` · research-integrity · status:live · effort:small · confidence:medium
- **What:** nmnist_collate_fn truncates any sample with more than 25 frames to e[:25] (113-114). With the N-MNIST ToFrame transform (time_window=1000 over ~300 ms recordings) samples contain far more than 25 windows, so only the first ~25 ms of temporal information is used and the rest is dropped without any log. DVS collate does the same (133-135). This is a modeling choice inherited from the orphan, but it materially affects the temporal-cost interpretation and should be surfaced when reporting accuracy with its timestep count.
- **Evidence:** if e.shape[0] > 25: e = e[:25] (113-114); dvs analogous (133-135).
- **Fix:** Document the 25-frame crop (and its effect on effective temporal coverage) where the timestep count is reported; if fuller temporal coverage is intended, subsample or bin across the whole window instead of taking the head.
- **Fix at:** `code/m7_trimodal_run_provenance.py:107-142`
- _related: PROV-018_

#### `PROV-024` [LOW] _events_to_dense divides by max_time which can be zero, producing NaN time indices
`code/m7_trimodal_run_provenance.py:1200-1210` · correctness · status:live · effort:trivial · confidence:medium
- **What:** In _events_to_dense, time_indices = (events['t'] / max_time * (time_bins-1)).astype(int) (1205) where max_time = events['t'].max() (1204). If all events share t=0 (or a single event at t=0), max_time is 0 and the division yields inf/NaN, and astype(int) on NaN is undefined behaviour. The len(events)>0 guard (1203) does not cover the all-zero-timestamp case, and the inner 'if len(events)>0 else 1' (1204) is redundant since it is already inside the len>0 block.
- **Evidence:** max_time = events['t'].max() if len(events) > 0 else 1 (1204); time_indices = (events['t'] / max_time * (time_bins - 1)).astype(int) (1205).
- **Fix:** Guard with max_time = max(events['t'].max(), 1) (or skip the sample) so the divisor is never zero; drop the redundant inner conditional.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1200-1210`

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

#### `CLE-015` [LOW] Magic gate-uniformity threshold 1e-6
`code/phase1_sanity_check.py:89` · numerics · status:live · effort:trivial · confidence:high
- **What:** all_identical is decided by (gates.std(dim=0) < 1e-6).all() with an unnamed, undocumented 1e-6 tolerance. The pass/fail of the gate-sanity check hinges on this constant; it should be a named, justified tolerance.
- **Evidence:** Line 89 `"all_identical": bool((gates.std(dim=0) < 1e-6).all()),`.
- **Fix:** Promote 1e-6 to a named constant with a one-line rationale (e.g. GATE_STD_EPS).
- **Fix at:** `code/phase1_sanity_check.py:89`
- _related: CLE-027_

#### `CLE-016` [LOW] T1 accuracy reported without spike-rate/sparsity or timestep cost
`code/phase1_sanity_check.py:45-56` · research-integrity · status:live · effort:small · confidence:medium
- **What:** eval_t1 (45-56) computes and the script prints T1 accuracy at post-T1/T2/T3 (130-132, 183-185) with no accompanying spike rate, sparsity, or timestep count -- the 'accuracy never travels alone' rule. Severity is low because this is a stdout diagnostic, not a persisted result or paper number, but the pattern should not be copied into result-writing code.
- **Evidence:** Lines 130-132 print `{acc:.2f}%` with no cost metric; eval_t1 returns only accuracy.
- **Fix:** If these numbers are ever surfaced beyond debugging, report mean spike rate and time_steps (25) alongside; otherwise label clearly as a debug-only accuracy.
- **Fix at:** `code/phase1_sanity_check.py:45-56`, `code/phase1_sanity_check.py:130-132`
- _related: CLE-024_

#### `CLE-017` [LOW] Unused import: dataclasses
`code/run_cl_confirmation_fixed.py:10` · dead-code · status:live · effort:trivial · confidence:high
- **What:** `import dataclasses` (10) is never used (grep: dataclasses. count 0). Same unused import appears in run_cl_rehearsal.py (see CLE-026).
- **Evidence:** Line 10; no dataclasses.* reference in the file.
- **Fix:** Remove the import.
- **Fix at:** `code/run_cl_confirmation_fixed.py:10`
- _related: CLE-026_

#### `CLE-018` [LOW] _is_complete_result treats a header-only CSV as complete (false skip)
`code/run_cl_confirmation_fixed.py:53-60` · correctness · status:live · effort:trivial · confidence:medium
- **What:** _is_complete_result checks only that all EXPECTED_COLUMNS are present as columns (58), not that any row exists. A CSV written with a header but zero data rows (or a truncated/failed prior write that still has the header) satisfies the check, so run_confirmation_cell prints 'SKIP' (186) and returns {} -- the cell is silently skipped despite having produced no result.
- **Evidence:** Line 57-58 `df = pd.read_csv(csv_path); return all(col in df.columns ...)`; line 185-187 skip logic.
- **Fix:** Also require len(df) >= 1 (and ideally that the single expected row's key fields are non-null) before treating the result as complete.
- **Fix at:** `code/run_cl_confirmation_fixed.py:53-60`

#### `CLE-019` [LOW] cfg.pin_memory and cfg.num_workers are ignored by the DataLoaders
`code/run_cl_confirmation_fixed.py:88-95` · quality · status:live · effort:small · confidence:high
- **What:** Every DataLoader hardcodes num_workers=0 and omits pin_memory, so cfg.pin_memory=True (371) and cfg.num_workers=0 (371) are never actually consumed. num_workers=0 is correct on macOS/MPS, but silently ignoring config fields means the saved config.json advertises settings the run did not use (pin_memory=True in particular). Same in run_cl_rehearsal.py.
- **Evidence:** Loaders at 88-95, 120-127, 141-148 pass num_workers=0 literally and no pin_memory; grep pin_memory=cfg count 0.
- **Fix:** Either pass cfg.pin_memory (and keep num_workers=0 with a comment on the macOS constraint) or drop the unused fields from the config so provenance matches behaviour.
- **Fix at:** `code/run_cl_confirmation_fixed.py:88-95`, `code/run_cl_confirmation_fixed.py:371`
- _related: CLE-020_

#### `CLE-022` [LOW] forgetting_T2_T3 and total_forgetting_T2_T3 are computed identically (always equal)
`code/run_cl_confirmation_fixed.py:328-330` · duplication · status:live · effort:trivial · confidence:high
- **What:** forgetting_t2_t3 = acc_t2_after - acc_t2_after_t3 (328) and total_forgetting_t2_t3 = acc_t2_after - acc_t2_after_t3 (330) are the same expression, so the two CSV columns are always byte-identical -- redundant, and asymmetric with the T1 pair where the two columns genuinely differ (327 vs 329). Either total_forgetting_T2_T3 was meant to be a different quantity, or one column is superfluous. Same in run_cl_rehearsal.py:337/339.
- **Evidence:** Lines 328 and 330 are identical expressions.
- **Fix:** Drop the redundant column or, if a distinct quantity was intended (e.g. an incremental T2 measure), compute it correctly. Mirror in run_cl_rehearsal.py.
- **Fix at:** `code/run_cl_confirmation_fixed.py:328-330`, `code/run_cl_rehearsal.py:337-339`
- _related: CLE-021_

#### `CLE-023` [LOW] Dead else branch in _events_to_dense max_time guard
`code/run_cl_confirmation_fixed.py:105` · dead-code · status:live · effort:trivial · confidence:high
- **What:** `max_time = events["t"].max() if len(events) > 0 else 1` (105) sits inside `if len(events) > 0:` (104), so the `else 1` is unreachable. Cosmetic but confusing, and copy-pasted verbatim into 4 files (see CLE-031).
- **Evidence:** Line 104 guard, line 105 redundant ternary.
- **Fix:** Drop the redundant ternary: inside the guard, `max_time = events["t"].max()`.
- **Fix at:** `code/run_cl_confirmation_fixed.py:105`
- _related: CLE-031_

#### `CLE-032` [LOW] ExperimentConfig literal block triplicated across the run/eval scripts
`code/run_cl_confirmation_fixed.py:364-377` · duplication · status:live · effort:medium · confidence:high
- **What:** The same ~14-line ExperimentConfig(...) literal is declared three times: run_cl_confirmation_fixed.py (364-377), run_cl_rehearsal.py (377-390), and run_rehearsal_transfer_eval_only.py (22-35). They must stay in lockstep for the eval scripts to match the runs, but nothing enforces it (see the replay_weight drift, CLE-003). 
- **Evidence:** Three near-identical ExperimentConfig literal blocks at the cited locations.
- **Fix:** Define the base config once (a factory or a JSON that all three load) and have the eval scripts read the config saved with the checkpoint rather than re-declaring it.
- **Fix at:** `code/run_cl_confirmation_fixed.py:364-377`, `code/run_cl_rehearsal.py:377-390`, `code/run_rehearsal_transfer_eval_only.py:22-35`
- _related: CLE-005, CLE-030_

#### `CLE-026` [LOW] Unused import: dataclasses
`code/run_cl_rehearsal.py:15` · dead-code · status:live · effort:trivial · confidence:high
- **What:** `import dataclasses` (15) is never used (grep: dataclasses. count 0). Same as run_cl_confirmation_fixed.py (CLE-017).
- **Evidence:** Line 15; no dataclasses.* reference.
- **Fix:** Remove the import.
- **Fix at:** `code/run_cl_rehearsal.py:15`
- _related: CLE-017_

#### `CLE-027` [LOW] Magic numbers: raw buffer size 200 (vs cfg 50), max_batches 100, drift eps 1e-6
`code/run_cl_rehearsal.py:178` · numerics · status:live · effort:small · confidence:medium
- **What:** RawInputReplayBuffer(samples_per_class=200) (178) hardcodes 200 while cfg.replay_samples_per_class=50 (386) governs the embedding buffer -- two different, unexplained capacities. raw_buffer.populate(..., max_batches=100) (214,275) hardcodes 100. The backbone-drift sanity uses total_l2 < 1e-6 (330) as a magic threshold. None are named or justified; they affect what rehearsal actually sees.
- **Evidence:** Line 178 samples_per_class=200; lines 214,275 max_batches=100; line 330 `if total_l2 < 1e-6`.
- **Fix:** Promote to named, documented constants or cfg fields; reconcile the 200 vs cfg's 50 so the rehearsal capacity is intentional and recorded.
- **Fix at:** `code/run_cl_rehearsal.py:178`, `code/run_cl_rehearsal.py:214`, `code/run_cl_rehearsal.py:275`, `code/run_cl_rehearsal.py:330`
- _related: CLE-015_

#### `CLE-028` [LOW] Backbone-drift check indexes sd_t3[k] by sd_t1's keys without a membership check
`code/run_cl_rehearsal.py:324-328` · correctness · status:live · effort:trivial · confidence:low
- **What:** The drift loop iterates keys of sd_t1 and computes (sd_t1[k] - sd_t3[k]).norm() (326-327) with no check that k exists in sd_t3. If the two checkpoints have different key sets (e.g. arch change between saves), this raises KeyError. Low likelihood because both come from the same model class, hence low confidence, but the sibling phase1_sanity_check.compare_state_dicts (65-66) does guard for the missing key -- an inconsistency worth aligning.
- **Evidence:** Lines 324-328; contrast phase1_sanity_check.py:65-66 which handles `k not in sd2`.
- **Fix:** Skip or record keys missing from sd_t3 (mirror compare_state_dicts) before subtracting.
- **Fix at:** `code/run_cl_rehearsal.py:324-328`

#### `CMP-010` [LOW] Eval composites use TRAIN-set exemplars and differ across models
`code/run_compositional_crossmodal.py:825-849` · correctness · status:live · effort:medium · confidence:medium
- **What:** CompositeBuilder samples digit exemplars from cached['nmnist_train']/['shd_train'], and eval reuses the same builder whose RNG (seed+7) has been advanced by a model-specific amount of training (compositional 2 phases, holistic 1, untrained 0). So the four models are evaluated on different exemplar draws, cross-model comparison is not on identical inputs, and no unseen-EXEMPLAR generalization is tested (only unseen combinations).
- **Evidence:** builder = CompositeBuilder(cached, np.random.default_rng(seed + 7)); ... ev = extract_2digit(model, builder, device, ...)
- **Fix:** Use a separate fixed-RNG builder for evaluation, independent of training draws, and snapshot the eval spec list so every model sees identical inputs; consider held-out exemplars for a stronger test.
- **Fix at:** `code/run_compositional_crossmodal.py:830`, `code/run_compositional_crossmodal.py:848-849`, `code/run_compositional_crossmodal.py:187-198`

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

#### `CMP-015` [LOW] Verdict thresholds are hardcoded magic numbers, only partly pre-registered
`code/run_compositional_crossmodal.py:1016-1027` · research-integrity · status:live · effort:small · confidence:medium
- **What:** The auto-verdict uses order>60, wn_held>2.0, asym<5, set_v2a>set_perm+1.0 as decision cutoffs embedded in code. report.txt lists the metrics as 'PRE-REGISTERED' but not these numeric thresholds. Docstring chance values (37-42) are analytic claims and 'Procrustes ~0%' (8) is an unsourced prior result (RULE 1: no CSV in repo).
- **Evidence:** if clears_null and order > 60 and wn_held > 2.0 and asym < 5 and beats_holistic: verdict.append('FACTORIZES: ...')
- **Fix:** Promote thresholds to named constants and record them in the pre-registration block and the config sidecar; mark 'Procrustes ~0%' as an unsourced prior observation.
- **Fix at:** `code/run_compositional_crossmodal.py:1016-1027`, `code/run_compositional_crossmodal.py:37-42`

#### `ALGN-010` [LOW] Unused import List
`code/run_correspondence_anchor_v2.py:23` · quality · status:live · effort:trivial · confidence:high
- **What:** typing.List is imported at line 23 (from typing import Dict, List, Tuple) but never used in the file.
- **Evidence:** line 23 from typing import Dict, List, Tuple; only reference to List is the import itself
- **Fix:** Drop List from the typing import.
- **Fix at:** `code/run_correspondence_anchor_v2.py:23`

#### `ALGN-011` [LOW] _events_to_dense: dead else-branch and divide-by-zero when all events at t=0
`code/run_correspondence_anchor_v2.py:93-103` · correctness · status:live · effort:trivial · confidence:medium
- **What:** Inside if len(events)>0, max_time = events['t'].max() if len(events)>0 else 1 has a dead else (already guarded). If every event has t==0 (max_time==0), time_indices = (events['t']/max_time*(time_bins-1)).astype(int) divides by zero -> inf/nan -> astype(int) is undefined behaviour. Copy exists in run_ordinal_anchor.py:92 and run_digit_only_alignment.py:122.
- **Evidence:** lines 97-98 max_time = events['t'].max() if len(events)>0 else 1; time_indices=(events['t']/max_time*(time_bins-1)).astype(int)
- **Fix:** Guard max_time with max(int(events['t'].max()),1); remove the redundant conditional.
- **Fix at:** `code/run_correspondence_anchor_v2.py:97-98`, `code/run_ordinal_anchor.py:97-98`, `code/run_digit_only_alignment.py:127-128`
- _related: ALGN-008_

#### `ALGN-012` [LOW] torch.load of on-disk paths without explicit weights_only
`code/run_correspondence_anchor_v2.py:155` · safety · status:live · effort:trivial · confidence:medium
- **What:** torch.load(cache_path) (155) and torch.load(CKPT_SOURCE) (533) load pickled files without weights_only=. Under torch==2.11 the default is weights_only=True (mitigating arbitrary-code risk), and the payloads are state-dict/tensor dicts so they load fine; flag for awareness and forward-compat. Same pattern in run_ordinal_anchor.py:410 and run_cross_modal_alignment_study.py:214,278,325.
- **Evidence:** line 155 torch.load(cache_path, map_location='cpu'); line 533 torch.load(CKPT_SOURCE, map_location=str(device)); no weights_only anywhere in cluster
- **Fix:** Pass weights_only=True explicitly for state-dict loads to be robust across torch versions.
- **Fix at:** `code/run_correspondence_anchor_v2.py:155`, `code/run_correspondence_anchor_v2.py:533`, `code/run_ordinal_anchor.py:410`, `code/run_cross_modal_alignment_study.py:214`, `code/run_cross_modal_alignment_study.py:278`, `code/run_cross_modal_alignment_study.py:325`

#### `ALGN-025` [LOW] master_config.json mislabels conditions as no_replay/replay (copy-paste from CL study)
`code/run_cross_modal_alignment_study.py:442` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** The module is an A/B/C alignment study (docstring 1-27; aggregation iterates A_no_align/B_full_512/C_split_128 at line 102), but base_cfg sets conditions=['no_replay','replay'] (line 442) - leftover from the replay CL study. base_cfg.to_json(master_config.json) (528) therefore persists a conditions field that contradicts what the study actually ran, poisoning provenance for anyone trusting the JSON.
- **Evidence:** line 442 conditions=['no_replay','replay']; line 102 for cond in ['A_no_align','B_full_512','C_split_128']; line 528 base_cfg.to_json(results_dir/'master_config.json')
- **Fix:** Set conditions=['A_no_align','B_full_512','C_split_128'] to match the actual run.
- **Fix at:** `code/run_cross_modal_alignment_study.py:442`

#### `ALGN-026` [LOW] Unused imports copy, os, Dict
`code/run_cross_modal_alignment_study.py:32-38` · quality · status:live · effort:trivial · confidence:high
- **What:** import copy (32) and import os (36) have zero real attribute uses (copy. and os. never called); typing.Dict (38) is imported but unused (only lowercase dict is used in annotations). Also note seed set here is [42,123,456,789,101112] (line 413) where 101112 looks like a typo divergence from the 1011 used in run_ordinal_anchor and run_digit_only.
- **Evidence:** line 32 import copy; line 36 import os; line 38 from typing import Dict, List (Dict unused); line 413 seeds=[42,123,456,789,101112]
- **Fix:** Remove unused copy/os imports and Dict; reconcile the 101112 vs 1011 seed if intended to match the other studies.
- **Fix at:** `code/run_cross_modal_alignment_study.py:32`, `code/run_cross_modal_alignment_study.py:36`, `code/run_cross_modal_alignment_study.py:38`, `code/run_cross_modal_alignment_study.py:413`

#### `ALGN-031` [LOW] aggregate_digit_results silently drops cells missing any EXPECTED_COLUMN
`code/run_digit_only_alignment.py:365-374` · correctness · status:live · effort:trivial · confidence:medium
- **What:** aggregate_digit_results skips any per-cell CSV that lacks all EXPECTED_COLUMNS (368-374, printing 'Skipping ... incomplete columns'). A cell written by an older schema (or partially, though atomic writes make partials unlikely) is silently excluded from the summary and Wilcoxon, so n can drop without the headline changing. Note the positive: because each cell is one atomically-overwritten file (_atomic_write_csv, 197-201), reruns do NOT accumulate duplicate rows here - unlike run_cross_modal_alignment_study (ALGN-022).
- **Evidence:** lines 368-374 if all(col in df.columns for col in EXPECTED_COLUMNS): dfs.append(df) else print('Skipping ... incomplete columns'); EXPECTED_COLUMNS defined 73-82; _atomic_write_csv 197-201
- **Fix:** Log skipped cells at WARN with the missing column names and surface the final n per condition so silent shrinkage is visible.
- **Fix at:** `code/run_digit_only_alignment.py:364-378`
- _related: ALGN-022_

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

#### `JNT-011` [LOW] _events_to_dense_shd duplicates SHD densification present in three other files
`code/run_joint_magnitude_ablation.py:72-82` · duplication · status:live · effort:small · confidence:high
- **What:** The SHD event-to-dense conversion exists in at least four near-identical copies across the code tree, with subtly different output shapes.
- **Evidence:** _events_to_dense_shd (72-82) returns (T,C) 2-D; run_joint_shared_encoder.py:258-268 returns (T,1,1,C); m7_trimodal_run_provenance.py:1200-1210 and run_correspondence_anchor_v2.py:93-103 return (T,1,1,C),label. Same algorithm, four sites.
- **Fix:** Centralize one parameterized SHD densification helper (bins/channels/shape) in a shared module and import it everywhere.
- **Fix at:** `code/run_joint_magnitude_ablation.py:72-82`, `code/run_joint_shared_encoder.py:258-268`, `code/m7_trimodal_run_provenance.py:1200-1210`
- _related: JNT-027_

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

#### `JNT-014` [LOW] Population std (ddof=0) reported as '± std' over n<=5, including the non-control shuffle
`code/run_joint_magnitude_ablation.py:948-951` · research-integrity · status:live · effort:trivial · confidence:medium
- **What:** Aggregates use numpy population std labeled with a plus/minus, which understates the sample spread at small n, and the aggregated set mixes in the mislabeled shuffle arm where applicable.
- **Evidence:** np.std (ddof=0) at 948-951, 959-962 and report 978-991 is printed as '± std'; n is 5 seeds (or fewer after JNT-003). Population SD < sample SD at small n.
- **Fix:** Use ddof=1 for a sample standard deviation or explicitly label the figure as population SD; keep n visible (already printed).
- **Fix at:** `code/run_joint_magnitude_ablation.py:948-951`
- _related: JNT-001_

#### `JNT-025` [LOW] run_one_seed is a fourth same-named per-seed runner with a different signature
`code/run_joint_shared_encoder.py:764` · naming · status:live · effort:trivial · confidence:high
- **What:** Multiple modules define run_one_seed with mutually incompatible signatures, and the ablation calls its equivalent run_one_arm, making cross-referencing error-prone.
- **Evidence:** run_one_seed(seed, cfg, shuffled_labels) here (764); m7_trimodal_run_provenance.run_one_seed(cfg, data_loaders, results_dir, ckpt_dir) (962); run_correspondence_anchor_v2.run_one_seed(seed, cfg, loaders, cached, device, head_type) (524); run_joint_magnitude_ablation.run_one_arm (756).
- **Fix:** Give experiment-specific runners distinct names (e.g. run_one_seed_shared_encoder) or converge on a shared interface.
- **Fix at:** `code/run_joint_shared_encoder.py:764`

#### `JNT-026` [LOW] Own nmnist_collate_fn/shd_collate_fn duplicate the provenance collates
`code/run_joint_shared_encoder.py:241-273` · duplication · status:live · effort:small · confidence:high
- **What:** This file re-implements the collate functions instead of importing them, unlike the magnitude ablation which imports the provenance versions.
- **Evidence:** nmnist_collate_fn (241-253) and shd_collate_fn (271-273) duplicate m7_trimodal_run_provenance.py:107-123. This file imports only CrossModalSupConLoss and spike_grad from provenance (47-50); run_joint_magnitude_ablation.py:49-55 imports the provenance collates.
- **Fix:** Import the provenance collate functions instead of re-defining them (as the magnitude ablation does).
- **Fix at:** `code/run_joint_shared_encoder.py:241-273`, `code/m7_trimodal_run_provenance.py:107-123`
- _related: JNT-011_

#### `JNT-027` [LOW] _events_to_dense is another SHD densification copy with a third shape-normalization site
`code/run_joint_shared_encoder.py:258-268` · duplication · status:live · effort:small · confidence:high
- **What:** Another copy of the SHD densifier, producing a (T,1,1,C) tensor that is then squeezed in two more places for one logical tensor.
- **Evidence:** _events_to_dense (258-268) returns (100,1,1,700); shd_collate_fn squeezes to (B,100,700) (273); PairedDigitDatasetFiltered.__getitem__ squeezes again (226-227). See JNT-011 for the other copies.
- **Fix:** Centralize densification and settle on one output shape to remove the repeated squeezes.
- **Fix at:** `code/run_joint_shared_encoder.py:258-268`, `code/run_joint_shared_encoder.py:226-227`, `code/run_joint_shared_encoder.py:273`
- _related: JNT-011_

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

#### `JNT-030` [LOW] best_state can be undefined -> NameError if first-epoch val loss is NaN or epochs==0
`code/run_joint_shared_encoder.py:805-839` · correctness · status:live · effort:trivial · confidence:medium
- **What:** The best-model restore assumes the improvement branch ran at least once; a NaN first-epoch validation loss (possible in SNN training) or epochs==0 leaves best_state unbound.
- **Evidence:** best_val_loss=inf (805); best_state assigned only inside 'if val_stats[loss] < best_val_loss' (828-831); model.load_state_dict(best_state) at 839. NaN < inf is False, so a NaN first epoch never sets best_state.
- **Fix:** Initialize best_state = copy.deepcopy(model.state_dict()) before the training loop.
- **Fix at:** `code/run_joint_shared_encoder.py:805-839`

#### `JNT-031` [LOW] Shuffled-control permutation index built on CPU while labels are on the MPS device
`code/run_joint_shared_encoder.py:512-514` · correctness · status:live · effort:trivial · confidence:low
- **What:** The label-permutation index is created on the default (CPU) device and used to index a device tensor, which can raise on MPS/CUDA depending on the torch version and would crash the unguarded shuffled-control run.
- **Evidence:** perm = torch.randperm(all_labels.size(0)) (513, and 560 in validate) with all_labels on DEVICE (mps); all_labels[perm] indexes across devices. The shuffled control is called unguarded in main (930), so a device error there aborts before summaries are written. torch is pinned at 2.11.0.
- **Fix:** Create the permutation on the tensor's device: torch.randperm(n, device=all_labels.device); optionally guard the shuffled-control call.
- **Fix at:** `code/run_joint_shared_encoder.py:512-514`, `code/run_joint_shared_encoder.py:559-561`
- _related: JNT-001_

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

#### `ALGN-020` [LOW] _spearman uses no-ties rho formula on tie-averaged ranks with normal-approx p
`code/run_ordinal_anchor.py:130-144` · numerics · status:live · effort:small · confidence:medium
- **What:** _spearman computes rho via the shortcut 1 - 6*sum(d^2)/(n*(n^2-1)) (line 136), which is only exact when there are no ties, yet it feeds tie-averaged ranks from _rankdata; with ties the rho is slightly biased. The p-value is a normal approximation to the t-distribution (lines 141-143). For n=45 pairs this is roughly OK but not exact; if it enters the paper it should be labeled approximate.
- **Evidence:** line 136 rho = 1.0 - (6.0*np.sum(d*d))/(n*(n*n-1.0)); lines 141-143 t-stat then normal-CDF two-sided p
- **Fix:** Use the Pearson-of-ranks definition (tie-consistent) for rho, or scipy.stats.spearmanr; mark p as approximate if kept.
- **Fix at:** `code/run_ordinal_anchor.py:130-144`

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

#### `RA-012` [LOW] replay_weight written as 1.0 while the run is configured no-replay
`code/run_rehearsal_transfer_eval_only.py:117` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** The CSV row sets replay_weight=1.0 under condition='C_rehearsal', but the reconstructed config at lines 31-34 has use_replay=False and replay_weight=0.3. The provenance column contradicts the config it claims to reproduce.
- **Evidence:** line 117 vs cfg 31-34; verdict CONTRADICTED
- **Fix:** Write cfg.replay_weight (and cfg.use_replay) rather than a literal, or correct the condition label.
- **Fix at:** `code/run_rehearsal_transfer_eval_only.py:117`, `code/run_rehearsal_transfer_eval_only.py:31-34`
- _confirmed by CLE-003; related: RA-001_

#### `TST-011` [LOW] Local TASKS is a hand-copied duplicate of the canonical module-level TASKS (silent-drift risk)
`code/test_cross_modal_alignment.py:61-65` · duplication · status:live · effort:trivial · confidence:high
- **What:** The test defines its own TASKS dict (lines 61-65) instead of importing the canonical TASKS from m7_trimodal_run_provenance (lines 955-959). The two are byte-for-byte equivalent TODAY (nmnist:10/visual, shd:20/audio, dvs:11/visual_motion), so there is no current bug, but it is a duplicated source of truth: if the real TASKS schema or class counts change, this test keeps building the model against a stale task spec and will not detect the drift. (Note: it is not literal Python shadowing, since this file does not import TASKS — it is a divergent copy.)
- **Evidence:** code/test_cross_modal_alignment.py:61-65 local TASKS; identical structure to m7_trimodal_run_provenance.py:955-959.
- **Fix:** Import TASKS from m7_trimodal_run_provenance and use it, or build the test's TASKS from that source, so there is a single source of truth.
- **Fix at:** `code/test_cross_modal_alignment.py:61-65`
- _related: TST-020_

#### `TST-012` [LOW] Vacuous assertions that are true by construction and validate nothing
`code/test_cross_modal_alignment.py:104-114` · quality · status:live · effort:small · confidence:high
- **What:** Two assertions cannot meaningfully fail. Line 105 `assert pen.item() >= 0` — orthogonal_penalty returns cross.pow(2).sum() (m7_trimodal_run_provenance.py:218-219), a sum of squares, which is always >= 0. Line 114 `assert loss_xmscl.item() > 0` — a SupCon loss is a mean of non-negative -log_prob terms and is essentially always positive. These checks only confirm the calls return a scalar; they give no evidence the quantities are correct. The scalar-ness checks (lines 104, 113 `ndim == 0`) are fine but weak.
- **Evidence:** code/test_cross_modal_alignment.py:105 `assert pen.item() >= 0`; orthogonal_penalty sum-of-squares at m7_trimodal_run_provenance.py:218-219; line 114 `assert loss_xmscl.item() > 0`.
- **Fix:** Replace tautological bounds with checks that could fail: e.g. penalty of orthogonalized vs correlated inputs differs as expected; SupCon loss decreases when identical-label embeddings are made more similar; or compare against a known reference value.
- **Fix at:** `code/test_cross_modal_alignment.py:104-114`
- _related: TST-013_

#### `TST-013` [LOW] Orthogonal-penalty test uses a raw slice, not the model's learned proj_private
`code/test_cross_modal_alignment.py:102-106` · correctness · status:live · effort:small · confidence:medium
- **What:** private_vis is computed by slicing full_vis[:, cfg.align_shared_dim:] (line 102), i.e. the tail dimensions of the full feature, whereas the model's actual private subspace is a learned nn.Linear projection self.proj_private (m7_trimodal_run_provenance.py:339). So the penalty exercised here is on a different quantity than the one the model uses in training, and (given the vacuous >=0 assertion, TST-012) the test would not catch a bug in proj_private wiring. It validates orthogonal_penalty's arithmetic but not the model's shared/private split.
- **Evidence:** code/test_cross_modal_alignment.py:102 `private_vis = full_vis[:, cfg.align_shared_dim:]`; model's proj_private is a learned Linear at m7_trimodal_run_provenance.py:339.
- **Fix:** Obtain private from the model's proj_private path (or from the model's own split helper) so the test reflects what training actually uses, and add a meaningful assertion on it.
- **Fix at:** `code/test_cross_modal_alignment.py:102-106`
- _related: TST-012_

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

#### `TST-020` [LOW] Inline ExperimentConfig duplicates the full production hyperparameter set
`code/test_ordinal_anchor_smoke.py:23-56` · duplication · status:live · effort:small · confidence:medium
- **What:** The test hand-writes a 34-field ExperimentConfig (lines 23-56) restating essentially all M7 production hyperparameters (num_patterns=100, time_steps=25, beta=0.9, hidden_dim=512, shared_dim=512, contrastive_temperature=0.07, replay_weight=0.3, etc.). Most are already the dataclass defaults (m7_trimodal_run_provenance.py:34-74), so this is redundant, and it is a second place these production values are pinned — if the real defaults change, this smoke config silently diverges. A smoke test should use small/cheap values (small dims, few epochs), not mirror the full production config.
- **Evidence:** code/test_ordinal_anchor_smoke.py:23-56 inline ExperimentConfig with production values; same fields/defaults defined in m7_trimodal_run_provenance.py:34-74.
- **Fix:** Only override the fields the smoke test needs (and make them small); rely on dataclass defaults for the rest. Avoid re-pinning production hyperparameters in a test.
- **Fix at:** `code/test_ordinal_anchor_smoke.py:23-56`
- _related: TST-011_

#### `TST-005` [LOW] Hardcoded accuracy-like literals passed into print_task_sanity
`code/test_provenance_smoke.py:103-106` · research-integrity · status:live · effort:trivial · confidence:medium
- **What:** The print_task_sanity call is fed fabricated numbers: after_training_acc=15.0 (line 103), learning_gain=15.0 - acc_shd_zero (line 104), random_init_baseline_acc=5.2 (line 105). These are placeholder literals, not measured quantities. Because this is a smoke test and the values are only printed (never written to a results CSV), the impact is low, but hardcoded accuracy-shaped numbers (15.0, 5.2) in a file whose output is labeled 'SANITY' are exactly the pattern RULE 1 warns against, and a reader skimming stdout could mistake them for measurements.
- **Evidence:** code/test_provenance_smoke.py:103 after_training_acc=15.0; line 104 learning_gain=15.0 - acc_shd_zero; line 105 random_init_baseline_acc=5.2.
- **Fix:** Either drive print_task_sanity from actually-computed values, or clearly comment these as dummy placeholders in the smoke path so they cannot be mistaken for results.
- **Fix at:** `code/test_provenance_smoke.py:103-106`

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

#### `TST-008` [LOW] Zero-shot section is not seeded, so its outcome is not reproducible by design
`code/test_provenance_smoke.py:66-93` · numerics · status:live · effort:trivial · confidence:medium
- **What:** The test never calls torch.manual_seed / np.random.seed in its own body. The model constructed at line 68 and the training at lines 73-78 consume whatever global RNG state is left over from the preceding random_init_baseline loop (which reseeds internally to cfg.seed+1000+s, lines 58-62). So the zero-shot accuracy that feeds the near-chance check (lines 81-93) depends on incidental RNG ordering, not on a declared seed. Combined with the tight/under-powered tolerances, pass/warn behavior is not reproducible across code changes that alter earlier RNG consumption.
- **Evidence:** No torch.manual_seed/np.random.seed in code/test_provenance_smoke.py; model init at line 68 follows random_init_baseline reseeding at lines 58-62; zero-shot check at lines 89-93.
- **Fix:** Set torch.manual_seed(cfg.seed) and np.random.seed(cfg.seed) at the start of the zero-shot block (or the test function) so the result is deterministic and reproducible.
- **Fix at:** `code/test_provenance_smoke.py:66-70`
- _related: TST-003, TST-004_


---
## INFO (20)

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

#### `RA-014` [INFO] Figure 4 memory derived from CONFIG and explicitly labelled (compliant pattern)
`code/generate_figures_3seed.py:281-328` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** The compliant counter-example to RA-006. SAMPLES_PER_CLASS/FEATURE_DIM/BYTES/CLASSES/FRAME_ELEMS are config constants, the arithmetic is printed for audit (304-310), and the figure title reads '(derived from CONFIG, not measured)'. It asserts a footprint from configuration, not a measurement, and says so.
- **Evidence:** lines 281-328, title at 323; verdict DISCLOSED (correct pattern)
- **Fix:** No fix. Keep as the reference pattern for how to present a config-derived quantity.
- **Fix at:** `code/generate_figures_3seed.py:281-328`
- _related: RA-006_

#### `DEP-018` [INFO] Seed-listed 'datasets' is NOT a missing dependency (it is tonic.datasets)
`code/m7_mps.py:67` · dependencies · status:live · effort:trivial · confidence:high
- **What:** A prior audit seed listed 'datasets' among undeclared third-party imports. This is a false positive and must not be acted on. 'datasets' appears only as `from tonic import datasets, transforms` -- the datasets submodule of tonic, which IS declared (tonic==1.6.0). The HuggingFace 'datasets' package is never imported anywhere in code/; the bare token 'datasets' otherwise appears only in stale print/`!pip install` notebook cruft. Recorded so a downstream fixer does not incorrectly add the HF datasets package to requirements.
- **Evidence:** from tonic import datasets, transforms at code/m7_mps.py:67, code/m7_nmnist_smoke_test.py:16, code/m7_raw.py:59, code/m7_smoke_test.py:67, code/m7_trimodal_cl.py:67 ; no `import datasets` / `from datasets import` anywhere; tonic==1.6.0 at code/requirements.txt:5
- **Fix:** No action. Do NOT add a 'datasets' package to code/requirements.txt.
- **Fix at:** `code/requirements.txt:5`
- _related: DEP-010_

#### `RUN-034` [INFO] Train/val split generator is fixed at manual_seed(42) independent of the experiment seed
`code/m7_trimodal_run.py:124-126` · numerics · status:live · effort:trivial · confidence:high
- **What:** All three random_split calls use torch.Generator().manual_seed(42), so the train/val partition is identical across experiment seeds 42/123/456 - only model init and DataLoader shuffling vary. This is the documented intent (splits fixed across seeds), recorded here so it is not mistaken for a bug or 'fixed' by wiring in the run seed.
- **Evidence:** lines 126, 154, 172 'generator=torch.Generator().manual_seed(42)'
- **Fix:** No change needed; keep as-is and document that the split is intentionally seed-independent.
- **Fix at:** `code/m7_trimodal_run.py:124-126`, `code/m7_trimodal_run.py:152-154`, `code/m7_trimodal_run.py:169-172`
- _related: RUN-019_

#### `SNN-016` [INFO] VERIFIED-CORRECT: num_workers=0 is respected everywhere (do not 'optimize')
`code/m7_trimodal_run.py:127-132` · numerics · status:live · effort:trivial · confidence:high
- **What:** num_workers=0 is a literal everywhere (m7 L127-132; provenance L1187; confirmation L88; rehearsal L66); run_joint_shared_encoder routes through cfg.num_workers (L440) defaulting 0. This is REQUIRED on macOS (tonic multiprocessing does not survive fork) and is not a performance knob. Recorded so a fixing agent does not raise it.
- **Evidence:** num_workers=0 literals across M7-lineage files
- **Fix:** Do NOT change. Leave num_workers=0 on macOS/MPS.
- **Fix at:** `code/m7_trimodal_run.py:127-132`

#### `SNN-017` [INFO] VERIFIED-CORRECT: reported accuracy uses the TEST split, not validation
`code/m7_trimodal_run.py:545-600` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** evaluate is called on test_loader_* for every reported accuracy (m7 L545/573/600; provenance L1007/1057/1109); best_val is used only for early-stop model selection. RULE 4 (report test, not validation) is satisfied - subject only to the best-vs-last caveat in SNN-009.
- **Evidence:** evaluate(test_loader_*) at the cited lines
- **Fix:** No change needed; keep reporting the test split.
- **Fix at:** `code/m7_trimodal_run.py:545`
- _related: SNN-009_

#### `SNN-018` [INFO] VERIFIED-CORRECT: dataset splits are pinned across seeds with a LOCAL generator
`code/m7_trimodal_run.py:126` · numerics · status:live · effort:trivial · confidence:high
- **What:** Splits use torch.Generator().manual_seed(42) in all M7-lineage files (m7 L126/154/172; provenance L1185/1217/1238; confirmation L86/118/139; rehearsal L64/96/117). Splits do not vary by seed - only init/shuffle do, as intended. This is a LOCAL generator so it does not perturb the global RNG (contrast SNN-001/007 which are about the global RNG driving shuffle).
- **Evidence:** torch.Generator().manual_seed(42) at the split sites
- **Fix:** No change needed; this is the correct split-pinning pattern.
- **Fix at:** `code/m7_trimodal_run.py:126`
- _related: SNN-007_

#### `SNN-019` [INFO] VERIFIED-CORRECT: beta=0.9 applied uniformly to every LIF neuron
`code/m7_trimodal_run.py:238-247` · numerics · status:live · effort:trivial · confidence:high
- **What:** beta=0.9 is applied uniformly to every snn.Leaky (m7 L238-247/283-289; provenance via cfg.beta); it is not a per-layer knob and no code path overrides it - consistent with the SpiNNaker tau_m mapping the docs rely on.
- **Evidence:** snn.Leaky(beta=0.9) uniform
- **Fix:** No change needed; keep beta uniform.
- **Fix at:** `code/m7_trimodal_run.py:238-247`

#### `SNN-020` [INFO] VERIFIED-CORRECT: no silent MPS CPU-fallback env is set
`code/m7_trimodal_run.py:1` · numerics · status:live · effort:trivial · confidence:high
- **What:** PYTORCH_ENABLE_MPS_FALLBACK appears in none of the audited training files; the numpy svd / adaptive_avg_pool2d paths run on CPU by construction, not via hidden fallback. So there is no undisclosed CPU fallback changing timing or numerics.
- **Evidence:** no PYTORCH_ENABLE_MPS_FALLBACK anywhere in the training files
- **Fix:** No change needed; do not set the fallback env without disclosing it.
- **Fix at:** `code/m7_trimodal_run.py:1`

#### `PROV-026` [INFO] Positive: the provenance hub is import-safe (has a __main__ guard and lazy data loading), unlike the orphan
`code/m7_trimodal_run_provenance.py:1378-1379` · safety · status:live · effort:trivial · confidence:high
- **What:** Unlike code/m7_trimodal_run.py (which loads datasets and runs the full 4-5h experiment at module import time with no guard), this file has if __name__=='__main__': main() (1378-1379), and all dataset loading/experiment work is inside build_data_loaders/main() called only from that guard. The only top-level side effects are cheap: spike_grad = surrogate.atan() (148) and the TASKS dict (955-959). This matters precisely because the module is imported by 19 files — importing it does NOT trigger the long run. No action needed; recorded so downstream tooling does not treat it like the orphan.
- **Evidence:** __main__ guard (1378); data loading inside build_data_loaders (1168) called from main (1339); top-level only spike_grad (148) and TASKS (955).
- **Fix:** None required. Preserve the guard and the lazy-loading structure in any refactor.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1378-1379`
- _related: PROV-007_

#### `PROV-027` [INFO] Shared hub contains the SpiNNaker-non-portable ops (LayerNorm, Softmax gate, HGRN multiplicative reset r*h_prev)
`code/m7_trimodal_run_provenance.py:222-255` · architecture · status:live · effort:trivial · confidence:high
- **What:** The model defined here is what the SpiNNaker port must map. It uses nn.LayerNorm inside Hopfield (228) and HGRN (243-245) and in each head (332); a Softmax gate (327); and the HGRN multiplicative reset r * h_prev (254). Per .claude/rules/spinnaker.md and CLAUDE.md, LayerNorm and Softmax are not native to sPyNNaker and the multiplicative reset r*h_prev is flagged as not expressible (risk R3, threatening the novelty claim). This is not a code defect; it is recorded so the port work references the actual hub source rather than the orphan.
- **Evidence:** LayerNorm in Hopfield (228), HGRN (243-245), heads (332); Softmax gate (327); h_tilde uses torch.cat([x, r * h_prev]) (254).
- **Fix:** No code change here. Use these exact lines as the reference when assessing SpiNNaker expressibility; do not assert portability without a board run (mapping != inference != hardware).
- **Fix at:** `code/m7_trimodal_run_provenance.py:222-255`, `code/m7_trimodal_run_provenance.py:326-333`

#### `CMP-016` [INFO] Runtime subprocess call to git for provenance
`code/run_compositional_crossmodal.py:1204-1207` · safety · status:live · effort:trivial · confidence:high
- **What:** main shells out to `git rev-parse HEAD` to stamp the config sidecar. It is benign and wrapped in try/except (falls back to 'unknown'), noted only for the safety inventory (subprocess exec).
- **Evidence:** sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO_ROOT).decode().strip()
- **Fix:** No action required; optionally read the SHA via a library or guard cwd existence.
- **Fix at:** `code/run_compositional_crossmodal.py:1204-1207`

#### `CMP-017` [INFO] Import chain is heavy but import-safe
`code/run_compositional_crossmodal.py:73-95` · dependencies · status:live · effort:medium · confidence:high
- **What:** Importing this file transitively imports run_joint_magnitude_ablation -> run_correspondence_anchor_v2 -> m7_trimodal_run_provenance, plus Dataloaders.paired_digit_loader, tonic and snntorch. All have __main__ guards and none load datasets at import, so importing does not start a run - but it is not free and pulls the whole magnitude/provenance stack. (Audited as text only; do not import to test.) snntorch/tonic/torch/numpy are all present in code/requirements.txt.
- **Evidence:** from run_joint_magnitude_ablation import (...); import snntorch as snn; (magnitude imports tonic + Dataloaders + run_correspondence_anchor_v2)
- **Fix:** No action required; if a lighter import is wanted, split the reused loaders/stems into a data-only module without the anchor_v2 dependency.
- **Fix at:** `code/run_compositional_crossmodal.py:73-95`

#### `ALGN-013` [INFO] Retrieval candidate-set sizes differ across analyses; magic verdict threshold
`code/run_correspondence_anchor_v2.py:419-435` · correctness · status:live · effort:small · confidence:medium
- **What:** compute_held_out_metrics v2a/a2v retrieval ranks among only the 4 held-out classes (chance 0.25, lines 419-435), while run_procrustes ranks among all 10 classes (chance 0.1, all_classes at line 489). The two retrieval numbers are reported side by side in report.txt yet are not comparable. The verdict threshold '0.25 + 0.1' (line 592) is a magic number; with only 4 held-out classes the retrieval statistic is quantised to {0,.25,.5,.75,1} and is high-variance across seeds.
- **Evidence:** lines 419-435 (4-way retrieval); line 489 all_classes = list(train_classes)+list(held_out_classes) (10-way); line 592 held_out['held_out_v2a_retrieval'] > 0.25 + 0.1
- **Fix:** Fix candidate sets to a documented common protocol and name the chance level explicitly; replace 0.25+0.1 with a named constant.
- **Fix at:** `code/run_correspondence_anchor_v2.py:419-435`, `code/run_correspondence_anchor_v2.py:489-508`, `code/run_correspondence_anchor_v2.py:592`

#### `ALGN-014` [INFO] Accuracy recorded without spike-rate/sparsity cost (RULE: accuracy never travels alone)
`code/run_correspondence_anchor_v2.py:595-615` · research-integrity · status:live · effort:medium · confidence:medium
- **What:** The per-seed result dict and CSV record nmnist/shd accuracies and retrieval percentages but never the spike rate / sparsity of the SNN backbones. time_steps=25 lives only in the cfg. Per project integrity rules an SNN accuracy must travel with its spike-rate/sparsity and timestep count. Same omission across all four cluster files.
- **Evidence:** lines 595-615 result={... 'nmnist_after':..., 'trained_v2a':...} with no spike-rate/sparsity field
- **Fix:** Record mean spike rate / sparsity and timestep count alongside every accuracy written to CSV.
- **Fix at:** `code/run_correspondence_anchor_v2.py:595-615`, `code/run_ordinal_anchor.py:541-562`, `code/run_cross_modal_alignment_study.py:355-397`, `code/run_digit_only_alignment.py:328-345`

#### `CMP-027` [INFO] The 2x2 SHARED-vs-SEPARATE design correctly isolates classifier-induced z-alignment (positive note)
`code/run_joint_diagnostic.py:97-127` · correctness · status:live · effort:trivial · confidence:high
- **What:** Retrieval is measured on encoder embeddings z (per_sample_retrieval and compute_held_out_metrics both call model.embed), not on classifier logits. In the SHARED cell the phase-2 within-modality CE backpropagates through the single shared_classifier for BOTH modalities, so the shared head induces cross-modal alignment of the 6-9 embeddings with no cross-modal supervision - a valid positive control for the leak mechanism, absent in the SEPARATE cell. The new per-sample retrieval with TRAIN prototypes (282-333) is a genuine improvement over the transductive quantized 4-class metric it replaces. Recorded as an integrity-positive; no fix.
- **Evidence:** SharedHeadJointModel.forward uses one shared_classifier; train_model_diag Pass 2 (204-213) CE through it for both modalities; retrieval uses model.embed (z)
- **Fix:** No action required.
- **Fix at:** `code/run_joint_diagnostic.py:97-127`, `code/run_joint_diagnostic.py:204-213`, `code/run_joint_diagnostic.py:282-333`

#### `JNT-015` [INFO] Seed set differs from repo policy and from the provenance default's 5th seed
`code/run_joint_magnitude_ablation.py:62` · naming · status:live · effort:trivial · confidence:high
- **What:** This ablation uses a 5-seed list whose 5th seed differs from the provenance default, and neither matches the 3-seed headline policy.
- **Evidence:** SEEDS=[42,123,456,789,1011] (62). Repo policy is [42,123,456] (CLAUDE.md); provenance default seeds use 101112 as the 5th (m7_trimodal_run_provenance.py:63) whereas both joint scripts use 1011 (run_joint_shared_encoder.py:85).
- **Fix:** Standardize the extra-seed set across the joint experiments and note it deviates from the 3-seed headline policy by design.
- **Fix at:** `code/run_joint_magnitude_ablation.py:62`, `code/run_joint_shared_encoder.py:85`

#### `TST-021` [INFO] Verified: importing the two top-level smoke tests does NOT trigger the 4-5h run or a dataset download
`code/test_provenance_smoke.py:1` · safety · status:live · effort:trivial · confidence:high
- **What:** Guidance for whoever fixes these. Statically verified that test_provenance_smoke.py and test_cross_modal_alignment.py only import m7_trimodal_run_provenance (and, for the latter, Dataloaders.paired_digit_loader). m7_trimodal_run_provenance is __main__-guarded (m7_trimodal_run_provenance.py:1378), performs no dataset load / mkdir / DataLoader construction at module scope, and does NOT import the unguarded m7_trimodal_run. Both tests force device='cpu' and use synthetic loaders. Therefore they are safe to move into tests/ and run under `pytest tests/` after being wrapped in test_ functions. This is the opposite of the dangerous code/m7_trimodal_run.py, which must never be imported. The ordinal smoke test is also safe to import (its heavy work is behind main()), but it raises ImportError immediately (TST-016).
- **Evidence:** m7_trimodal_run_provenance.py:1378 __main__ guard; no module-level tonic.datasets/mkdir/DataLoader in that file; it does not import m7_trimodal_run; device cpu at test_provenance_smoke.py:31 and test_cross_modal_alignment.py:58.
- **Fix:** No action required. Use this when relocating the dataset-free smoke tests into tests/ per TST-001.
- **Fix at:** `code/test_provenance_smoke.py:1`, `code/test_cross_modal_alignment.py:1`
- _related: TST-001, TST-002, TST-009, TST-016_

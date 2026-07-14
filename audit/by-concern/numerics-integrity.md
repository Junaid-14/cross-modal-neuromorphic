# Numerics & research integrity (RULE 1-5)

_Generated 2026-07-15. 99 canonical findings in categories: numerics, research-integrity. Duplicates omitted (see `findings.jsonl`)._

Severity: CRITICAL 3, HIGH 34, MED 36, LOW 18, INFO 8


## CRITICAL (3)

#### `FIG-001` [CRITICAL] Both figure scripts write the identical 5 PNG filenames to the same dir - last run wins, can silently ship fabricated figures
`code/generate_figures.py:82-250` · research-integrity · status:live · effort:small · confidence:high
- **What:** generate_figures.py (RESULTS_DIR='./ICONS_M7/results', PLOT_DIR=results/plots, lines 10-11) and generate_figures_3seed.py (lines 28-29) both save the SAME five basenames (fig1_accuracy_comparison.png, fig2_forgetting.png, fig3_gate_trajectory.png, fig4_memory_efficiency.png, fig5_architecture_schematic.png) into the SAME directory. Legacy writes at 82/118/149/181/250; corrected writes at 179/222/276/326/376. Whichever runs last overwrites the other under identical names. generate_figures.py draws every data panel from hardcoded/fabricated numbers and needs no CSV, so running it last (or at all) silently replaces the corrected CSV-sourced figures with fabricated ones bearing the trusted filenames. Neither file namespaces or version-stamps its output.
- **Evidence:** PLOT_DIR / 'fig1_accuracy_comparison.png' in both files
- **Fix:** Retire/delete generate_figures.py (superseded by generate_figures_3seed.py). If kept, write to a clearly-marked separate dir (e.g. plots_LEGACY_FABRICATED/) with renamed outputs. Consolidate on one generator.
- **Fix at:** `code/generate_figures.py:10-11`, `code/generate_figures.py:82`, `code/generate_figures.py:118`, `code/generate_figures.py:149`, `code/generate_figures.py:181`, `code/generate_figures.py:250`, `code/generate_figures_3seed.py:28-29`, `code/generate_figures_3seed.py:179`, `code/generate_figures_3seed.py:222`, `code/generate_figures_3seed.py:276`, `code/generate_figures_3seed.py:326`, `code/generate_figures_3seed.py:376`
- _related: FIG-002, FIG-008_

#### `SNN-001` [CRITICAL] random_init_baseline silently reshuffles training data via the global RNG
`code/m7_trimodal_run_provenance.py:804-823` · numerics · status:live · effort:small · confidence:high
- **What:** After per-seed seeding (L964-965) and construction of the real model (L971), random_init_baseline calls torch.manual_seed(cfg.seed+1000+s) and builds 3 more models, leaving the GLOBAL torch generator at a state that depends on the model's parameter count. Training is not re-seeded before T1 (L997), and DataLoader(shuffle=True) draws its per-epoch shuffle from that global generator. So training data order - and every trained accuracy - is coupled to the baseline routine: change n_seeds, shared_dim, num_patterns, or add a layer and the main result moves with no other edit. Inherited by run_cl_confirmation_fixed.py (call L212) and run_cl_rehearsal.py (call L187).
- **Evidence:** def L804-823; torch.manual_seed L812; called L983; no re-seed before T1 train L997
- **Fix:** Snapshot/restore RNG around the baseline (state=torch.get_rng_state() ... torch.set_rng_state(state)), or build baseline models with a local torch.Generator, and re-seed torch.manual_seed(cfg.seed) immediately before T1.
- **Fix at:** `code/m7_trimodal_run_provenance.py:804-823`, `code/m7_trimodal_run_provenance.py:980-997`
- _related: SNN-007_

#### `RA-001` [CRITICAL] Hardcoded accuracies written into a results CSV
`code/run_rehearsal_transfer_eval_only.py:104-118` · research-integrity · status:live · effort:small · confidence:high
- **What:** The result dict written to cl_rehearsal_results.csv mixes live transfer metrics with hardcoded accuracy literals (T1_after_T1=97.50, T2_after_T2=72.26, T3_after_T3=76.89, T1_after_T2=95.81, T1_after_T3=95.65, T2_after_T3=73.14) and derives every forgetting field from those literals. This manufactures the exact artifact RULE 1 designates as the source of truth; downstream tools would ingest fabricated accuracies as measured. The checkpoint it claims to evaluate (final_dvs_dvs.pt) does not exist.
- **Evidence:** lines 106-107 accuracy literals; 108-112 forgetting arithmetic on them; verdict UNSOURCED
- **Fix:** Delete the hardcoded accuracy/forgetting fields; write only columns computed live this run, or gate the whole CSV write behind a real evaluation.
- **Fix at:** `code/run_rehearsal_transfer_eval_only.py:106-112`
- _confirmed by CLE-002; related: RA-012_


## HIGH (34)

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

#### `RA-005` [HIGH] 3 of 4 columns in the printed COMPARISON TABLE are hardcoded literals
`code/run_cl_rehearsal.py:435-444` · research-integrity · status:live · effort:small · confidence:high
- **What:** Only the first column (C_rehearsal) is live; the A_no_align, C_unfrozen and C_frozen columns are ~24 hardcoded literals (accuracies, forgetting, V2A/prototype transfer). Printed under a bare 'COMPARISON TABLE' header with no disclaimer, the table reads as four measured conditions when three are baked in.
- **Evidence:** lines 435-444; verdict UNSOURCED
- **Fix:** Load the comparison conditions from a CSV, or print only the live column and label the rest explicitly as 'prior-run literals, unsourced'.
- **Fix at:** `code/run_cl_rehearsal.py:435-444`
- _confirmed by SNN-012, CLE-025_

#### `ALGN-004` [HIGH] Hardcoded baseline numbers printed as a comparison table (RULE 1)
`code/run_correspondence_anchor_v2.py:716-718` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** The BASELINE COMPARISON block prints six literal numbers as if measured: A_no_align Retention ~97.0% V2A ~13.3%; Frozen-C Retention 13.3% V2A 14.8%; Unfrozen+rehearsal Retention 95.7% V2A 14.4%. They sit directly beneath the freshly-computed Correspondence rows (line 715), so a reader treats them as comparable results, but none trace to a CSV in this repo. This violates RULE 1 (never put a number anywhere that is not in a CSV).
- **Evidence:** lines 716-718 print(f'   A_no_align ... Retention ~97.0% | V2A ~13.3% | ...') and the two following lines with 13.3/14.8 and 95.7/14.4
- **Fix:** Remove the hardcoded rows or load them from an actual baseline CSV; do not print unsourced numbers next to measured ones.
- **Fix at:** `code/run_correspondence_anchor_v2.py:716-718`

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

#### `JNT-016` [HIGH] Hardcoded baseline numbers written into the report as if measured (RULE 1 violation)
`code/run_joint_shared_encoder.py:989-993` · research-integrity · status:live · effort:small · confidence:high
- **What:** The generated report embeds frozen-separate baseline figures as hardcoded literals, dressed with a plus/minus as if they came from a run, with no CSV backing.
- **Evidence:** report string 990-991: 'Frozen-separate held-out V2A:  ~0.22 ± 0.20 (chance = 0.25)' and 'Frozen-separate Procrustes:     0.00' are literals written to report.txt. There is no results CSV in-repo. (Same anti-pattern appears out-of-scope at run_correspondence_anchor_v2.py:716-718.)
- **Fix:** Read baseline numbers from their source CSV at runtime or omit them entirely; never embed plus/minus-formatted result literals in a generated report.
- **Fix at:** `code/run_joint_shared_encoder.py:989-993`


## MED (36)

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

#### `CLE-011` [MED] Magic verdict thresholds >20 and >12 for V2A survival
`code/eval_transfer_fixed.py:127-131` · numerics · status:live · effort:small · confidence:medium
- **What:** The verdict tiers hardcode V2A>20 (SURVIVED) and V2A>12 (PARTIALLY) with no derivation or link to chance. Chance here is 100/num_classes = 10% (num_classes=10 at line 95), so 20 and 12 are ~2x and ~1.2x chance, but that rationale is undocumented and the constants are unnamed, making the pass/fail scientific call opaque and easy to drift.
- **Evidence:** Lines 127 `if post_t2_v2a > 20:` and 131 `elif post_t2_v2a > 12:`.
- **Fix:** Define named thresholds relative to the eval's own chance (e.g. chance = 100/num_classes; survived if V2A > chance + margin) and document the margin choice.
- **Fix at:** `code/eval_transfer_fixed.py:127`, `code/eval_transfer_fixed.py:131`

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

#### `PROV-018` [MED] Timestep count is inconsistent: cfg.time_steps=25 is dead, visual/DVS are cropped to 25 frames, but SHD runs at 100 timesteps
`code/m7_trimodal_run_provenance.py:121-123` · numerics · status:live · effort:medium · confidence:high
- **What:** cfg.time_steps=25 (52) is never read (PROV-015). The visual and DVS collate fns hard-crop/pad to 25 frames (113-116, 133-140), but SHD is densified to 100 time bins (_events_to_dense time_bins=100, 1201) and shd_collate_fn does no crop/pad (121-123), so SHD tensors reach the model with T=100. Since every backbone loops over T=x.size(1) (275, 353), N-MNIST/DVS run 25 steps while SHD runs 100 — a 4x difference in temporal depth. This makes cfg.time_steps=25 doubly misleading and complicates the mandatory 'accuracy travels with timestep count' reporting, because the per-task timestep cost is not the single documented value.
- **Evidence:** shd_collate_fn stacks without crop (121-123); _events_to_dense time_bins=100 (1201); nmnist/dvs cropped to 25 (113-116,133-140); model loops T=x.size(1) (275,353); cfg.time_steps unread (grep).
- **Fix:** Make the timestep count explicit and consistent: either drive all collate/densify paths from cfg.time_steps, or document per-modality timestep counts and report accuracy alongside the actual per-task T (25 visual / 100 audio). Remove or wire cfg.time_steps.
- **Fix at:** `code/m7_trimodal_run_provenance.py:121-123`, `code/m7_trimodal_run_provenance.py:1200-1210`, `code/m7_trimodal_run_provenance.py:52`
- _related: PROV-015_

#### `SNN-011` [MED] Provenance runs 5 seeds (incl. 101112), not the policy [42,123,456]; three inconsistent seed sets across scripts
`code/m7_trimodal_run_provenance.py:63` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** provenance seeds=[42,123,456,789,101112] (L63, used L1320) vs m7 SEEDS=[42,123,456] (L516). Joint scripts use ...,789,1011 (run_joint_magnitude_ablation L62; run_joint_shared_encoder L85). The two 'M7' entrypoints aggregate over different n (5 vs 3) so their mean+/-std are not comparable, and provenance violates the documented seed policy the loader enforces (load_results.py L156-163). 101112 vs 1011 looks like a typo that fragments cross-script comparability.
- **Evidence:** provenance L63 seeds list; m7 L516
- **Fix:** Set provenance seeds=[42,123,456]; reconcile the joint scripts' fifth seed with the intended value.
- **Fix at:** `code/m7_trimodal_run_provenance.py:63`, `code/m7_trimodal_run_provenance.py:1320`
- _related: SNN-013_

#### `CLE-020` [MED] Saved config misrepresents actual training length (cfg 50/12 vs hardcoded 15/5)
`code/run_cl_confirmation_fixed.py:224-231` · research-integrity · status:live · effort:medium · confidence:high
- **What:** cfg sets max_epochs=50, patience=12 (368) but every train_task call hardcodes num_epochs=15, patience=5 (226,266,302), and align uses num_epochs=5 (241). The config saved next to each checkpoint via cfg.to_json (231,271,307) and results/master_config.json (402) therefore records 50/12, not the 15/5 the run truly used. Anyone reproducing from the saved config would train 50 epochs and not reproduce the checkpoints -- a provenance/reproducibility defect. Identical pattern in run_cl_rehearsal.py (201/249/291 vs 381).
- **Evidence:** cfg max_epochs=50, patience=12 at line 368; train_task hardcodes num_epochs=15, patience=5 at 226/266/302; cfg.to_json at 231/271/307/402.
- **Fix:** Drive num_epochs/patience from cfg (add per-phase fields if 15 vs 50 is intentional) so the persisted config equals what actually ran. Apply the same fix in run_cl_rehearsal.py.
- **Fix at:** `code/run_cl_confirmation_fixed.py:224-231`, `code/run_cl_confirmation_fixed.py:264-271`, `code/run_cl_confirmation_fixed.py:300-307`, `code/run_cl_rehearsal.py:199-207`, `code/run_cl_rehearsal.py:247-257`, `code/run_cl_rehearsal.py:289-299`
- _related: CLE-019_

#### `CLE-024` [MED] Results CSV stores accuracy/forgetting with no spike-rate/sparsity cost metric
`code/run_cl_confirmation_fixed.py:332-347` · research-integrity · status:live · effort:medium · confidence:medium
- **What:** The persisted result dict (332-347) carries T1/T2/T3 test accuracy, FWT and forgetting but no spike rate or sparsity; only the timestep count survives, and only indirectly via the separately saved config (time_steps=25). Per the 'accuracy never travels alone' rule, an SNN accuracy in a results CSV that downstream figures/analysis read must ship with its spike-rate/sparsity and timestep cost. Same for run_cl_rehearsal.py:341-356.
- **Evidence:** result dict keys at 332-347 contain no spike/sparsity field; time_steps only in cfg.to_json output.
- **Fix:** Add measured mean spike rate / sparsity (and time_steps) columns to the result row, or document that cost metrics live in a companion file. Mirror in run_cl_rehearsal.py.
- **Fix at:** `code/run_cl_confirmation_fixed.py:332-347`, `code/run_cl_rehearsal.py:341-356`
- _related: CLE-016_

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

#### `CMP-006` [MED] Aggregation double-counts a seed when a timestamped rerun CSV coexists with its base
`code/run_compositional_crossmodal.py:953-997` · research-integrity · status:live · effort:small · confidence:medium
- **What:** load_rows_from_disk globs model_*_seed*.csv, which matches both model_X_seedN.csv and the model_X_seedN__<ts>.csv variant that _model_csv_path writes when the base already exists. write_summary_and_report means over ALL rows with no collapse to unique (model,seed), so a rerun inflates n_seeds and corrupts mean/std - the same near-duplicate inflation documented for generate_figures_3seed in CLAUDE.md.
- **Evidence:** for p in sorted(RESULTS_DIR.glob('model_*_seed*.csv')): ... ; vals = [r[key] for r in sub ...]  # no unique-seed collapse
- **Fix:** Collapse to one row per (model, seed) (keep the latest timestamp) before aggregating, and warn when duplicates are found.
- **Fix at:** `code/run_compositional_crossmodal.py:953-966`, `code/run_compositional_crossmodal.py:984-997`, `code/run_compositional_crossmodal.py:817-822`
- _related: CMP-022_

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

#### `CMP-022` [MED] Aggregation double-counts a seed when a timestamped rerun CSV coexists with its base
`code/run_joint_diagnostic.py:469-505` · research-integrity · status:live · effort:small · confidence:medium
- **What:** load_rows_from_disk globs cell_*_seed*.csv, which matches both cell_<kind>_supcon_<x>_seedN.csv and the __<ts>.csv variant _cell_csv_path writes when the base exists. write_summary_and_report means over all gate-passing rows per (kind,use_supcon) with no collapse to unique seed, so a rerun inflates n_seeds and corrupts the mean/std - the same near-duplicate inflation documented for generate_figures_3seed.
- **Evidence:** for p in sorted(RESULTS_DIR.glob('cell_*_seed*.csv')) ... vals = [r[m] for r in sub if r.get('gate_passed')]  # no unique-seed collapse
- **Fix:** Collapse to one row per (model_kind, use_supcon, seed) (latest timestamp) before aggregating; warn on duplicates.
- **Fix at:** `code/run_joint_diagnostic.py:469-484`, `code/run_joint_diagnostic.py:487-505`, `code/run_joint_diagnostic.py:357-362`
- _related: CMP-006_

#### `JNT-003` [MED] Silent outcome-dependent seed exclusion via probe>40 assert inside a swallowing try/except
`code/run_joint_magnitude_ablation.py:800-801` · research-integrity · status:live · effort:small · confidence:medium
- **What:** A per-seed probe-accuracy assertion, combined with the main-loop try/except, silently drops any seed whose within-modality probe falls below 40% from the aggregate, biasing mean/std.
- **Evidence:** run_one_arm asserts probe['visual']>40 and probe['audio']>40 (800-801). main wraps each arm/seed in try/except that prints a traceback and continues (917-928), so a failed assert removes that seed from all_results; aggregates then run over the survivors with n=len(sub) (940,948-951). Excluding seeds on an outcome threshold is outcome-dependent filtering.
- **Fix:** Do not gate inclusion on a performance threshold. Record the probe value and report it; if the gate is a genuine validity check, fail the whole sweep loudly rather than dropping one seed. Distinguish real exceptions (OOM) from the assert.
- **Fix at:** `code/run_joint_magnitude_ablation.py:800-801`, `code/run_joint_magnitude_ablation.py:917-928`
- _related: JNT-002_

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

#### `TST-003` [MED] Assertion tolerance (<8.0 pp) contradicts the success message it prints (~3pp)
`code/test_provenance_smoke.py:62-63` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** Line 62 asserts `abs(ri - chance) < 8.0` but line 63 prints `All random-init baselines within ~3pp of chance`. The asserted bound (8.0 pp) is nearly 3x the claimed bound (3 pp), so the printed claim is not what the code checks. The 8.0 is also a magic tolerance with no derivation, and it is statistically under-powered: each random_init_baseline evaluates on DummyLoader(10, ...) which yields 10 batches x 4 = ~40 samples (lines 52-54, 42-48), averaged over n_seeds=2 (line 59). At that sample size an accuracy estimate has a large spread, so the 8.0 bound is arbitrary and can occasionally fail spuriously. Under the repo's RULE-1 culture, printing 'within ~3pp' while enforcing 8pp is a mislabeled claim.
- **Evidence:** code/test_provenance_smoke.py:62 `assert abs(ri - chance) < 8.0`; line 63 message `within ~3pp of chance`; sample count from DummyLoader(10,10) at lines 52-54 and n_seeds=2 at line 59.
- **Fix:** Pick one tolerance, justify it from the sample size, and make the printed message match the asserted bound. Either raise the message to ~8pp or tighten the assert to ~3pp AND increase the DummyLoader sample count so ~3pp is statistically defensible.
- **Fix at:** `code/test_provenance_smoke.py:62-63`
- _related: TST-004_


## LOW (18)

#### `ARCH-020` [LOW] 95% CI uses normal z=1.96 with n=2-3 seeds
`code/archive/run_cl_confirmation.py:497` · research-integrity · status:archive · effort:small · confidence:medium
- **What:** aggregate_confirmation_results computes ci95 = 1.96 * std / sqrt(n) (line 497) and writes ci95_lower/ci95_upper labeled as 95% CI. With the seed policy [42,123,456] n is 2-3, where the normal approximation is inappropriate — a t-multiplier (e.g. t_.975,df=2 ~ 4.30) would be far wider. Labeling a z-based interval '95% CI' at n<=3 overstates precision.
- **Evidence:** ci95 = 1.96 * std / np.sqrt(n) line 497; std uses ddof=1 line 495; seeds=[42,123,456] line 539.
- **Fix:** Use scipy.stats.t.ppf for small-n intervals, or report mean +/- std with explicit n rather than a CI, and label accordingly. Retirement: any 'CI' from this script should be treated as under-wide.
- **Fix at:** `code/archive/run_cl_confirmation.py:495-497`

#### `ARCH-017` [LOW] Null baseline shuffles only 2 held-out classes -> 50% chance it equals same-class
`code/archive/run_correspondence_anchor.py:524-531` · research-integrity · status:archive · effort:small · confidence:medium
- **What:** compute_held_out_alignment builds its null baseline by np.random.shuffle of the held-out class list (line 526), but held_out_classes=(8,9) has just two elements, so the shuffle is the identity with probability 0.5. When identity, null_sims compares each class prototype with itself, making null_baseline equal to same-class similarity and the verdict test (same_class > null_baseline + 0.1, line 651) meaningless. Uses the global numpy RNG (seeded line 550), not the local rng. Compounds ARCH-015.
- **Evidence:** np.random.shuffle(shuffled_classes) line 526 with held_out_classes length 2; verdict comparison line 651.
- **Fix:** Use a permutation that excludes fixed points, or average many derangements, or use off_diag as the control; and use a local Generator. Retirement: another reason v1 held-out numbers are untrustworthy.
- **Fix at:** `code/archive/run_correspondence_anchor.py:524-531`, `code/archive/run_correspondence_anchor.py:651`
- _related: ARCH-015, ARCH-001_

#### `RA-013` [LOW] Non-standard 5-seed set (disclosed, distinct purpose)
`code/final_evaluation.py:26` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** SEEDS=[42,123,456,789,999] differs from the project policy [42,123,456]. It is disclosed in the docstring as '5 random seeds for Poisson input encoding' variability, a legitimate distinct purpose, so not a violation - but easily conflated with the CL seed set.
- **Evidence:** line 26; docstring 8,310-311; verdict DISCLOSED
- **Fix:** No code change required; ensure any reported number states '5 Poisson-input seeds', never '5 CL seeds'.
- **Fix at:** `code/final_evaluation.py:26`
- _confirmed by FIG-021; related: RA-011_

#### `FIG-012` [LOW] Figure 4 hardcodes CONFIG constants (currently correct, disclosed) - single-source-of-truth drift hazard
`code/generate_figures_3seed.py:285-299` · research-integrity · status:live · effort:small · confidence:high
- **What:** SAMPLES_PER_CLASS=50, FEATURE_DIM=512, BYTES=4, CLASSES={nmnist:10,shd:20,dvs:11} and FRAME_ELEMS (285-295) are hardcoded. Verified against source: m7_trimodal_run.py:530 constructs EmbeddingReplayBuffer(feature_dim=512, samples_per_class=50), and time_steps=25 (line 58) matches the nmnist/dvs frame time dims - so the constants are CURRENTLY ACCURATE. The figure title says 'derived from CONFIG, not measured' (322-323) and the arithmetic is printed for audit (304-310), so this is disclosed derivation, NOT fabrication (contrast the legacy [80,150,4], FIG-007). Residual risk: these are hand-copied duplicates of m7 CONFIG; if CONFIG changes they silently go stale while the figure still claims CONFIG-derivation.
- **Evidence:** SAMPLES_PER_CLASS=50 vs m7_trimodal_run.py:530 samples_per_class=50
- **Fix:** Import/read the constants from a single source (the CONFIG dict) rather than re-declaring them, or comment-pin them to m7_trimodal_run.py:530/58 with an update-together note.
- **Fix at:** `code/generate_figures_3seed.py:285-299`
- _related: FIG-007_

#### `FIG-014` [LOW] aggregate() computes mean/std over ALL numeric columns (incl. seed and all-NaN step); re-validate under pinned pandas 3.0.2
`code/generate_figures_3seed.py:88-89` · numerics · status:live · effort:small · confidence:low
- **What:** sub.mean(numeric_only=True) (88) and sub.std(numeric_only=True, ddof=1).fillna(0.0) (89) reduce over every numeric column, which includes 'seed' (mean-of-seeds is meaningless) and the all-NaN 'step' column (mean/std -> NaN, std filled to 0). Currently harmless because m()/s() only read named accuracy/forgetting columns, but it is sloppy and could surface if a numeric column is later treated as a metric. requirements.txt pins pandas==3.0.2; pandas 3.0 changed several reduction/string-dtype defaults, so this reduction path (and ddof=1 single-row -> NaN -> fillna(0) for single-seed) should be re-validated on the pinned version. Low confidence on any specific 3.0 breakage; flagged for verification.
- **Evidence:** sub.mean(numeric_only=True) / sub.std(numeric_only=True, ddof=1)
- **Fix:** Select the metric columns explicitly before mean/std (drop 'seed'/'step'), and add a smoke test under pandas 3.0.2.
- **Fix at:** `code/generate_figures_3seed.py:88-89`
- _related: FIG-016_

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

#### `PROV-023` [LOW] N-MNIST collate silently crops each sample to the first 25 time-windows, discarding most of the temporal window
`code/m7_trimodal_run_provenance.py:113-116` · research-integrity · status:live · effort:small · confidence:medium
- **What:** nmnist_collate_fn truncates any sample with more than 25 frames to e[:25] (113-114). With the N-MNIST ToFrame transform (time_window=1000 over ~300 ms recordings) samples contain far more than 25 windows, so only the first ~25 ms of temporal information is used and the rest is dropped without any log. DVS collate does the same (133-135). This is a modeling choice inherited from the orphan, but it materially affects the temporal-cost interpretation and should be surfaced when reporting accuracy with its timestep count.
- **Evidence:** if e.shape[0] > 25: e = e[:25] (113-114); dvs analogous (133-135).
- **Fix:** Document the 25-frame crop (and its effect on effective temporal coverage) where the timestep count is reported; if fuller temporal coverage is intended, subsample or bin across the whole window instead of taking the head.
- **Fix at:** `code/m7_trimodal_run_provenance.py:107-142`
- _related: PROV-018_

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

#### `CLE-027` [LOW] Magic numbers: raw buffer size 200 (vs cfg 50), max_batches 100, drift eps 1e-6
`code/run_cl_rehearsal.py:178` · numerics · status:live · effort:small · confidence:medium
- **What:** RawInputReplayBuffer(samples_per_class=200) (178) hardcodes 200 while cfg.replay_samples_per_class=50 (386) governs the embedding buffer -- two different, unexplained capacities. raw_buffer.populate(..., max_batches=100) (214,275) hardcodes 100. The backbone-drift sanity uses total_l2 < 1e-6 (330) as a magic threshold. None are named or justified; they affect what rehearsal actually sees.
- **Evidence:** Line 178 samples_per_class=200; lines 214,275 max_batches=100; line 330 `if total_l2 < 1e-6`.
- **Fix:** Promote to named, documented constants or cfg fields; reconcile the 200 vs cfg's 50 so the rehearsal capacity is intentional and recorded.
- **Fix at:** `code/run_cl_rehearsal.py:178`, `code/run_cl_rehearsal.py:214`, `code/run_cl_rehearsal.py:275`, `code/run_cl_rehearsal.py:330`
- _related: CLE-015_

#### `CMP-015` [LOW] Verdict thresholds are hardcoded magic numbers, only partly pre-registered
`code/run_compositional_crossmodal.py:1016-1027` · research-integrity · status:live · effort:small · confidence:medium
- **What:** The auto-verdict uses order>60, wn_held>2.0, asym<5, set_v2a>set_perm+1.0 as decision cutoffs embedded in code. report.txt lists the metrics as 'PRE-REGISTERED' but not these numeric thresholds. Docstring chance values (37-42) are analytic claims and 'Procrustes ~0%' (8) is an unsourced prior result (RULE 1: no CSV in repo).
- **Evidence:** if clears_null and order > 60 and wn_held > 2.0 and asym < 5 and beats_holistic: verdict.append('FACTORIZES: ...')
- **Fix:** Promote thresholds to named constants and record them in the pre-registration block and the config sidecar; mark 'Procrustes ~0%' as an unsourced prior observation.
- **Fix at:** `code/run_compositional_crossmodal.py:1016-1027`, `code/run_compositional_crossmodal.py:37-42`

#### `ALGN-025` [LOW] master_config.json mislabels conditions as no_replay/replay (copy-paste from CL study)
`code/run_cross_modal_alignment_study.py:442` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** The module is an A/B/C alignment study (docstring 1-27; aggregation iterates A_no_align/B_full_512/C_split_128 at line 102), but base_cfg sets conditions=['no_replay','replay'] (line 442) - leftover from the replay CL study. base_cfg.to_json(master_config.json) (528) therefore persists a conditions field that contradicts what the study actually ran, poisoning provenance for anyone trusting the JSON.
- **Evidence:** line 442 conditions=['no_replay','replay']; line 102 for cond in ['A_no_align','B_full_512','C_split_128']; line 528 base_cfg.to_json(results_dir/'master_config.json')
- **Fix:** Set conditions=['A_no_align','B_full_512','C_split_128'] to match the actual run.
- **Fix at:** `code/run_cross_modal_alignment_study.py:442`

#### `JNT-014` [LOW] Population std (ddof=0) reported as '± std' over n<=5, including the non-control shuffle
`code/run_joint_magnitude_ablation.py:948-951` · research-integrity · status:live · effort:trivial · confidence:medium
- **What:** Aggregates use numpy population std labeled with a plus/minus, which understates the sample spread at small n, and the aggregated set mixes in the mislabeled shuffle arm where applicable.
- **Evidence:** np.std (ddof=0) at 948-951, 959-962 and report 978-991 is printed as '± std'; n is 5 seeds (or fewer after JNT-003). Population SD < sample SD at small n.
- **Fix:** Use ddof=1 for a sample standard deviation or explicitly label the figure as population SD; keep n visible (already printed).
- **Fix at:** `code/run_joint_magnitude_ablation.py:948-951`
- _related: JNT-001_

#### `ALGN-020` [LOW] _spearman uses no-ties rho formula on tie-averaged ranks with normal-approx p
`code/run_ordinal_anchor.py:130-144` · numerics · status:live · effort:small · confidence:medium
- **What:** _spearman computes rho via the shortcut 1 - 6*sum(d^2)/(n*(n^2-1)) (line 136), which is only exact when there are no ties, yet it feeds tie-averaged ranks from _rankdata; with ties the rho is slightly biased. The p-value is a normal approximation to the t-distribution (lines 141-143). For n=45 pairs this is roughly OK but not exact; if it enters the paper it should be labeled approximate.
- **Evidence:** line 136 rho = 1.0 - (6.0*np.sum(d*d))/(n*(n*n-1.0)); lines 141-143 t-stat then normal-CDF two-sided p
- **Fix:** Use the Pearson-of-ranks definition (tie-consistent) for rho, or scipy.stats.spearmanr; mark p as approximate if kept.
- **Fix at:** `code/run_ordinal_anchor.py:130-144`

#### `RA-012` [LOW] replay_weight written as 1.0 while the run is configured no-replay
`code/run_rehearsal_transfer_eval_only.py:117` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** The CSV row sets replay_weight=1.0 under condition='C_rehearsal', but the reconstructed config at lines 31-34 has use_replay=False and replay_weight=0.3. The provenance column contradicts the config it claims to reproduce.
- **Evidence:** line 117 vs cfg 31-34; verdict CONTRADICTED
- **Fix:** Write cfg.replay_weight (and cfg.use_replay) rather than a literal, or correct the condition label.
- **Fix at:** `code/run_rehearsal_transfer_eval_only.py:117`, `code/run_rehearsal_transfer_eval_only.py:31-34`
- _confirmed by CLE-003; related: RA-001_

#### `TST-005` [LOW] Hardcoded accuracy-like literals passed into print_task_sanity
`code/test_provenance_smoke.py:103-106` · research-integrity · status:live · effort:trivial · confidence:medium
- **What:** The print_task_sanity call is fed fabricated numbers: after_training_acc=15.0 (line 103), learning_gain=15.0 - acc_shd_zero (line 104), random_init_baseline_acc=5.2 (line 105). These are placeholder literals, not measured quantities. Because this is a smoke test and the values are only printed (never written to a results CSV), the impact is low, but hardcoded accuracy-shaped numbers (15.0, 5.2) in a file whose output is labeled 'SANITY' are exactly the pattern RULE 1 warns against, and a reader skimming stdout could mistake them for measurements.
- **Evidence:** code/test_provenance_smoke.py:103 after_training_acc=15.0; line 104 learning_gain=15.0 - acc_shd_zero; line 105 random_init_baseline_acc=5.2.
- **Fix:** Either drive print_task_sanity from actually-computed values, or clearly comment these as dummy placeholders in the smoke path so they cannot be mistaken for results.
- **Fix at:** `code/test_provenance_smoke.py:103-106`

#### `TST-008` [LOW] Zero-shot section is not seeded, so its outcome is not reproducible by design
`code/test_provenance_smoke.py:66-93` · numerics · status:live · effort:trivial · confidence:medium
- **What:** The test never calls torch.manual_seed / np.random.seed in its own body. The model constructed at line 68 and the training at lines 73-78 consume whatever global RNG state is left over from the preceding random_init_baseline loop (which reseeds internally to cfg.seed+1000+s, lines 58-62). So the zero-shot accuracy that feeds the near-chance check (lines 81-93) depends on incidental RNG ordering, not on a declared seed. Combined with the tight/under-powered tolerances, pass/warn behavior is not reproducible across code changes that alter earlier RNG consumption.
- **Evidence:** No torch.manual_seed/np.random.seed in code/test_provenance_smoke.py; model init at line 68 follows random_init_baseline reseeding at lines 58-62; zero-shot check at lines 89-93.
- **Fix:** Set torch.manual_seed(cfg.seed) and np.random.seed(cfg.seed) at the start of the zero-shot block (or the test function) so the result is deterministic and reproducible.
- **Fix at:** `code/test_provenance_smoke.py:66-70`
- _related: TST-003, TST-004_


## INFO (8)

#### `RA-014` [INFO] Figure 4 memory derived from CONFIG and explicitly labelled (compliant pattern)
`code/generate_figures_3seed.py:281-328` · research-integrity · status:live · effort:trivial · confidence:high
- **What:** The compliant counter-example to RA-006. SAMPLES_PER_CLASS/FEATURE_DIM/BYTES/CLASSES/FRAME_ELEMS are config constants, the arithmetic is printed for audit (304-310), and the figure title reads '(derived from CONFIG, not measured)'. It asserts a footprint from configuration, not a measurement, and says so.
- **Evidence:** lines 281-328, title at 323; verdict DISCLOSED (correct pattern)
- **Fix:** No fix. Keep as the reference pattern for how to present a config-derived quantity.
- **Fix at:** `code/generate_figures_3seed.py:281-328`
- _related: RA-006_

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

#### `ALGN-014` [INFO] Accuracy recorded without spike-rate/sparsity cost (RULE: accuracy never travels alone)
`code/run_correspondence_anchor_v2.py:595-615` · research-integrity · status:live · effort:medium · confidence:medium
- **What:** The per-seed result dict and CSV record nmnist/shd accuracies and retrieval percentages but never the spike rate / sparsity of the SNN backbones. time_steps=25 lives only in the cfg. Per project integrity rules an SNN accuracy must travel with its spike-rate/sparsity and timestep count. Same omission across all four cluster files.
- **Evidence:** lines 595-615 result={... 'nmnist_after':..., 'trained_v2a':...} with no spike-rate/sparsity field
- **Fix:** Record mean spike rate / sparsity and timestep count alongside every accuracy written to CSV.
- **Fix at:** `code/run_correspondence_anchor_v2.py:595-615`, `code/run_ordinal_anchor.py:541-562`, `code/run_cross_modal_alignment_study.py:355-397`, `code/run_digit_only_alignment.py:328-345`

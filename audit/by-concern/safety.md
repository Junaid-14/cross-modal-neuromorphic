# Safety (import-time, destructive, dangerous ops)

_Generated 2026-07-15. 24 canonical findings in categories: safety. Duplicates omitted (see `findings.jsonl`)._

Severity: CRITICAL 2, HIGH 8, MED 9, LOW 2, INFO 3


## CRITICAL (2)

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


## HIGH (8)

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

#### `ALGN-021` [HIGH] Checkpoint-skip staleness trap on T1/T2/T3 (silent stale results)
`code/run_cross_modal_alignment_study.py:211-214` · safety · status:live · effort:medium · confidence:high
- **What:** Each of T1/T2/T3 skips training whenever {condition}_seed{seed}_T{n}.pt exists, loading it with only a '⏩ Loading checkpoint' print (212-214, 276-278, 323-325). This is the documented repo-wide trap: a checkpoint left over from older code silently yields stale results presented as a fresh run, with no code-version or hash guard. Combined with the rerun-append aggregation (ALGN-022), a stale run can quietly enter the summary.
- **Evidence:** lines 212-214 if ckpt_t1.exists(): print('   ⏩ Loading checkpoint'); model.load_state_dict(torch.load(ckpt_t1...)['model_state_dict']); same at 276-278 and 323-325
- **Fix:** Stamp checkpoints with a code/version hash and refuse or warn loudly on mismatch; at minimum log the checkpoint mtime and a hash.
- **Fix at:** `code/run_cross_modal_alignment_study.py:211-214`, `code/run_cross_modal_alignment_study.py:275-278`, `code/run_cross_modal_alignment_study.py:322-325`
- _related: ALGN-022_

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


## MED (9)

#### `FIG-009` [MED] No __main__ guard: module body (incl. load_summary) runs at import
`code/generate_figures_3seed.py:93-94` · safety · status:live · effort:small · confidence:high
- **What:** generate_figures_3seed.py has no `if __name__=='__main__'` guard (grep: 0). The whole analysis executes at import: csv_path assignment and load_summary(csv_path) at 93-94, aggregation at 96, and all five figures at 136-378. Because the results CSV does not exist, importing calls sys.exit() inside load_summary (58-62) - a clean, informative exit, so this is far less dangerous than generate_figures.py. But if the CSV DID exist, a mere import would render and write five PNGs (and it always creates ./ICONS_M7/results/plots via the module-level mkdir at 30). Static inspection therefore cannot import this module; it must be read as text.
- **Evidence:** grep __main__ -> 0 hits; module-level load_summary at 94
- **Fix:** Wrap the module body in main() behind `if __name__=='__main__'` so importing is side-effect-free; keep the informative missing-CSV exit inside main().
- **Fix at:** `code/generate_figures_3seed.py:93-94`, `code/generate_figures_3seed.py:136-378`
- _related: FIG-029_

#### `CLE-029` [MED] Per-task checkpoint-skip silently loads stale weights instead of retraining
`code/run_cl_confirmation_fixed.py:220-222` · safety · status:live · effort:medium · confidence:high
- **What:** For each task, `if ckpt.exists(): print skip; model.load_state_dict(torch.load(ckpt))` (220-222, 260-262, 296-298) skips training whenever a matching final_{task}_{task}.pt is present. This is the documented repo trap (training silently skipped on existing checkpoint) replicated per task: a stale checkpoint from older code yields stale results that look freshly computed, with only a transient stdout note. Same in run_cl_rehearsal.py (195-197, 243-245, 285-287).
- **Evidence:** Lines 220-222, 260-262, 296-298 (confirmation); 195-197, 243-245, 285-287 (rehearsal).
- **Fix:** Record and check a code/version/config hash against the checkpoint before skipping, or require an explicit --resume flag; at minimum warn loudly (not just print) and log the checkpoint mtime.
- **Fix at:** `code/run_cl_confirmation_fixed.py:220-222`, `code/run_cl_confirmation_fixed.py:260-262`, `code/run_cl_confirmation_fixed.py:296-298`, `code/run_cl_rehearsal.py:195-197`, `code/run_cl_rehearsal.py:243-245`, `code/run_cl_rehearsal.py:285-287`
- _related: CLE-018_

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

#### `JNT-019` [MED] Import-time filesystem side effect: RESULTS_DIR.mkdir at module top level
`code/run_joint_shared_encoder.py:56-57` · safety · status:live · effort:trivial · confidence:high
- **What:** Merely importing the module creates a results directory on disk, a side effect that static analysis and tooling should not trigger.
- **Evidence:** RESULTS_DIR = Path('results/joint_shared_encoder'); RESULTS_DIR.mkdir(parents=True, exist_ok=True) at module scope (56-57). The magnitude ablation correctly defers its mkdir into main() (run_joint_magnitude_ablation.py:866).
- **Fix:** Move the mkdir into main()/run_one_seed; module import must have no filesystem side effects.
- **Fix at:** `code/run_joint_shared_encoder.py:56-57`

#### `TST-009` [MED] Entire alignment smoke test executes at import time (no __main__ guard, no functions)
`code/test_cross_modal_alignment.py:1-169` · safety · status:live · effort:small · confidence:high
- **What:** Like the provenance smoke test, there is no __main__ guard and no functions; the body (model build line 75, forward passes lines 86-96, orthogonal/SupCon checks lines 103-115, a 2-epoch alignment run lines 130-138, and 3 optimizer steps lines 152-157) all execute on import. Import==execution. VERIFIED SAFE from the 4-5h trap: device='cpu' (line 58), FakePairedDataset random tensors (lines 26-40), and it imports only m7_trimodal_run_provenance (guarded) plus Dataloaders.paired_digit_loader — not the unguarded m7_trimodal_run. But collecting/importing it still runs real backprop and can error during collection.
- **Evidence:** No `if __name__`/`def ` (only class FakePairedDataset and methods); executable top-level statements lines 67-169; device cpu at line 58; align_cross_modal 2-epoch run at lines 130-138.
- **Fix:** Move the body into test_* functions with a __main__ guard so import/collection does not run training.
- **Fix at:** `code/test_cross_modal_alignment.py:67-169`
- _related: TST-001, TST-002_

#### `TST-002` [MED] Entire smoke test executes at import time (no __main__ guard, no functions)
`code/test_provenance_smoke.py:1-110` · safety · status:live · effort:small · confidence:high
- **What:** There is no `if __name__ == '__main__'` guard and no functions; every statement (lines 17-110) runs the moment the module is imported. That includes constructing the model and running one epoch of ContinualLearner.train_task plus several evaluate() calls. Importing the module IS running the smoke test, so any pytest collection or plain `import` executes training, and a failed assertion surfaces as an import/collection error rather than a clean test failure. VERIFIED SAFE regarding the 4-5h trap: it forces device='cpu' (line 31) and uses DummyLoader random tensors (lines 36-54), and its only heavy import m7_trimodal_run_provenance is __main__-guarded (that module, line 1378) and does not import the unguarded m7_trimodal_run — so importing this file does NOT download datasets or start the long run. The problem is purely that import==execution.
- **Evidence:** No `if __name__` / `def ` in file; top-level executable statements lines 17-34 and 52-110; device forced cpu at line 31; DummyLoader defined lines 36-54.
- **Fix:** Wrap the body in test functions (e.g. test_random_init_baseline, test_zero_shot_near_chance) and a __main__ guard so importing/collecting the module does not execute training.
- **Fix at:** `code/test_provenance_smoke.py:17-110`
- _related: TST-001, TST-009_


## LOW (2)

#### `FIG-010` [LOW] Import-time filesystem side effect: mkdir creates ICONS_M7/results/plots on import
`code/generate_figures_3seed.py:30` · safety · status:live · effort:trivial · confidence:high
- **What:** PLOT_DIR.mkdir(parents=True, exist_ok=True) runs at module level (line 30), so merely importing the module creates ./ICONS_M7/results/plots/ (CWD-relative; only lands where docs expect when CWD is code/). Same pattern in generate_figures.py line 12. Non-destructive, but a side effect that should live inside main().
- **Evidence:** PLOT_DIR.mkdir(parents=True, exist_ok=True) at module scope
- **Fix:** Move the mkdir into main()/the figure-writing path so import has no filesystem effect.
- **Fix at:** `code/generate_figures_3seed.py:30`, `code/generate_figures.py:12`
- _related: FIG-009_

#### `ALGN-012` [LOW] torch.load of on-disk paths without explicit weights_only
`code/run_correspondence_anchor_v2.py:155` · safety · status:live · effort:trivial · confidence:medium
- **What:** torch.load(cache_path) (155) and torch.load(CKPT_SOURCE) (533) load pickled files without weights_only=. Under torch==2.11 the default is weights_only=True (mitigating arbitrary-code risk), and the payloads are state-dict/tensor dicts so they load fine; flag for awareness and forward-compat. Same pattern in run_ordinal_anchor.py:410 and run_cross_modal_alignment_study.py:214,278,325.
- **Evidence:** line 155 torch.load(cache_path, map_location='cpu'); line 533 torch.load(CKPT_SOURCE, map_location=str(device)); no weights_only anywhere in cluster
- **Fix:** Pass weights_only=True explicitly for state-dict loads to be robust across torch versions.
- **Fix at:** `code/run_correspondence_anchor_v2.py:155`, `code/run_correspondence_anchor_v2.py:533`, `code/run_ordinal_anchor.py:410`, `code/run_cross_modal_alignment_study.py:214`, `code/run_cross_modal_alignment_study.py:278`, `code/run_cross_modal_alignment_study.py:325`


## INFO (3)

#### `PROV-026` [INFO] Positive: the provenance hub is import-safe (has a __main__ guard and lazy data loading), unlike the orphan
`code/m7_trimodal_run_provenance.py:1378-1379` · safety · status:live · effort:trivial · confidence:high
- **What:** Unlike code/m7_trimodal_run.py (which loads datasets and runs the full 4-5h experiment at module import time with no guard), this file has if __name__=='__main__': main() (1378-1379), and all dataset loading/experiment work is inside build_data_loaders/main() called only from that guard. The only top-level side effects are cheap: spike_grad = surrogate.atan() (148) and the TASKS dict (955-959). This matters precisely because the module is imported by 19 files — importing it does NOT trigger the long run. No action needed; recorded so downstream tooling does not treat it like the orphan.
- **Evidence:** __main__ guard (1378); data loading inside build_data_loaders (1168) called from main (1339); top-level only spike_grad (148) and TASKS (955).
- **Fix:** None required. Preserve the guard and the lazy-loading structure in any refactor.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1378-1379`
- _related: PROV-007_

#### `CMP-016` [INFO] Runtime subprocess call to git for provenance
`code/run_compositional_crossmodal.py:1204-1207` · safety · status:live · effort:trivial · confidence:high
- **What:** main shells out to `git rev-parse HEAD` to stamp the config sidecar. It is benign and wrapped in try/except (falls back to 'unknown'), noted only for the safety inventory (subprocess exec).
- **Evidence:** sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO_ROOT).decode().strip()
- **Fix:** No action required; optionally read the SHA via a library or guard cwd existence.
- **Fix at:** `code/run_compositional_crossmodal.py:1204-1207`

#### `TST-021` [INFO] Verified: importing the two top-level smoke tests does NOT trigger the 4-5h run or a dataset download
`code/test_provenance_smoke.py:1` · safety · status:live · effort:trivial · confidence:high
- **What:** Guidance for whoever fixes these. Statically verified that test_provenance_smoke.py and test_cross_modal_alignment.py only import m7_trimodal_run_provenance (and, for the latter, Dataloaders.paired_digit_loader). m7_trimodal_run_provenance is __main__-guarded (m7_trimodal_run_provenance.py:1378), performs no dataset load / mkdir / DataLoader construction at module scope, and does NOT import the unguarded m7_trimodal_run. Both tests force device='cpu' and use synthetic loaders. Therefore they are safe to move into tests/ and run under `pytest tests/` after being wrapped in test_ functions. This is the opposite of the dangerous code/m7_trimodal_run.py, which must never be imported. The ordinal smoke test is also safe to import (its heavy work is behind main()), but it raises ImportError immediately (TST-016).
- **Evidence:** m7_trimodal_run_provenance.py:1378 __main__ guard; no module-level tonic.datasets/mkdir/DataLoader in that file; it does not import m7_trimodal_run; device cpu at test_provenance_smoke.py:31 and test_cross_modal_alignment.py:58.
- **Fix:** No action required. Use this when relocating the dataset-free smoke tests into tests/ per TST-001.
- **Fix at:** `code/test_provenance_smoke.py:1`, `code/test_cross_modal_alignment.py:1`
- _related: TST-001, TST-002, TST-009, TST-016_

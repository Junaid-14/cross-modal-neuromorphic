# Dependencies & config

_Generated 2026-07-15. 30 canonical findings in categories: dependencies. Duplicates omitted (see `findings.jsonl`)._

Severity: HIGH 11, MED 15, LOW 2, INFO 2


## HIGH (11)

#### `DEAD-018` [HIGH] benchmark_mps_real.py hardcodes another user's machine path and an MPS-only device
`code/benchmark_mps_real.py:6-15` · dependencies · status:dead · effort:trivial · confidence:high
- **What:** L6 does sys.path.insert(0, /Users/agent1/Documents/neuromorphic/m7-trimodal-cl) - a foreign absolute path from a different developer's machine that does not exist here; the subsequent legacy imports (L9-11) depend on it and will ModuleNotFoundError on this repo. L15 hardcodes DEVICE=torch.device('mps') with no CUDA/CPU fallback, so it also crashes on any non-Apple-Silicon host. The file cannot run anywhere as written.
- **Evidence:** sys.path.insert(0, '/Users/agent1/Documents/neuromorphic/m7-trimodal-cl') L6; DEVICE=torch.device('mps') L15.
- **Fix:** Delete. If ever revived, derive paths relative to the repo and auto-select the device MPS then CUDA then CPU.
- **Fix at:** `code/benchmark_mps_real.py:6`, `code/benchmark_mps_real.py:15`
- _related: DEAD-017_

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

#### `DEAD-002` [HIGH] m7_raw.py has uncommented !pip shell-magic lines - it is not valid Python (SyntaxError)
`code/m7_raw.py:2-23` · dependencies · status:dead · effort:trivial · confidence:high
- **What:** Lines 2-4 and 17-23 are raw IPython/Jupyter shell-magic ('!pip install ...') left uncommented. As a .py module this is a SyntaxError: CPython cannot parse, import, or run the file at all. Confirms it is an unconverted notebook cell dump. This is unique among the dead group - the machine-converted sibling m7_mps.py has these exact lines commented out ('# !pip ...' at its L2-4/L17-23).
- **Evidence:** L2 '!pip install --upgrade jinja2 -q', L3-4, L17-19, L22-23 all begin with '!'; contrast m7_mps.py L2-4/L17-23 which are commented.
- **Fix:** None if the file is deleted (recommended, see DEAD-001). If retained, every '!' line must be commented. Never try to import it to 'check' it.
- **Fix at:** `code/m7_raw.py:2-4`, `code/m7_raw.py:17-23`
- _related: DEAD-001_

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

#### `ALGN-027` [HIGH] --confirm does 'import run_cl_confirmation' but that module lives only in code/archive/
`code/run_digit_only_alignment.py:615-617` · dependencies · status:live · effort:small · confidence:high
- **What:** Under --confirm the entrypoint runs 'import run_cl_confirmation; run_cl_confirmation.main()' (616-617). run_cl_confirmation.py exists ONLY at code/archive/run_cl_confirmation.py; it was moved out of code/, and archive/ is not on sys.path (only REPO_ROOT/code and REPO_ROOT are inserted at 50-51), so this raises ModuleNotFoundError. A differently-named code/run_cl_confirmation_fixed.py exists but is not what the import targets, and the file's own run_full_cl_confirmation (461) is never wired to the flag. The documented --confirm CLI path is dead.
- **Evidence:** lines 615-617 if args.confirm: import run_cl_confirmation; run_cl_confirmation.main(); verified run_cl_confirmation.py only at code/archive/; sys.path inserts only code/ and repo root (50-51)
- **Fix:** Call the local run_full_cl_confirmation(base_cfg) instead of importing the archived module, or restore/relocate a live run_cl_confirmation module and add it to sys.path.
- **Fix at:** `code/run_digit_only_alignment.py:615-617`, `code/run_digit_only_alignment.py:461-505`
- _confirmed by ARCH-002, DEP-015; related: ALGN-029, ALGN-002_

#### `ALGN-017` [HIGH] CKPT_SOURCE and feature cache missing (script un-runnable)
`code/run_ordinal_anchor.py:63-70` · dependencies · status:live · effort:medium · confidence:high
- **What:** CKPT_SOURCE (63-70) is the same missing results/cl_confirmation/.../final_shd_shd.pt as ALGN-002; run_one_seed's torch.load(CKPT_SOURCE) at line 410 raises FileNotFoundError. main() also calls _load_cached_features (imported from v2, line 615) which needs the missing results/hypergraph_anchor/cache (ALGN-003); the docstring says 'from v2' but the cache is written by v1. Script cannot start in the current tree.
- **Evidence:** lines 63-70 CKPT_SOURCE path; line 410 ckpt=torch.load(CKPT_SOURCE, map_location=str(device)); line 615 cached=_load_cached_features(device); verified no results/ dir
- **Fix:** Same as ALGN-002/ALGN-003: repair the upstream producer and/or ship fixtures; correct the 'from v2' comment (cache is a v1 artifact).
- **Fix at:** `code/run_ordinal_anchor.py:63-70`, `code/run_ordinal_anchor.py:410`, `code/run_ordinal_anchor.py:614-615`
- _related: ALGN-002, ALGN-003, ALGN-027_


## MED (15)

#### `ARCH-022` [MED] Stale hardcoded absolute sys.path to /Users/agent1/... (different machine/user)
`code/archive/debug_gradient.py:5` · dependencies · status:archive · effort:trivial · confidence:high
- **What:** Both legacy debug/benchmark scripts sys.path.insert a hardcoded absolute path '/Users/agent1/Documents/neuromorphic/m7-trimodal-cl' that does not exist on this machine (repo lives under /Users/zainhassan/...). The insert is dead and misleading; the subsequent 'from Models.registry import build_model' then only resolves if the CWD happens to be the repo root (Models/ at repo root on the implicit path). Portable only by accident.
- **Evidence:** debug_gradient.py:5 sys.path.insert(0,'/Users/agent1/Documents/neuromorphic/m7-trimodal-cl'); benchmark_mps.py:6 identical path; Models.registry import at debug_gradient.py:9 and benchmark_mps.py:9.
- **Fix:** None for retirement (both DELETE per ARCH-008/ARCH-009). If ever revived, replace with a REPO_ROOT-relative sys.path (Path(__file__).resolve().parents[...]).
- **Fix at:** `code/archive/debug_gradient.py:5`, `code/archive/benchmark_mps.py:6`
- _related: ARCH-008, ARCH-009_

#### `DEP-013` [MED] Archive scripts compute REPO_ROOT one directory too shallow after relocation
`code/archive/run_cl_confirmation.py:49-51` · dependencies · status:live · effort:trivial · confidence:high
- **What:** REPO_ROOT = Path(__file__).resolve().parent.parent. For a file at code/archive/X.py this resolves to code/, not the repo root. So sys.path.insert(0, REPO_ROOT / 'code') points to code/code/ (nonexistent) and sys.path.insert(0, REPO_ROOT) adds code/ (not the repo root). The repo-local `from m7_trimodal_run_provenance import ...` still resolves (that module is in code/), but `from Dataloaders...` / `from utils.cross_modal_transfer...` cannot, because the legacy package lives at the repo root which is never added. These files worked before being moved into archive/; the parent.parent depth was not updated for the extra directory level.
- **Evidence:** code/archive/run_cl_confirmation.py:49 REPO_ROOT=Path(__file__).resolve().parent.parent ; :50-51 sys.path.insert(REPO_ROOT/'code'), sys.path.insert(REPO_ROOT) ; :204 from Dataloaders.paired_digit_loader import ... . Same shape in code/archive/run_correspondence_anchor.py:39-41 with :51 from utils.cross_modal_transfer import ...
- **Fix:** For archive-depth files use parent.parent.parent (or search upward for a repo-root marker) so REPO_ROOT is the actual repo root.
- **Fix at:** `code/archive/run_cl_confirmation.py:49-51`, `code/archive/run_correspondence_anchor.py:39-41`
- _related: DEP-011_

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

#### `DEP-012` [MED] Hardcoded foreign absolute path in sys.path.insert breaks repo-local imports off the origin machine
`code/benchmark_mps_real.py:6` · dependencies · status:live · effort:trivial · confidence:high
- **What:** sys.path.insert(0, '/Users/agent1/Documents/neuromorphic/m7-trimodal-cl') hardcodes another user's directory. That path does not exist on this machine (user zainhassan; repo is at /Users/zainhassan/Documents/cross-modal-neuromorphic), so the subsequent legacy imports `from Models.registry import build_model` (line 9) and `from training.pipeline import train_step` cannot resolve -> guaranteed ImportError whenever the script is run here. It also leaks a foreign filesystem layout. Live file code/benchmark_mps_real.py plus two archive files repeat the exact literal.
- **Evidence:** code/benchmark_mps_real.py:6 sys.path.insert(0, '/Users/agent1/Documents/neuromorphic/m7-trimodal-cl') ; code/archive/benchmark_mps.py:6 (same) ; code/archive/debug_gradient.py:5 (same). benchmark_mps_real.py:9 from Models.registry import build_model
- **Fix:** Replace the hardcoded path with a portable insert such as sys.path.insert(0, str(Path(__file__).resolve().parent.parent)), or adopt proper packaging (see DEP-011).
- **Fix at:** `code/benchmark_mps_real.py:6`, `code/archive/benchmark_mps.py:6`, `code/archive/debug_gradient.py:5`
- _related: DEP-011_

#### `PROV-016` [MED] build_data_loaders unconditionally imports and builds the paired cross-modal loader from the legacy root package, coupling the hub to Dataloaders/ even when alignment is off
`code/m7_trimodal_run_provenance.py:1251-1271` · dependencies · status:live · effort:small · confidence:high
- **What:** build_data_loaders does 'from Dataloaders.paired_digit_loader import get_paired_digit_loaders' (1251) and then constructs paired_loader (1262-1271) UNCONDITIONALLY, even though the paired loader is only consumed when cfg.align_cross_modal is True (1017). The hub has no sys.path insertion, so this import requires each of the 19 importing callers to have already put the repo root on sys.path; if not, build_data_loaders raises ModuleNotFoundError. It also does real work (materialising paired samples) in the common case where it is never used. This couples the shared hub to the legacy bi-modal package that CLAUDE.md marks as separate from M7.
- **Evidence:** from Dataloaders.paired_digit_loader import get_paired_digit_loaders (1251); paired_loader built unconditionally (1262-1271); only used under cfg.align_cross_modal (1017); no sys.path insert in this file (grep).
- **Fix:** Build the paired loader only when cfg.align_cross_modal, and move/guard the import behind that flag. If cross-modal alignment is a first-class hub feature, vendor the needed loader into code/ so the hub does not depend on the legacy root package or on caller sys.path setup.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1250-1271`, `code/m7_trimodal_run_provenance.py:1017-1027`

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

#### `DEP-011` [MED] Legacy repo-root package imports resolve only via runtime sys.path hacks plus correct CWD
`code/run_ablation.py:35-45` · dependencies · status:live · effort:medium · confidence:high
- **What:** code/ has no __init__.py, so scripts reach the repo-root legacy package (Models/, Dataloaders/, training/, utils/) by doing sys.path.insert(0, REPO_ROOT) at runtime before importing. All referenced modules exist (Models/registry.py, training/pipeline.py, training/settings.py, Dataloaders/*_loader.py, utils/cross_modal_transfer.py), so these resolve WHEN the insert runs, but the pattern is fragile: imports must physically follow the insert (hence the noqa: E402 markers scattered through the run_* scripts), and any tool that imports the module without first running the insert (linters, IDEs, test collectors, a different CWD) fails to resolve the legacy package.
- **Evidence:** Importers of legacy packages: code/run_ablation.py:35-45 (Models.registry, training.pipeline, training.settings, Dataloaders.*) ; utils.cross_modal_transfer at code/eval_transfer_fixed.py:29, code/run_cl_rehearsal.py:44, code/run_correspondence_anchor_v2.py:45, code/run_cross_modal_alignment_study.py:61, code/run_digit_only_alignment.py:64, code/run_ordinal_anchor.py:48, code/run_rehearsal_transfer_eval_only.py:19 ; sys.path.insert(0, str(REPO_ROOT)) at code/run_ablation.py and ~15 other code/ files
- **Fix:** Make the legacy package importable as a real package (pip install -e a package at repo root, or a conftest.py/path shim, or convert code/ into a package) so imports do not depend on runtime sys.path mutation and CWD.
- **Fix at:** `code/run_ablation.py:35-45`, `code/eval_transfer_fixed.py:29`, `code/run_cl_rehearsal.py:44`, `code/run_correspondence_anchor_v2.py:45`, `code/run_cross_modal_alignment_study.py:61`, `code/run_digit_only_alignment.py:64`, `code/run_ordinal_anchor.py:48`, `code/run_rehearsal_transfer_eval_only.py:19`
- _related: DEP-012, DEP-013, DEP-015_

#### `DEP-014` [MED] `import run_cl_confirmation` references a module that exists only in archive/
`code/run_digit_only_alignment.py:616-617` · dependencies · status:live · effort:trivial · confidence:high
- **What:** Under the --confirm CLI branch the script does `import run_cl_confirmation; run_cl_confirmation.main()`. There is no run_cl_confirmation.py in code/ (only code/run_cl_confirmation_fixed.py); the sole run_cl_confirmation.py lives in code/archive/, which is never added to sys.path (the script inserts only REPO_ROOT/'code' and REPO_ROOT). Running with --confirm therefore raises ImportError at call time. Almost certainly a stale reference left after the file was renamed/superseded by run_cl_confirmation_fixed.py.
- **Evidence:** code/run_digit_only_alignment.py:616 import run_cl_confirmation ; :617 run_cl_confirmation.main() . find shows run_cl_confirmation.py only under code/archive/ (code/ has run_cl_confirmation_fixed.py). sys.path inserts at code/run_digit_only_alignment.py:50-51
- **Fix:** Point the import at run_cl_confirmation_fixed (verify it exposes main()), or restore/relocate the intended module onto the import path.
- **Fix at:** `code/run_digit_only_alignment.py:616-617`
- _related: DEP-013_

#### `JNT-009` [MED] Dead imports from run_correspondence_anchor_v2 pull in a heavy transitive dependency
`code/run_joint_magnitude_ablation.py:56-59` · dependencies · status:live · effort:trivial · confidence:high
- **What:** Two symbols imported from run_correspondence_anchor_v2 are never used, yet importing that module transitively imports utils.cross_modal_transfer, so an unrelated breakage there fails this script at import for functionality it does not use.
- **Evidence:** Imports 'compute_held_out_metrics as _compute_held_out_metrics_v2' and 'run_procrustes' (56-59). Neither is called: the file defines its own compute_held_out_metrics (572) and run_procrustes_z (648, called at 795). Importing run_correspondence_anchor_v2 runs 'from utils.cross_modal_transfer import evaluate_cross_modal_transfer' (run_correspondence_anchor_v2.py:45).
- **Fix:** Delete the run_correspondence_anchor_v2 import block (56-59); the file is self-sufficient.
- **Fix at:** `code/run_joint_magnitude_ablation.py:56-59`
- _related: JNT-010_

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


## LOW (2)

#### `FIG-030` [LOW] code/ script reaches into the legacy bi-modal package via sys.path hack
`code/final_evaluation.py:219-220` · dependencies · status:live · effort:trivial · confidence:high
- **What:** final_evaluation.py inserts the repo root on sys.path (219) to import Dataloaders.nmnist_loader.get_nmnist_loaders (220). Verified: Dataloaders/nmnist_loader.py exists and defines get_nmnist_loaders (line 13), so the import resolves - it is not broken. But it couples a code/ script to the legacy bi-modal package (Models/Dataloaders/training/utils) that CLAUDE.md keeps separate from the M7 entry point, and depends on CWD/path assumptions. code/ has no __init__.py, hence the sys.path.insert.
- **Evidence:** sys.path.insert(0, repo_root); from Dataloaders.nmnist_loader import get_nmnist_loaders
- **Fix:** Acceptable for an archived verify script; if kept active, document the dependency on the legacy Dataloaders package and the repo-root sys.path requirement.
- **Fix at:** `code/final_evaluation.py:219-220`
- _related: FIG-027_

#### `DEP-009` [LOW] Third-party 'wandb' imported but absent from requirements.txt (optional, guarded)
`code/m7_mps.py:84` · dependencies · status:live · effort:trivial · confidence:high
- **What:** `import wandb` runs inside try/except ImportError with a WANDB_AVAILABLE flag in four live scripts and degrades gracefully with a printed hint. Not in the seed list; adds it. No crash, but W&B logging is silently unavailable on any clean install because the dependency is never declared.
- **Evidence:** code/m7_mps.py:83-88 try: import wandb ... except ImportError: WANDB_AVAILABLE=False ; same at code/m7_raw.py:76, code/m7_smoke_test.py:84, code/m7_trimodal_cl.py:84
- **Fix:** Declare wandb as an optional extra (e.g. a logging extra) so it is installable when W&B logging is wanted; leaving it undeclared is the reason logging never activates.
- **Fix at:** `code/requirements.txt:13`, `code/m7_mps.py:84`, `code/m7_raw.py:76`, `code/m7_smoke_test.py:84`, `code/m7_trimodal_cl.py:84`


## INFO (2)

#### `DEP-018` [INFO] Seed-listed 'datasets' is NOT a missing dependency (it is tonic.datasets)
`code/m7_mps.py:67` · dependencies · status:live · effort:trivial · confidence:high
- **What:** A prior audit seed listed 'datasets' among undeclared third-party imports. This is a false positive and must not be acted on. 'datasets' appears only as `from tonic import datasets, transforms` -- the datasets submodule of tonic, which IS declared (tonic==1.6.0). The HuggingFace 'datasets' package is never imported anywhere in code/; the bare token 'datasets' otherwise appears only in stale print/`!pip install` notebook cruft. Recorded so a downstream fixer does not incorrectly add the HF datasets package to requirements.
- **Evidence:** from tonic import datasets, transforms at code/m7_mps.py:67, code/m7_nmnist_smoke_test.py:16, code/m7_raw.py:59, code/m7_smoke_test.py:67, code/m7_trimodal_cl.py:67 ; no `import datasets` / `from datasets import` anywhere; tonic==1.6.0 at code/requirements.txt:5
- **Fix:** No action. Do NOT add a 'datasets' package to code/requirements.txt.
- **Fix at:** `code/requirements.txt:5`
- _related: DEP-010_

#### `CMP-017` [INFO] Import chain is heavy but import-safe
`code/run_compositional_crossmodal.py:73-95` · dependencies · status:live · effort:medium · confidence:high
- **What:** Importing this file transitively imports run_joint_magnitude_ablation -> run_correspondence_anchor_v2 -> m7_trimodal_run_provenance, plus Dataloaders.paired_digit_loader, tonic and snntorch. All have __main__ guards and none load datasets at import, so importing does not start a run - but it is not free and pulls the whole magnitude/provenance stack. (Audited as text only; do not import to test.) snntorch/tonic/torch/numpy are all present in code/requirements.txt.
- **Evidence:** from run_joint_magnitude_ablation import (...); import snntorch as snn; (magnitude imports tonic + Dataloaders + run_correspondence_anchor_v2)
- **Fix:** No action required; if a lighter import is wanted, split the reused loaders/stems into a data-only module without the anchor_v2 dependency.
- **Fix at:** `code/run_compositional_crossmodal.py:73-95`

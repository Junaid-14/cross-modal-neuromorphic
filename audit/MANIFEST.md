# Manifest — every file under `code/` (audited 2026-07-15)

42 files with findings. Static analysis only. `status`: live / dead / archive. `verdict` = highest-severity finding's title (proxy).

| file | LOC | cluster | status | crit | high | med | low | info | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `m7_trimodal_run.py` | 719 | Core entrypoint (orphan) | live | 4 | 13 | 13 | 13 | 6 | No __main__ guard: importing the module runs the full 6-run experiment and downloads/mater |
| `m7_trimodal_run_provenance.py` | 1379 | Shared library (hub) | live | 2 | 10 | 13 | 6 | 2 | EmbeddingReplayBuffer replay is fundamentally broken: task_ids used as class keys, label_o |
| `run_correspondence_anchor_v2.py` | 749 | Cross-modal alignment | live | 1 | 4 | 4 | 3 | 2 | Held-out training path is doubly broken: CE reduced before mask, and 6 anchors vs labels 0 |
| `generate_figures.py` | 254 | Figures + hw-eval | live | 2 | 9 | 1 | 1 | 0 | Both figure scripts write the identical 5 PNG filenames to the same dir - last run wins, c |
| `run_rehearsal_transfer_eval_only.py` | 127 | CL confirmation / eval | live | 2 | 3 | 2 | 3 | 0 | Fabricated accuracy/forgetting literals written to a results CSV (RULE 1) |
| `run_ordinal_anchor.py` | 774 | Cross-modal alignment | live | 1 | 2 | 1 | 2 | 0 | Stage 1 inherits the broken held-out training from v2 |
| `run_compositional_crossmodal.py` | 1227 | Joint / compositional | live | 0 | 1 | 9 | 5 | 2 | Near-miss negatives leak held-out numbers into training |
| `run_joint_shared_encoder.py` | 1004 | Joint / compositional | live | 0 | 1 | 8 | 8 | 0 | Hardcoded baseline numbers written into the report as if measured (RULE 1 violation) |
| `final_evaluation.py` | 317 | Figures + hw-eval | live | 0 | 3 | 5 | 6 | 1 | '95% CI' computed as mean +/- 1.96*std - missing /sqrt(n); mislabeled statistic |
| `run_joint_magnitude_ablation.py` | 1007 | Joint / compositional | live | 0 | 3 | 6 | 5 | 1 | Shuffled-label control does not shuffle labels (mislabeled null) |
| `m7_mps.py` | 3900 | Dead monolith / tooling | dead | 0 | 4 | 5 | 1 | 1 | m7_mps.py is a dead machine-generated MPS clone that runs a training smoke-test on import |
| `run_digit_only_alignment.py` | 678 | Cross-modal alignment | live | 0 | 2 | 3 | 2 | 0 | --confirm does 'import run_cl_confirmation' but that module lives only in code/archive/ |
| `eval_transfer_fixed.py` | 141 | CL confirmation / eval | live | 0 | 2 | 2 | 2 | 0 | Missing config_nmnist.json crashes main() on the normal path |
| `run_cl_rehearsal.py` | 452 | CL confirmation / eval | live | 0 | 2 | 1 | 3 | 0 | Hardcoded comparison-table numbers printed to the log alongside live results |
| `run_cross_modal_alignment_study.py` | 534 | Cross-modal alignment | live | 0 | 1 | 3 | 2 | 0 | Checkpoint-skip staleness trap on T1/T2/T3 (silent stale results) |
| `archive/run_correspondence_anchor.py` | 767 | Archive (superseded) | archive | 0 | 1 | 2 | 2 | 0 | Held-out-class masking is a no-op: held-out digits are still trained on |
| `test_ordinal_anchor_smoke.py` | 84 | Smoke tests | live | 0 | 1 | 2 | 2 | 0 | ImportError on import: _build_digit_loaders is not exported by run_ordinal_anchor |
| `phase1_sanity_check.py` | 199 | CL confirmation / eval | live | 0 | 1 | 0 | 3 | 0 | Missing config_nmnist.json crashes main() (points at results/cl_confirmation) |
| `requirements.txt` | 13 | Dependencies / config | live | 0 | 2 | 2 | 0 | 0 | Future-dated version pins likely unresolvable on PyPI for CPython 3.11 |
| `benchmark_mps_real.py` | 54 | Dead monolith / tooling | dead | 0 | 1 | 2 | 0 | 0 | benchmark_mps_real.py hardcodes another user's machine path and an MPS-only device |
| `convert_mps.py` | 182 | Dead monolith / tooling | dead | 0 | 1 | 2 | 0 | 0 | convert_mps.py overwrites code/m7_mps.py on import (no __main__ guard) |
| `modify_notebook.py` | 569 | Dead monolith / tooling | dead | 0 | 2 | 1 | 0 | 0 | modify_notebook.py crashes immediately - its target notebook no longer exists |
| `m7_raw.py` | 3877 | Dead monolith / tooling | dead | 0 | 1 | 1 | 0 | 0 | m7_raw.py has uncommented !pip shell-magic lines - it is not valid Python (SyntaxError) |
| `m7_trimodal_cl.py` | 3759 | Dead monolith / tooling | dead | 0 | 1 | 1 | 0 | 0 | m7_trimodal_cl.py is a dead notebook monolith that executes the ENTIRE experiment on impor |
| `monitor_and_stop.py` | 61 | Dead monolith / tooling | dead | 0 | 1 | 1 | 0 | 0 | monitor_and_stop.py can send kill -15/-9 to an unrelated process (hardcoded PID 16672) |
| `m7_nmnist_smoke_test.py` | 472 | Dead monolith / tooling | dead | 0 | 1 | 0 | 0 | 0 | m7_nmnist_smoke_test.py is a dead standalone smoke test that downloads N-MNIST and trains  |
| `m7_smoke_test.py` | 2070 | Dead monolith / tooling | dead | 0 | 1 | 0 | 0 | 0 | m7_smoke_test.py is a dead truncated copy of m7_trimodal_cl.py that trains on import |
| `run_cl_confirmation_fixed.py` | 408 | CL confirmation / eval | live | 0 | 0 | 5 | 6 | 0 | Saved config misrepresents actual training length (cfg 50/12 vs hardcoded 15/5) |
| `run_joint_diagnostic.py` | 700 | Joint / compositional | live | 0 | 0 | 6 | 3 | 1 | 'Above chance' verdict rests on a binomial z that treats correlated per-sample retrievals  |
| `generate_figures_3seed.py` | 382 | Figures + hw-eval | live | 0 | 0 | 1 | 7 | 1 | No __main__ guard: module body (incl. load_summary) runs at import |
| `test_provenance_smoke.py` | 110 | Smoke tests | live | 0 | 0 | 4 | 4 | 1 | None of the three code/ smoke tests are real pytest tests, and none are in the validation  |
| `test_cross_modal_alignment.py` | 169 | Smoke tests | live | 0 | 0 | 2 | 5 | 0 | Entire alignment smoke test executes at import time (no __main__ guard, no functions) |
| `archive/run_cl_confirmation.py` | 627 | Archive (superseded) | archive | 0 | 0 | 3 | 2 | 0 | Archived module still imported by live run_digit_only_alignment.py:616 |
| `archive/verify_full.py` | 355 | Archive (superseded) | archive | 0 | 0 | 4 | 1 | 0 | Brian2/PyNN verify cluster imports brian2 and/or pyNN — absent from requirements.txt |
| `archive/verify_brian2_raw.py` | 376 | Archive (superseded) | archive | 0 | 0 | 2 | 1 | 0 | Sim-vs-sim agreement is NOT SpiNNaker-hardware verification; no energy/latency producible |
| `archive/debug_gradient.py` | 98 | Archive (superseded) | archive | 0 | 0 | 1 | 1 | 0 | Stale hardcoded absolute sys.path to /Users/agent1/... (different machine/user) |
| `run_ablation.py` | 474 | J-legacy | live | 0 | 0 | 2 | 0 | 0 | _save_csv opens in 'w' mode, silently discarding prior ablation rows on a narrower re-run |
| `.gitignore` | 46 | Dependencies / config | live | 0 | 0 | 0 | 1 | 1 | Stale .gitignore comment references notebooks that do not exist |
| `archive/benchmark_mps.py` | 63 | Archive (superseded) | archive | 0 | 0 | 0 | 2 | 0 | Legacy MPS throughput benchmark (targets bi-modal package, not M7) |
| `archive/verify_simulators.py` | 362 | Archive (superseded) | archive | 0 | 0 | 0 | 2 | 0 | Superseded by verify_full.py (earlier per-sample-rebuild variant) |
| `archive/debug_brian2_exact.py` | 195 | Archive (superseded) | archive | 0 | 0 | 0 | 1 | 0 | One-off debug scratch ('why is exact SpiNNaker sim silent') |
| `archive/verify_tiny.py` | 216 | Archive (superseded) | archive | 0 | 0 | 0 | 0 | 1 | Proof-of-concept variant search (tiny synthetic net); keep-archived |
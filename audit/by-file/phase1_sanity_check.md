# Audit — `code/phase1_sanity_check.py` (199 lines)

Cluster **F-cl-eval** · shard `w5_cle` · prefix `CLE`

## Narrative

A diagnostic script that tries to pinpoint when T1 (N-MNIST) accuracy collapses for
`C_split_128` seed 42. It (1) evaluates T1 accuracy at the post-T1/T2/T3 checkpoints, (2)
checks whether the `nmnist` backbone + head are byte-identical across checkpoints (a
freeze/plumbing check), and (3) sanity-checks gate outputs (no NaN, not all-zero, not
all-identical). The logic is generally sound and the script is properly guarded with
`main()` + `__main__`. `compare_state_dicts` (59-73) is notably careful — it handles keys
missing from the second dict, which is exactly the guard the drift check in
`run_cl_rehearsal.py` lacks (see CLE-028).

The issues are lower-severity: the script cannot run today because it opens a config from a
`results/cl_confirmation/...` directory that does not exist (this is the first statement of
`main()`), it carries three unused imports, and it prints T1 accuracy with no accompanying
spike-rate/timestep cost. Note the checkpoint directory here is `cl_confirmation` — **not**
`cl_confirmation_fixed` as in `eval_transfer_fixed.py` — so the two diagnostics point at
different result trees; worth confirming which run this is meant to inspect.

## Findings by severity

### HIGH
- **CLE-013** (100-103, safety) — **Missing `config_nmnist.json` crashes `main()`.**
  `CKPT_DIR = results/cl_confirmation/checkpoints/C_split_128_seed42` (line 32) does not
  exist, so `open(config_path)` (101) raises `FileNotFoundError` before anything else. The
  three `torch.load` calls (128, 139-140, 161) would also fail, but the config open is first.
  Guard the config and each checkpoint; reconcile the `cl_confirmation` vs
  `cl_confirmation_fixed` directory naming with `eval_transfer_fixed.py`.

### LOW
- **CLE-016** (45-56, 130-132, research-integrity) — **T1 accuracy reported without
  spike-rate/sparsity or timestep cost.** `eval_t1` returns only accuracy and the script
  prints `{acc:.2f}%` at 130-132 / 183-185. Low because it is a stdout debug output, not a
  persisted result — but the "accuracy never travels alone" rule means this pattern must not
  migrate into result-writing code.
- **CLE-014** (14-15, 29, dead-code) — `numpy` (14), `pandas` (15) and `random_split` (29)
  imported but unused (grep: `np.`=0, `pd.`=0, `random_split`=import only).
- **CLE-015** (89, numerics) — magic `1e-6` tolerance in
  `all_identical = (gates.std(dim=0) < 1e-6).all()`. Promote to a named, documented constant.
- **CLE-030** (34-42, duplication) — `build_nmnist_test_loader` duplicates the same helper in
  `eval_transfer_fixed.py` and the two run scripts. Consolidate.

## Cross-checks performed
- Confirmed `__main__` guard + `main()` present.
- Confirmed `results/cl_confirmation` does not exist on disk.
- Confirmed `numpy`, `pandas`, `random_split` have zero uses beyond import.
- Confirmed the model-forward unpacking `logits, gate, _ = model(...)` and `_, gate, _`
  (52, 83) is consistent with the model's 3-tuple return used elsewhere in the cluster.
- Confirmed `compare_state_dicts` correctly guards `k not in sd2` (65-66) — no defect there;
  cited as the good pattern the rehearsal drift loop should adopt.
- **No fabricated/hardcoded result numbers written by this file.**

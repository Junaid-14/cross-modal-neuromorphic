# Audit — `code/run_rehearsal_transfer_eval_only.py` (127 lines)

Cluster **F-cl-eval** · shard `w5_cle` · prefix `CLE`

## Narrative

This is the most dangerous file in the cluster and the reason it carries a **critical**
research-integrity finding. It is a 127-line **module-level script with no `main()` and no
`__main__` guard** — every statement executes on import. Its stated job is "standalone
transfer eval + CSV write for the completed rehearsal run", but what it actually does is:

1. Reconstruct an `ExperimentConfig` by hand (22-35) — a copy of `run_cl_rehearsal.py`'s
   config, with a comment admitting it "must match".
2. Instantiate the model/buffer/trainer and `torch.load` a final T3 checkpoint (37-46).
3. Build full N-MNIST + SHD test loaders, densifying the entire SHD test set into a Python
   list (49-77).
4. Run the live cross-modal transfer eval (79-89).
5. **Write a results CSV whose accuracy and forgetting columns are hardcoded literals**,
   interleaved with the live transfer metrics (91-127).

Two things make this critical rather than merely sloppy. First, the CSV **mixes fabricated
numbers with measured ones under identical formatting** (CLE-002) — a direct RULE-1
violation, and the fabricated set matches none of the other number sets circulating in the
cluster. Second, because there is no guard, that fabricated write is an **import-time side
effect** (CLE-001). Today the module happens to crash first at `torch.load` of a missing
checkpoint (CLE-004), which is the only thing preventing the fabricated CSV from being
produced on import.

The live transfer metrics themselves (`v2a`, `a2v`, `prototype_alignment`, `sil_visual`) are
computed correctly via `utils.cross_modal_transfer` — the problem is exclusively the
hardcoded accuracy/forgetting block and the unguarded execution.

## Findings by severity

### CRITICAL
- **CLE-002** (104-123, research-integrity) — **Fabricated accuracy/forgetting literals in a
  results CSV.** `T1_after_T1=97.50`, `T2_after_T2=72.26`, `T3_after_T3=76.89`,
  `T1_after_T2=95.81`, `T1_after_T3=95.65`, `T2_after_T3=73.14` (106-107) are hardcoded, and
  the forgetting fields (108-112) are arithmetic on those literals. They are written to
  `results/cl_rehearsal/cl_rehearsal_results.csv` next to the live transfer columns with the
  same formatting, so measured vs invented is indistinguishable downstream. RULE 1. Fix:
  compute these live from the checkpoint (as `run_cl_rehearsal.run_rehearsal_cell` does) or
  drop the columns entirely.

### HIGH
- **CLE-001** (1-127, safety) — **No `__main__` guard; the whole file runs on import.** Model
  construction (38), `torch.load` (45), full dataset build + SHD densify (54-77), transfer
  eval (81) and the CSV write (120-123) all fire as import side effects. Wrap 37-127 in
  `def main()` + `if __name__ == "__main__": main()`.
- **CLE-004** (43-45, safety) — **`torch.load` of a missing checkpoint crashes on import.**
  `results/cl_rehearsal/checkpoints/C_rehearsal_seed42/final_dvs_dvs.pt` does not exist
  (`results/` is absent), so import raises `FileNotFoundError`. Guard existence; validate the
  `model_state_dict` key.
- **CLE-003** (117, research-integrity) — **`replay_weight=1.0` written to the CSV contradicts
  the reconstructed cfg** (`use_replay=False`, `replay_weight=0.3`, line 31). 1.0 looks like
  the rehearsal `raw_replay_weight`, a different knob; recording it under `replay_weight`
  mislabels provenance.

### MEDIUM
- **CLE-005** (21-35, duplication) — **`cfg` is a hand-copied "must match" duplicate** of
  `run_cl_rehearsal.py`'s config; already drifted (CLE-003). Load the config saved with the
  checkpoint instead of re-declaring.
- **CLE-033** (19, dependencies) — **`evaluate_cross_modal_transfer` transitively imports
  scikit-learn**, which is absent from `code/requirements.txt`; on a clean pinned env this
  import (line 19) fails with `ModuleNotFoundError: sklearn`. Shared across the cluster.

### LOW
- **CLE-006** (116, quality) — CSV column `silhouette_score` stores only `sil_visual`;
  `sil_audio` is dropped. Rename/expand the schema.
- **CLE-007** (17, 39-40, 51, dead-code) — `random_split` (51) and `dvs_collate_fn` (17)
  imported but unused; `trainer` (40) and `buffer` (39) constructed but never used.
- **CLE-030** (49-77, duplication) — `_events_to_dense` / loader code duplicated across the
  cluster; consolidate into a shared import-safe module.
- **CLE-031** (69, dead-code) — dead `else 1` inside a `len(events) > 0` guard (one of four
  copies).
- **CLE-032** (22-35, duplication) — the `ExperimentConfig` literal is triplicated across the
  run/eval scripts.

## Cross-checks performed
- Confirmed **no** `__main__` / `def main` (grep count 0).
- Confirmed `results/cl_rehearsal` does not exist on disk.
- Confirmed the imported symbols resolve in `m7_trimodal_run_provenance.py` and
  `utils/cross_modal_transfer.py`; the latter imports sklearn (not in requirements).
- The hardcoded literals here match neither `run_cl_rehearsal.py`'s stdout comparison-table
  columns nor each other's runs — i.e. an independent unsourced set.

# Audit — `code/final_evaluation.py` (317 lines)

**Cluster:** K-figures-hweval · **Shard:** w6_fig · **Prefix:** FIG · **Status:** live

## Verdict: KEEP but RELABEL + RE-SITE — a hardware-verify experiment with real integrity problems

A "final locked" evaluation that runs a **CPU discrete-time LIF simulator** of the N-MNIST head
against a PyTorch baseline, over 5 Poisson-input seeds, from a checkpoint. It has a proper
`__main__` guard (import-safe — the one clean structural win in this cluster). But it carries
three genuine research-integrity problems: a **mislabeled "95% CI"** (missing `/√n`), **accuracy
reported without spike-rate/sparsity**, and **"SpiNNaker-deployed" / Brian2-equivalence prose**
that overstates what a simulator can establish. It also depends on a checkpoint that does not
exist, and it is **misfiled** in `code/` root while its sibling hardware-verify scripts live in
`code/archive/`.

Note on seeds: the 5 seeds vary **only the Poisson input encoding** for a fixed network + fixed
`SCALE`, so this is an input-noise ensemble, a different axis from the training-seed policy.

---

## HIGH (research-integrity)

### FIG-018 — "95% CI" is `mean ± 1.96·std`, missing `/√n` (line 289)
A CI of the mean needs `± 1.96·std/√n` (or a t-multiplier). Missing `/√5` ≈ 2.24×, so the printed
interval is ~2.24× too wide and is **not a CI of the mean** — it is a spread interval for single
observations, mislabeled. Compounding: `spinnaker_accs.std()` (288) is numpy's **population** std
(`ddof=0`), and `1.96` (z) is used where `t_{.975,4}=2.776` fits n=5. Anything copied from this
line into the paper as a "95% CI" is a mislabeled statistic.

### FIG-019 — Accuracy reported without spike rate / sparsity (lines 251–252, 282–291)
Both the PyTorch baseline (`pt_acc`) and the sim accuracies are printed with **no spike-rate or
sparsity** metric. The sim already accumulates head spikes (`s3_acc`, 147/179) and counts all-zero
samples (`zero_count`, 182/271/291) — but never reports network rate/sparsity. Timestep count is
partial: `DURATION_MS=100` is printed for the sim (202/285) but the baseline's timestep count
(T≈25 for N-MNIST) is not. Violates research-integrity.md's "accuracy never travels alone."

### FIG-020 — "SpiNNaker-deployed" overclaim + unsourced Brian2-equivalence (lines 303–308)
The HONEST STATEMENT calls this "the final, locked evaluation of the **SpiNNaker-deployed
network**" (303) and claims it "was **verified against raw Brian2** to produce identical argmax
predictions on **20 stratified reference samples**" (306–308). Per CLAUDE.md/spinnaker.md nothing
has run on a board (`HARDWARE_ACCESS_CHECKLIST` entirely unchecked). This is a **simulator**, so
"deployed" conflates simulation with hardware (Mapping ≠ inference ≠ real hardware).
`build_spinnaker_network()` is neither defined nor imported here, and no Brian2 comparison runs in
this file — the "20 samples" equivalence is asserted as fact with no in-file/in-repo artifact
(the corresponding `verify_brian2_raw.py` is in `code/archive/`).

---

## MEDIUM

### FIG-021 — `SEEDS=[42,123,456,789,999]` deviates from policy `[42,123,456]` (line 26)
Adds 789/999 beyond the documented set (`m7_trimodal_run.py:516`). Nuance: these seed the Poisson
input RNG (`run_spinnaker_sim` seed, 104), not model init/shuffle — a legitimately different axis.
Still flagged: inconsistent with the documented seeds (invites conflation) and n=5 feeds the
mislabeled CI (FIG-018).

### FIG-022 — `torch.load` of a missing checkpoint crashes the only run path (line 207)
`torch.load('ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt', ...)`. Repo has **no
checkpoints** → immediate `FileNotFoundError`; the whole eval depends on it. Path is CWD-relative
(no `./`; note `ICONS_M7/...` vs the figure scripts' `./ICONS_M7`). Filename lacks the `seed{seed}`
field of the documented pattern. Unlike the figure script's graceful `sys.exit`, this is a bare
traceback. Minor: no explicit `weights_only=` (torch 2.11 defaults it `True`, so secure-by-default,
but a non-tensor checkpoint would then fail).

### FIG-023 — `collect_stratified` silently under-collects; batch_size=1 coupling (lines 88–99)
If a class has fewer than `per_class=200` samples, the `all(...)` break condition never trips, the
finite loader is exhausted, and it returns **fewer** samples with **no error** — while main() still
prints "200/class" (200/223). Assumes batch_size=1 via `int(y.item())` (92; batch_size=1 set at
221). `counts` keyed 0–9 (90) → KeyError if a label ≥10. **Not** an infinite loop (loader is
finite), contrary to a "may loop forever" reading.

### FIG-024 — Headline accuracy hinges on hand-tuned `SCALE=7.0` and other magic scales
`SCALE=7.0` (27→120), `RATE_SCALE=10.0` (28→111), `bias*10.0` (123), `rates/1000.0` (112) all move
the number. "Calibrated once and never adjusted" (8/304), but the calibration and its target are
not in this file or any CSV — an undocumented researcher degree-of-freedom. Also many unnamed LIF
magic constants: `DURATION_MS=100` (29), `refrac_steps=2` (134), thresholds −50/−65, τ 20/5
(130–133).

---

## LOW / INFO

### FIG-025 — argmax over all-zero head counts silently predicts class 0 (lines 183–184)
A sample with zero head spikes → `np.argmax` returns 0 → silently classified as class 0, biasing
toward class 0. `zero_count` tracks the count but predictions still default to 0. If "Confirms 0
all-zero" is ever violated, accuracy already absorbed those as class-0 guesses.

### FIG-026 — `mean_confusion_matrix` misnamed; dead guard; transposed orientation (187–192)
Computes a **single** seed's CM, not a mean (averaging is later at 276). `if p >= 0` (190) is dead
(argmax ≥ 0 always). `cm[p, t]` (191) is `[pred, true]`, transpose of the conventional
`[true, pred]`; the print (294–297) could be misread against sklearn convention.

### FIG-027 — Misfiled: hardware-verify script in `code/` root (line 1)
Peers `verify_brian2_raw.py`, `verify_simulators.py`, `verify_full.py`, `verify_tiny.py`,
`debug_brian2_exact.py` all live in `code/archive/` (confirmed). This one, named "final," sits in
`code/` root implying a current deliverable. Move it to `code/archive/`.

### FIG-028 — `ConvFrontend` duplicates `FullModel` conv layers; both re-implement `SNN_Backbone`
`ConvFrontend` (33–53) restates conv1/pool1/lif1/conv2/pool2/lif2 identical to `FullModel` (59–64),
and both re-implement `m7_trimodal_run.py`'s backbone. Three copies → silent divergence risk if the
backbone changes.

### FIG-029 — POSITIVE: has a `__main__` guard (import-safe) (316–317)
Unlike both figure scripts, work is behind `if __name__=='__main__': main()`. Only module-level
constants and `spike_grad=surrogate.atan()` (24) evaluate on import. Use this as the template for
guarding `generate_figures*.py`.

### FIG-030 — Reaches into the legacy bi-modal package via `sys.path` hack (219–220)
`sys.path.insert(repo_root)` then `from Dataloaders.nmnist_loader import get_nmnist_loaders`.
Verified: `Dataloaders/nmnist_loader.py:13` defines it — import resolves, not broken. But it
couples a `code/` script to the legacy package and to CWD/path assumptions (`code/` has no
`__init__.py`).

---

## Notes / non-issues
- All third-party imports (`torch`, `snntorch`, `numpy`) are in `requirements.txt`.
- The Poisson encoding does not clip `probs` at 1.0 (112) — large features saturate to
  always-spike; minor, not separately filed.

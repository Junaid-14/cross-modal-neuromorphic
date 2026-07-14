# Audit — `code/generate_figures_3seed.py` (382 lines)

**Cluster:** K-figures-hweval · **Shard:** w6_fig · **Prefix:** FIG · **Status:** live

## Verdict: KEEP — this is the CORRECTED generator, and it is largely sound

This script is the fix for the three fabrications in `generate_figures.py`. It sources
accuracy/forgetting from `cl_trimodal_results.csv`, reads the gate trajectory from
`gate_trajectory.csv` (or **skips loudly** rather than fabricating), and derives the memory
figure from CONFIG with the **arithmetic printed** and titled "derived from CONFIG, not
measured." The de-duplication logic is **correct** and defends against the CSV n-inflation trap
described in CLAUDE.md. Its docstring (1–17) is an honest changelog of what was fixed.

Remaining issues are structural, not integrity failures: **no `__main__` guard** (the body runs
at import), a couple of low-severity fragilities, and the shared **filename collision** with the
legacy script (FIG-001) — the single most important cross-file risk.

---

## MEDIUM

### FIG-009 — No `__main__` guard; body runs at import
`grep __main__` → 0 hits. `load_summary(csv_path)` executes at module level (93–94), then all
figures (136–378). Because the CSV is absent, importing calls `sys.exit()` inside `load_summary`
(58–62) — a **clean, informative** exit, far safer than the legacy script. But if the CSV
existed, a mere import would render and write five PNGs, and the module-level `mkdir` (30) always
fires. Wrap the body in `main()`.

---

## LOW / INFO

### FIG-011 — POSITIVE: de-dup is correct (with a wording nit)
`load_summary` filters summary rows via `df['step'].isna() | blank-step` (65) — more robust than
the docs' `.isna()`-only ritual — exact-dedups (66), **warns** on near-duplicate `seed+condition`
rows (68–75), and collapses to one row per `(seed,condition)` with `groupby(...).first()` (77).
This is the right analysis population. Nit: `GroupBy.first()` returns the first **non-null value
per column**, not the first physical row, so the comment "Keeping the FIRST row" (74) is slightly
inaccurate; with scattered NaNs across near-duplicates it could column-wise mix. Harmless for
complete summary rows.

### FIG-012 — Figure 4 hardcodes CONFIG constants (currently correct, disclosed)
`SAMPLES_PER_CLASS=50`, `FEATURE_DIM=512`, `CLASSES`, `FRAME_ELEMS` (285–295). **Verified
against source:** `m7_trimodal_run.py:530` builds `EmbeddingReplayBuffer(feature_dim=512,
samples_per_class=50)` and `time_steps=25` (line 58) matches the frame time dims — so the
constants are **currently accurate**. Disclosed as "derived from CONFIG, not measured" (322–323)
with printed arithmetic (304–310). **Not fabrication** (contrast legacy `[80,150,4]`, FIG-007).
Residual risk: hand-copied duplicates of CONFIG that go stale silently if CONFIG changes.

### FIG-013 — Figure 4 "Raw dense frames" uses DVS pooled 34×34, not raw 128×128
`FRAME_ELEMS['dvs']=25*2*34*34` (294) uses the pooled resolution while the bar is labeled "Raw
dense frames" (313). DVS raw frames are 2×128×128 (~14× larger). This **understates** the raw
total and thus the compression ratio (322) — conservative/anti-flattering, but inconsistent with
the "raw" label.

### FIG-014 — mean/std aggregate over all numeric columns; re-validate under pandas 3.0.2
`sub.mean(numeric_only=True)` (88) and `sub.std(numeric_only=True, ddof=1).fillna(0.0)` (89)
reduce over **every** numeric column, including `seed` (meaningless mean) and the all-NaN `step`
(NaN → filled to 0). Harmless now because `m()`/`s()` only read named metric columns. Note
`requirements.txt` pins **pandas==3.0.2**; 3.0 changed several reduction/string-dtype defaults, so
this path (and the single-seed `ddof=1`→NaN→`fillna(0)` behavior) should be re-validated.
Confidence low on any specific 3.0 breakage.

### FIG-015 — Expected seed set `{42,123,456}` hardcoded (line 120)
Duplicates the policy at `m7_trimodal_run.py:516`. Used only for a warning (121–124), so benign —
a policy check, not a plotted number — but a second copy that can drift.

### FIG-016 — Summary-row mask at line 65 is fragile
`df['step']` raises **KeyError** if the CSV lacks a `step` column (the dual-schema contract is
assumed, not checked). Also under pandas 3.0's new string dtype, `str.strip()==''` can yield a
nullable boolean and masking with `<NA>` can raise. In practice blank cells parse to float NaN
(safe), so conditional — confidence low.

### FIG-017 — Shared duplication with the legacy script
`box()` (339–343) and the Figure 5 schematic (330–378) are duplicated in `generate_figures.py`.
Resolved by retiring the legacy file (FIG-008).

---

## Cross-file
- **FIG-001 (CRITICAL):** identical output filenames/dir as `generate_figures.py`. This
  corrected script's real figures can be overwritten by the legacy fabricated ones.

## Notes / non-issues
- The closing reminder (381–382) explicitly tells the reader to report **test** accuracy and to
  pair accuracy with spike rate / sparsity + timestep — self-aware of RULE 4 and the
  accuracy-alone rule.
- Figure 3's `plt.close()` (278) sits outside `if drew:` — an undrawn figure is discarded, not
  fabricated. Fine.
- `aggregate` uses `ddof=1` and `fillna(0.0)` — sample std with graceful single-seed handling.

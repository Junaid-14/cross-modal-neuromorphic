# Audit — `code/generate_figures.py` (254 lines)

**Cluster:** K-figures-hweval · **Shard:** w6_fig · **Prefix:** FIG · **Status:** live

## Verdict: RETIRE / DELETE

This is the **LEGACY** figure generator. **Every data-bearing panel is fabricated** —
accuracy, forgetting, gate trajectory, and memory are all hardcoded literals, several
explicitly commented as such ("simulated from log data"). It has **no `__main__` guard** and
needs **no CSV**, so importing it (or running it) unconditionally writes five fabricated PNGs.
Critically, it writes the **same five filenames** to the **same directory** as the corrected
generator `generate_figures_3seed.py`, so whichever runs last wins — the mechanism by which
fabricated figures can silently ship under trusted names. It is fully superseded by
`generate_figures_3seed.py`.

Under RULE 1 ("never put a number anywhere that is not in a CSV"), and given the repo has **no
results CSV at all**, this file is the single largest research-integrity liability in the
cluster.

---

## CRITICAL

### FIG-001 — Filename collision with the corrected generator (HEADLINE)
Both scripts define `RESULTS_DIR='./ICONS_M7/results'` → `PLOT_DIR=results/plots` and save the
identical basenames `fig1_accuracy_comparison.png … fig5_architecture_schematic.png`.
Legacy savefigs: lines 82, 118, 149, 181, 250. Corrected savefigs:
`generate_figures_3seed.py` lines 179, 222, 276, 326, 376. Last writer wins; nothing
namespaces or version-stamps output. Because this file needs no CSV, it always produces
output, so it is the more likely "last writer." **`fix_locations` spans both files.**

### FIG-002 — No `__main__` guard; import writes fabricated figures with no inputs
`grep __main__` → 0 hits. The whole body (data block 25–48, figure code 50–252) runs at
import. With all data hardcoded, `import generate_figures` creates `./ICONS_M7/results/plots/`
(line 12) and writes all five fabricated PNGs — no error, no CSV required. This is why static
inspection of this file must be text-only.

---

## HIGH (all research-integrity / RULE 1)

### FIG-003 — Fabricated accuracy (lines 30–32)
`acc_t1=[97.30,97.37]`, `acc_t2=[80.26,81.98]`, `acc_t3=[84.85,84.47]` hardcoded, rendered as
`%.1f%%` bar labels (63–70), titled as M7 results (73). No CSV, no seeds, no test/val
disclosure.

### FIG-004 — Fabricated forgetting (lines 36–37)
`forget_replay=[0.00,0.00,0.00]` etc., plotted (91–107) under "Catastrophic Forgetting"
(110). The all-zero replay row rendered as `+0.00` is the RULE 5 overclaim made visual.

### FIG-005 — Fabricated gate trajectory drives the central novelty claim (lines 39–48)
`gate_*_shd` dicts hardcoded, commented **"simulated from log data"** (39), plotted (122–146)
and annotated **"Replay triggers HGRN specialisation"** (144–146). This is the paper's core
novelty, drawn from invented numbers. `contested-facts.md` records this figure was previously
fabricated; the real per-epoch split now exists in `gate_trajectory.csv`.

### FIG-006 — Mislabeled "single-task" reference lines (lines 78–79)
`axhline(y=97.68, label='N-MNIST single-task')` / `axhline(y=82.16, label='SHD single-task')`
— these are the **legacy bi-modal** paper's M3/M2 numbers, mislabeled and overlaid on a
tri-modal plot. Secondary bug: `ax.legend()` at line 76 runs **before** these axhlines, so
their labels never appear in the legend — two unexplained dashed lines.

### FIG-007 — Fabricated memory footprint + invented compression (line 157)
`memory=[80,150,4]` MB hardcoded; "20× / 37× smaller" annotations (173–178) are just 80/4 and
150/4 of invented values. The corrected script derives this from CONFIG with printed
arithmetic.

### FIG-008 — Retire recommendation (whole file)
Superseded, fabricated throughout, unguarded, colliding. Delete (or archive with a FABRICATED
marker and a non-colliding output path). Consolidate on `generate_figures_3seed.py`.

---

## LOW

### FIG-017 — `box()` helper + Figure 5 schematic duplicated across both scripts
Legacy 185–252 vs corrected 330–378 (and `box()` 192–197 vs 339–343) are near-identical.
Resolved by retiring this file (FIG-008).

### FIG-010 (primary in the corrected file) — import-time `mkdir`
Line 12 `PLOT_DIR.mkdir(...)` runs at module scope here too; same pattern flagged at
`generate_figures_3seed.py:30`.

---

## Notes / non-issues
- All imports used (`matplotlib.pyplot`, `matplotlib.patches`, `numpy`, `pathlib.Path`).
- `ax.set_xscale('linear')` (170) is a no-op; not worth a finding.
- Figure 5 (architecture schematic) asserts no numbers and is not itself an integrity problem
  — but it is the duplicated block (FIG-017).

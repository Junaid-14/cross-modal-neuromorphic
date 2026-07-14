---
description: Load, validate and summarise the tri-modal continual-learning results CSV correctly — applying the mandatory dedup ritual and checking integrity gates before any number is quoted or any figure is generated. Use whenever the user asks about results, accuracy, forgetting, FWT, mean±std, or wants to generate figures. Also use before running generate_figures_3seed.py, which crashes on an incomplete CSV.
allowed-tools: Bash(python3 *), Read, Glob
---

# Analysing the continual-learning results

**Never hand-roll the CSV load.** Run the canonical loader:

```bash
python3 .claude/skills/analyze-cl-results/scripts/load_results.py
```

It finds the CSV, applies the dedup ritual, runs the integrity gates, and prints
mean ± std by condition. It uses only the standard library, so it works **without a
virtualenv** — which matters, because there isn't one.

## Why you must not do this by hand

`cl_trimodal_results.csv` holds **two schemas in one file**:

- **incremental rows** (written mid-run) — these *have* a `step` value
- **summary rows** (written at end of seed) — these have **no** `step`

and reruns *append*, so duplicates pile up. The documented ritual is:

```python
df[df["step"].isna()].drop_duplicates()
```

**That ritual is not sufficient, and this is the trap.** `drop_duplicates()` is an
**exact-match** dedup. If a seed is re-run and its summary row differs in even the last
float digit, *both rows survive*. You then get two rows for one seed, `n` is inflated, and
your reported **± std is computed across a duplicated seed instead of independent ones** —
a silently wrong error bar, which is the worst kind.

The loader detects this (`NEAR-DUPLICATE SUMMARY ROWS`) and aggregates over **unique seeds**,
never over rows. If you see that warning, the CSV needs cleaning and any previously
published ± std drawn from it should be re-checked.

## Integrity gates the loader enforces

| Gate | Why it matters |
|---|---|
| Both `no_replay` **and** `replay` present | `code/generate_figures_3seed.py` indexes both positionally and raises `IndexError` otherwise. **Always run the loader before that script.** |
| Seeds vs the documented `[42, 123, 456]` | The code now runs all three (fixed 2026-07-14; it was `[123, 456]`). **Any CSV written before that fix contains 2-seed results only** — do not label them 3-seed. |
| One summary row per (seed, condition) | See above — the `drop_duplicates()` trap. |

## When there is no CSV

**This is the current state of the repo, and it is not an error.** The loader says
`NO RESULTS YET` and exits 0.

The consequence is the important part, and you must state it rather than glossing over it:
**every number currently quoted in `docs/` and `paper/` has no in-repo source.** Under
RULE 1 ("never put a number anywhere that is not in a CSV") they are all unverified claims.
Do not repeat them as established. To produce real ones, use `/run-m7-experiment`.

## Before any of this reaches the paper

- Report **test** accuracy, not validation — SHD's val/test gap is 15–19 pp.
- Accuracy travels with **spike rate / sparsity and timestep count**. Always.
- **Figures 3 and 4 were hardcoded** until 2026-07-14. Fig 3 now plots `gate_trajectory.csv`
  (or is skipped if absent); Fig 4 is derived from CONFIG. **Any figure not regenerated since
  may still show fabricated values.**

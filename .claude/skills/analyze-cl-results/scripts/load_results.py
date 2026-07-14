#!/usr/bin/env python3
"""The single canonical loader for the tri-modal continual-learning results CSV.

Why this file exists
--------------------
`cl_trimodal_results.csv` writes **two different schemas into one file**:

  * incremental rows, saved after each step   -> these HAVE a `step` value
  * summary rows, saved at the end of a seed  -> these have NO `step`

and because reruns append rather than overwrite, duplicates accumulate. So the only
valid population for analysis is:

    df[df["step"].isna()].drop_duplicates()

That ritual is currently retyped from memory at each call site, and getting it wrong
silently analyses the wrong rows. This module does it once, correctly, and everything
else should call it.

It deliberately uses only the standard library, so it works with **no virtualenv and no
pandas installed** — which is the repo's current state. If pandas is available it is used
for the aggregation, but it is never required.

Usage
-----
    python3 load_results.py [path/to/cl_trimodal_results.csv]

Exit codes: 0 = report produced (including "no results yet"), 1 = a real problem.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

# Where the training scripts actually write. NOTE: they use BASE_DIR = './ICONS_M7',
# which only resolves as documented when the working directory is `code/`.
CANDIDATE_PATHS = (
    "code/ICONS_M7/results/cl_trimodal_results.csv",
    "ICONS_M7/results/cl_trimodal_results.csv",
    "results/cl_trimodal_results.csv",
)

ACC_COLS = ["T1_after_T1", "T2_after_T2", "T3_after_T3",
            "T1_after_T2", "T1_after_T3", "T2_after_T3"]
FORGET_COLS = ["forgetting_T1_T2", "forgetting_T1_T3", "forgetting_T2_T3"]
EXPECTED_CONDITIONS = {"no_replay", "replay"}
DOCUMENTED_SEEDS = {"42", "123", "456"}


def repo_root() -> Path:
    # .claude/skills/analyze-cl-results/scripts/load_results.py -> repo root
    return Path(__file__).resolve().parents[4]


def find_csv(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    root = repo_root()
    for rel in CANDIDATE_PATHS:
        p = root / rel
        if p.is_file():
            return p
    # Last resort: anywhere in the tree.
    for p in root.rglob("cl_trimodal_results.csv"):
        if ".venv" not in p.parts:
            return p
    return None


def load(path: Path) -> list[dict[str, str]]:
    """Return ONLY the valid analysis population: summary rows, deduplicated.

    This is the stdlib equivalent of `df[df['step'].isna()].drop_duplicates()`.
    """
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))

    # `step.isna()` -> the column is absent, empty, or whitespace.
    summary = [r for r in rows if not (r.get("step") or "").strip()]

    # `drop_duplicates()` -> exact duplicate rows collapse, order preserved.
    seen: set[tuple] = set()
    deduped: list[dict[str, str]] = []
    for r in summary:
        key = tuple(sorted(r.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def fnum(row: dict[str, str], col: str) -> float | None:
    v = (row.get(col) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def mean_std(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f} (n=1, no std)"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f} (n={len(vals)})"


def main() -> int:
    explicit = sys.argv[1] if len(sys.argv) > 1 else None
    path = find_csv(explicit)

    if path is None:
        print("NO RESULTS YET.\n")
        print("There is no `cl_trimodal_results.csv` anywhere in this repo.")
        print("This is the expected state before any training run has completed —")
        print("it is not an error.\n")
        print("Consequence (RULE 1): every number currently quoted in docs/ and paper/")
        print("has no in-repo source. Do not repeat any of them as established.\n")
        print("To produce results: /run-m7-experiment  (needs a venv + ~23 GB of datasets)")
        return 0

    raw_count = sum(1 for _ in path.open(encoding="utf-8", errors="replace")) - 1
    rows = load(path)

    print(f"Source: {path}")
    print(f"Raw rows: {raw_count}  ->  valid summary rows after dedup: {len(rows)}")
    print("  (filter: no `step` value, then drop exact duplicates)\n")

    if not rows:
        print("WARNING: the file exists but contains NO summary rows.")
        print("Only incremental (`step`-bearing) rows are present, which means no seed")
        print("has completed. Nothing here is safe to quote.")
        return 0

    seeds = sorted({(r.get("seed") or "").strip() for r in rows if (r.get("seed") or "").strip()})
    conds = sorted({(r.get("condition") or "").strip() for r in rows if (r.get("condition") or "").strip()})
    print(f"Seeds present:      {seeds}")
    print(f"Conditions present: {conds}\n")

    # --- integrity gates ---------------------------------------------------
    problems: list[str] = []

    missing_conds = EXPECTED_CONDITIONS - set(conds)
    if missing_conds:
        problems.append(
            f"Missing condition(s) {sorted(missing_conds)}. "
            f"`code/generate_figures_3seed.py` indexes both `no_replay` and `replay` "
            f"positionally and will raise IndexError on this CSV. Do not run it yet."
        )

    missing_seeds = DOCUMENTED_SEEDS - set(seeds)
    if missing_seeds:
        problems.append(
            f"Missing seed(s) {sorted(missing_seeds)} vs the documented policy [42, 123, 456]. "
            f"The code now runs all three (fixed 2026-07-14; SEEDS was [123, 456]). A CSV "
            f"missing seed 42 was therefore written by the OLD 2-seed code — do not label "
            f"anything from it as a 3-seed result."
        )

    # The documented ritual — `df[df['step'].isna()].drop_duplicates()` — is an EXACT-match
    # dedup. It does NOT guarantee one row per (seed, condition). If a seed is re-run and
    # the summary row differs in even the last float digit, both rows survive, `n` is
    # inflated, and the reported std is computed over a duplicated seed instead of over
    # independent ones. That is a silently wrong error bar. Catch it.
    from collections import Counter

    pairs = Counter(
        ((r.get("seed") or "").strip(), (r.get("condition") or "").strip()) for r in rows
    )
    dupe_pairs = {k: v for k, v in pairs.items() if v > 1}
    if dupe_pairs:
        problems.append(
            "NEAR-DUPLICATE SUMMARY ROWS: "
            + ", ".join(f"seed={s} condition={c} appears {n}x" for (s, c), n in sorted(dupe_pairs.items()))
            + ". drop_duplicates() is exact-match, so re-runs that differ in a single digit "
              "BOTH survive. Aggregating over rows would inflate n and corrupt the ± std. "
              "This loader aggregates over UNIQUE SEEDS (first row wins per seed) to avoid "
              "that — but the CSV needs cleaning, and any previously-published ± std drawn "
              "from it should be re-checked."
        )

    if problems:
        print("INTEGRITY WARNINGS")
        for p in problems:
            print(f"  ! {p}")
        print()

    # --- aggregate ---------------------------------------------------------
    # n is the number of INDEPENDENT SEEDS, never the number of rows.
    print("Aggregates by condition (mean ± std across unique seeds)\n")
    for cond in conds:
        sub = [r for r in rows if (r.get("condition") or "").strip() == cond]

        by_seed: dict[str, dict[str, str]] = {}
        for r in sub:
            seed = (r.get("seed") or "").strip()
            by_seed.setdefault(seed, r)  # first row wins; dupes already flagged above

        cond_seeds = sorted(by_seed)
        print(f"  [{cond}]  seeds={cond_seeds}  (n={len(cond_seeds)} independent seeds)")
        for col in ACC_COLS + FORGET_COLS + ["fwt"]:
            vals = [v for v in (fnum(r, col) for r in by_seed.values()) if v is not None]
            if vals:
                print(f"    {col:<18} {mean_std(vals)}")
        print()

    print("Reminders before any of this reaches the paper:")
    print("  - These are the ONLY numbers with an in-repo source. Everything else is unsourced.")
    print("  - Report TEST accuracy, not validation (SHD val/test gap is 15-19 pp).")
    print("  - Accuracy must travel with spike rate / sparsity and timestep count.")
    print("  - Figures generated before 2026-07-14 may be fabricated (Figs 3 & 4 were hardcoded);")
    print("    regenerate them before using any figure in the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

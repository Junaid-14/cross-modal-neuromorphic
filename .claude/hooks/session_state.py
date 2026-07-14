#!/usr/bin/env python3
"""SessionStart hook — inject the REAL state of this repo into every session.

Why this exists
---------------
This repo's prose has historically described directories, a notebook, datasets
and checkpoints that do not exist on disk. CLAUDE.md said "the notebook is
canonical" when there was no notebook; every runbook opened with
`source code/.venv/bin/activate` when there was no venv.

A session that trusts the prose starts from a false premise. This hook makes
every session start from ground truth instead. If the tree and the docs ever
disagree again, this output wins.

Contract
--------
stdout: JSON with hookSpecificOutput.additionalContext (a SessionStart hook may
inject context this way). Never fail loudly: a broken SessionStart hook would
block all work, so every error path exits 0 with no output.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Directories that are never worth walking (huge, or irrelevant).
PRUNE = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules",
         ".ipynb_checkpoints", ".obsidian", ".mypy_cache", ".ruff_cache"}


def repo_root() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()


def sh(*args: str) -> str:
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=5)
        return p.stdout.strip()
    except Exception:
        return ""


def walk_find(suffixes: tuple[str, ...], limit: int = 200) -> list[Path]:
    """Find files by suffix, pruning heavy dirs. Bounded so we never hang."""
    hits: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in PRUNE]
        for fn in filenames:
            if fn.endswith(suffixes):
                hits.append(Path(dirpath) / fn)
                if len(hits) >= limit:
                    return hits
    return hits


def describe_results_csv(paths: list[Path]) -> list[str]:
    """Summarise any results CSV we find.

    The tri-modal CSV writes TWO schemas into one file: incremental rows carry a
    `step` column, summary rows do not. Analysis must filter `step.isna()` then
    drop duplicates. We report both counts so nobody silently analyses the wrong
    population.
    """
    import csv

    lines: list[str] = []
    for p in paths:
        try:
            with p.open(newline="", encoding="utf-8", errors="replace") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            continue
        if not rows:
            lines.append(f"  - {p.relative_to(ROOT)}: EMPTY")
            continue
        seeds = sorted({r.get("seed", "") for r in rows if r.get("seed")})
        conds = sorted({r.get("condition", "") for r in rows if r.get("condition")})
        summary_rows = [r for r in rows if not (r.get("step") or "").strip()]
        lines.append(
            f"  - {p.relative_to(ROOT)}: {len(rows)} rows "
            f"({len(summary_rows)} summary rows with empty `step`); "
            f"seeds={seeds or 'none'}; conditions={conds or 'none'}"
        )
        missing = {"no_replay", "replay"} - set(conds)
        if missing:
            lines.append(
                f"    WARNING: missing condition(s) {sorted(missing)} — "
                f"code/generate_figures_3seed.py will raise IndexError on this CSV."
            )
    return lines


def main() -> None:
    L: list[str] = []
    L.append("## Live repo state (SessionStart hook — this is ground truth, not prose)")
    L.append("")

    # --- git --------------------------------------------------------------
    branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD") or "?"
    dirty = sh("git", "status", "--porcelain")
    n_dirty = len([x for x in dirty.splitlines() if x.strip()])
    head = sh("git", "log", "-1", "--format=%h %s")
    L.append(f"- **git**: branch `{branch}`, {n_dirty} uncommitted change(s). HEAD: {head or '?'}")

    # --- environment ------------------------------------------------------
    venv = ROOT / "code" / ".venv" / "bin" / "python"
    if venv.exists():
        ver = sh(str(venv), "--version")
        L.append(f"- **venv**: PRESENT at `code/.venv` ({ver or 'unknown version'})")
    else:
        py311 = shutil.which("python3.11")
        hint = f"`python3.11` available at {py311}" if py311 else "`python3.11` NOT on PATH"
        L.append(
            f"- **venv**: **ABSENT**. `code/.venv` does not exist, so every command in the "
            f"docs that starts with `source code/.venv/bin/activate` will fail. {hint}. "
            f"Run the `/bootstrap-env` skill to create it."
        )

    # --- datasets ---------------------------------------------------------
    ds = [p for p in (ROOT / "ICONS_M7", ROOT / "code" / "ICONS_M7") if p.exists()]
    if ds:
        L.append(f"- **datasets**: present at {', '.join(str(p.relative_to(ROOT)) for p in ds)}")
    else:
        L.append(
            "- **datasets**: **ABSENT**. No `ICONS_M7/` anywhere (~23 GB of N-MNIST / SHD / "
            "DVS-Gesture). Any training run will try to download them."
        )

    # --- results ----------------------------------------------------------
    csvs = [p for p in walk_find((".csv",)) if ".venv" not in p.parts]
    if csvs:
        L.append(f"- **results**: {len(csvs)} CSV file(s) found:")
        L.extend(describe_results_csv(csvs))
    else:
        L.append(
            "- **results**: **NONE**. There is no results CSV anywhere in the repo. "
            "Every number in the docs and the paper therefore has **no in-repo source** — "
            "treat all of them as unverified until a run produces a CSV. (RULE 1: never put "
            "a number anywhere that is not in a CSV.)"
        )

    # --- checkpoints ------------------------------------------------------
    ckpts = walk_find((".pt", ".pth"))
    L.append(
        f"- **checkpoints**: {len(ckpts)} `.pt`/`.pth` file(s)."
        + (
            ""
            if ckpts
            else " None. Note `code/m7_trimodal_run.py` SILENTLY SKIPS training when a matching "
            "`final_*.pt` exists — with none present, a run will train from scratch (correct)."
        )
    )

    # --- notebooks --------------------------------------------------------
    nbs = walk_find((".ipynb",))
    if nbs:
        L.append(f"- **notebooks**: {len(nbs)} found.")
    else:
        L.append(
            "- **notebooks**: **NONE**. Older docs call the notebook 'canonical' and reference "
            "'Cell 11' / 'Cell 30'. There is no notebook in this repo. `code/modify_notebook.py` "
            "is the only surviving record of its structure, and its target file is gone."
        )

    L.append("")
    L.append(
        "**Default validation loop:** `pytest tests/` (6 suites, tonic mocked — fast, safe, "
        "needs no datasets). This is the only quick check that works today."
    )

    out = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n".join(L),
        }
    }
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Never block a session on a broken context hook.
        pass
    sys.exit(0)

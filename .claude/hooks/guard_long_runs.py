#!/usr/bin/env python3
"""PreToolUse(Bash) hook — never let a multi-hour experiment start by accident.

The specific hazard this exists for
-----------------------------------
`code/m7_trimodal_run.py` has **no `if __name__ == "__main__"` guard** and loads
its datasets at module import time. So this:

    python -c "import m7_trimodal_run"

does not "check the imports". It starts a 4-5 hour tri-modal training run. A
casual `import` to introspect the module is indistinguishable, to the shell,
from launching the full experiment.

`code/modify_notebook.py` is a different hazard: it rewrites a notebook **in
place** via exact-string replacement — and the notebook it targets no longer
exists in the repo.

We do not block these. We surface them and ask, so a human makes the call.

Contract: stdout JSON with hookSpecificOutput.permissionDecision in
{allow, ask, deny}. Exit 0. Emitting nothing = defer to normal permissions.
"""
from __future__ import annotations

import json
import re
import sys

# Scripts whose *import alone* triggers the full run (verified: no __main__ guard,
# datasets loaded at module level).
IMPORT_IS_A_RUN = {
    "m7_trimodal_run": (
        "importing this module RUNS the full 4-5 hour tri-modal CL experiment "
        "(no __main__ guard; datasets load at import time)"
    ),
}

# Long-running or destructive scripts, keyed by filename.
HEAVY = {
    "m7_trimodal_run.py": "full tri-modal CL run — ~4-5 hours on MPS",
    "m7_trimodal_run_provenance.py": "tri-modal CL run with provenance — hours",
    "m7_trimodal_cl.py": "tri-modal CL variant — hours",
    "m7_mps.py": "monolithic MPS run — hours",
    "m7_raw.py": "monolithic raw run — hours",
    "m7_smoke_test.py": "tri-modal smoke test — long, and downloads datasets",
    "run_compositional_crossmodal.py": "compositional cross-modal study — long",
    "run_joint_shared_encoder.py": "joint shared-encoder experiment — long",
    "run_joint_magnitude_ablation.py": "magnitude ablation — long",
    "run_joint_diagnostic.py": "joint diagnostic sweep — long",
    "run_cross_modal_alignment_study.py": "alignment study — long",
    "run_correspondence_anchor_v2.py": "correspondence anchor study (8 seeds) — long",
    "run_ordinal_anchor.py": "ordinal anchor study — long",
    "run_digit_only_alignment.py": "digit-only alignment ablation — long",
    "run_cl_rehearsal.py": "rehearsal CL run — long",
    "run_ablation.py": "ablation sweep — long",
}

DESTRUCTIVE = {
    "modify_notebook.py": (
        "rewrites a Jupyter notebook IN PLACE via exact-string replacement — and its "
        "target notebook does not exist in this repo any more, so it will either fail "
        "or corrupt whatever it does find. Back up before running."
    ),
}


def ask(reason: str) -> None:
    sys.stdout.write(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    sys.exit(0)


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except Exception:
        return
    if payload.get("tool_name") != "Bash":
        return

    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not cmd.strip():
        return

    # 1) The import-is-a-run trap.
    for mod, why in IMPORT_IS_A_RUN.items():
        if re.search(rf"\b(?:import|from)\s+{re.escape(mod)}\b", cmd):
            ask(
                f"This command imports `{mod}` — {why}.\n\n"
                f"If you only meant to inspect the module, read the file instead. "
                f"Confirm only if you intend to start the run."
            )

    # 2) Destructive utilities.
    for fn, why in DESTRUCTIVE.items():
        if fn in cmd:
            ask(f"`{fn}` {why}\n\nCommand: {cmd.strip()[:400]}")

    # 3) Long-running experiments.
    for fn, why in HEAVY.items():
        if fn in cmd and re.search(r"\b(python|python3|python3\.\d+|uv run|nohup|caffeinate)\b", cmd):
            ask(
                f"This launches `{fn}` — {why}.\n\n"
                f"Pre-flight worth checking first (the `/run-m7-experiment` skill does this):\n"
                f"  - Is `code/.venv` present and are the datasets downloaded?\n"
                f"  - Do stale `final_*.pt` checkpoints exist? m7_trimodal_run.py SILENTLY SKIPS\n"
                f"    training when one matches, and loads it instead — producing stale results.\n"
                f"  - Are the seeds right? SEEDS should be [42, 123, 456] (fixed 2026-07-14 —\n"
                f"    it was [123, 456], which silently made every 3-seed claim a 2-seed number).\n\n"
                f"Command: {cmd.strip()[:400]}"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)

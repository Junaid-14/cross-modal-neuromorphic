#!/usr/bin/env python3
"""PostToolUse(Write|Edit) hook — byte-compile any Python file we just touched.

Cheap, immediate feedback. The scripts in `code/` are large (some >100 KB) and
edited by exact-string replacement; a mangled edit can otherwise sit unnoticed
until someone burns four hours discovering it at runtime.

This only checks files THIS session just wrote — it never audits the repo at
large, so pre-existing breakage in untouched files stays out of the way.

Contract: exit 2 on failure. For PostToolUse, exit 2 sends stderr back to Claude
as a blocking error (the tool already ran, so this prompts a fix rather than
preventing the write). Exit 0 otherwise.
"""
from __future__ import annotations

import json
import py_compile
import sys
import tempfile
from pathlib import Path

# Don't police files we don't own the quality of.
SKIP_PARTS = {".venv", "__pycache__", "archive", "site-packages", ".git"}


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        return 0

    tool_input = payload.get("tool_input") or {}
    raw_path = tool_input.get("file_path") or ""
    if not raw_path:
        return 0

    path = Path(raw_path)
    if path.suffix != ".py" or not path.is_file():
        return 0
    if SKIP_PARTS & set(path.parts):
        return 0

    # Compile to a throwaway location so we never litter __pycache__.
    with tempfile.TemporaryDirectory() as td:
        cfile = str(Path(td) / "out.pyc")
        try:
            py_compile.compile(str(path), cfile=cfile, doraise=True)
        except py_compile.PyCompileError as exc:
            sys.stderr.write(
                f"Syntax error in the file you just wrote — {path}:\n\n{exc}\n\n"
                f"Fix it before continuing; the file is currently unimportable.\n"
            )
            return 2
        except Exception:
            return 0  # unreadable/encoding oddity — not our business

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)

#!/usr/bin/env python3
"""Record a teachable lesson for the M7 automation assistants.

Usage:
    python code/automation/lesson.py "The magazine keeps including robotics papers..."
    python code/automation/lesson.py --list

Lessons are saved to knowledge/01-Lessons/ and read by other assistants before
they start work.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from automation import write_note
from automation.vault import list_recent_notes, log_session, read_note


def _list_lessons(*, n: int = 10) -> str:
    notes = list_recent_notes("lessons", n=n)
    if not notes:
        return "No lessons recorded yet."
    parts = []
    for note in notes:
        text = read_note(note, max_chars=1000)
        parts.append(f"## {note.stem}\n{text}")
    return "\n\n".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    """Record or list lessons for the automation assistants."""
    parser = argparse.ArgumentParser(description="Record a lesson for the automation assistants.")
    parser.add_argument("lesson", nargs="?", help="The lesson text to record")
    parser.add_argument("--list", action="store_true", dest="list_lessons", help="List recent lessons")
    parser.add_argument("--n", type=int, default=10, help="Number of lessons to list")
    args = parser.parse_args(argv)

    if args.list_lessons:
        print(_list_lessons(n=args.n))
        return 0

    if not args.lesson or not args.lesson.strip():
        print("Error: provide a lesson text or use --list.", file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = write_note(
        "lessons",
        f"{stamp}.md",
        args.lesson.strip(),
        frontmatter={
            "timestamp": stamp,
            "source": "lesson",
            "lesson": args.lesson.strip(),
        },
    )
    log_session("lesson", path, note="recorded new lesson")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

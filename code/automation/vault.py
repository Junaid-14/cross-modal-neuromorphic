"""Shared utilities for reading and writing the `knowledge/` vault."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .constants import VAULT_FOLDERS, VAULT_ROOT


def ensure_vault() -> None:
    """Create all vault folders if they don't exist."""
    VAULT_ROOT.mkdir(parents=True, exist_ok=True)
    for folder in VAULT_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)


def write_note(folder_key: str, filename: str, content: str, *, frontmatter: Optional[dict] = None) -> Path:
    """Write a markdown note into the vault and return its path.

    Parameters
    ----------
    folder_key:
        Key in ``VAULT_FOLDERS``.
    filename:
        Name of the markdown file (should include ``.md``).
    content:
        Markdown body.
    frontmatter:
        Optional dict rendered as YAML frontmatter.
    """
    ensure_vault()
    folder = VAULT_FOLDERS[folder_key]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename

    lines: List[str] = []
    if frontmatter:
        lines.append("---")
        for key, value in frontmatter.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("---")
        lines.append("")

    lines.append(content.rstrip())
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def list_recent_notes(folder_key: str, *, n: int = 5) -> List[Path]:
    """Return the ``n`` most recently modified notes in a vault folder."""
    folder = VAULT_FOLDERS[folder_key]
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix == ".md"]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]


def read_note(path: Path, *, max_chars: int = 4000) -> str:
    """Read a note, returning up to ``max_chars`` characters."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:max_chars]


def read_recent_lessons(*, n: int = 10) -> str:
    """Read the most recent lesson notes as a single context block."""
    lessons = list_recent_notes("lessons", n=n)
    if not lessons:
        return "(no lessons recorded yet)"
    parts = []
    for lesson in lessons:
        text = read_note(lesson, max_chars=2000)
        # Strip YAML frontmatter for brevity
        text = re.sub(r"^---\n.*?---\n", "", text, flags=re.DOTALL)
        parts.append(f"### {lesson.name}\n{text.strip()}")
    return "\n\n".join(parts)


def log_session(command: str, report_path: Path, *, status: str = "ok", note: str = "") -> Path:
    """Append a short diary entry for an automation run."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sessions_dir = VAULT_FOLDERS["sessions"]
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"{datetime.now():%Y-%m-%d}.md"

    entry = f"- [{now}] `{command}` → `{report_path}` ({status})"
    if note:
        entry += f" — {note}"
    entry += "\n"

    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
        if f"## {datetime.now():%Y-%m-%d}" not in existing:
            existing += f"\n## {datetime.now():%Y-%m-%d}\n\n"
    else:
        existing = f"# Session Log\n\n## {datetime.now():%Y-%m-%d}\n\n"

    existing += entry
    log_path.write_text(existing, encoding="utf-8")
    return log_path

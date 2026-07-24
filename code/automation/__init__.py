"""Automation assistants for the M7 tri-modal continual learning project.

Each module in this package implements one of the slash commands documented in
`docs/AUTOMATION_TUTORIAL.md`. All assistants write their reports into the
`knowledge/` vault and never edit production code or the paper on their own.
"""

from .constants import VAULT_ROOT
from .vault import list_recent_notes, read_recent_lessons, write_note

__all__ = ["VAULT_ROOT", "write_note", "list_recent_notes", "read_recent_lessons"]

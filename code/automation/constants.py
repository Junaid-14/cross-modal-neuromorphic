"""Paths and shared constants for the automation assistants."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VAULT_ROOT = REPO_ROOT / "knowledge"

VAULT_FOLDERS = {
    "mocs": VAULT_ROOT / "00-MOCs",
    "lessons": VAULT_ROOT / "01-Lessons",
    "magazine": VAULT_ROOT / "10-Research-Magazine",
    "repo_watch": VAULT_ROOT / "20-Repo-Watch",
    "consultant": VAULT_ROOT / "30-Consultant",
    "paper": VAULT_ROOT / "40-Paper",
    "sessions": VAULT_ROOT / "50-Sessions",
    "patches": VAULT_ROOT / "60-Patches",
}

STEERING_PATH = REPO_ROOT / "steering" / "project_state.json"
REQUIREMENTS_PATH = REPO_ROOT / "code" / "requirements.txt"
PAPER_PATH = REPO_ROOT / "paper" / "cross_modal_correspondence.tex"

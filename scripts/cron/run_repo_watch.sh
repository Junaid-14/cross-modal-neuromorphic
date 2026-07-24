#!/usr/bin/env bash
# Run the daily Repo Watch assistant.
# This script is intended to be invoked by cron.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_FILE="$REPO_ROOT/knowledge/50-Sessions/cron.log"
VENV="$REPO_ROOT/code/.venv/bin/activate"

mkdir -p "$(dirname "$LOG_FILE")"

echo "[$(date -Iseconds)] Starting repo-watch" >> "$LOG_FILE"

# shellcheck source=/dev/null
if [ -f "$VENV" ]; then
    . "$VENV"
fi

cd "$REPO_ROOT"
python code/automation/repo_watch.py >> "$LOG_FILE" 2>&1 || echo "[$(date -Iseconds)] repo-watch failed" >> "$LOG_FILE"

echo "[$(date -Iseconds)] Finished repo-watch" >> "$LOG_FILE"

#!/usr/bin/env bash
# Run the daily Research Magazine assistant.
# This script is intended to be invoked by cron.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_FILE="$REPO_ROOT/knowledge/50-Sessions/cron.log"
VENV="$REPO_ROOT/code/.venv/bin/activate"

mkdir -p "$(dirname "$LOG_FILE")"

echo "[$(date -Iseconds)] Starting research-magazine" >> "$LOG_FILE"

# shellcheck source=/dev/null
if [ -f "$VENV" ]; then
    . "$VENV"
fi

cd "$REPO_ROOT"
python code/automation/magazine.py >> "$LOG_FILE" 2>&1 || echo "[$(date -Iseconds)] research-magazine failed" >> "$LOG_FILE"

echo "[$(date -Iseconds)] Finished research-magazine" >> "$LOG_FILE"

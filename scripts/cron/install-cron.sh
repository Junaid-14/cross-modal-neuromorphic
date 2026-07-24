#!/usr/bin/env bash
# One-time setup: install cron jobs for the M7 automation assistants.
# Safe to re-run; it will not create duplicate entries.
#
# This is the Python-only cron schedule. For the macOS launchd schedule that uses
# claude -p, see scripts/cron/install.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CRON_FILE="/tmp/m7_automation_cron"

cat > "$CRON_FILE" <<EOF
# M7 automation assistants (installed by scripts/cron/install-cron.sh)
57 8 * * * cd "$REPO_ROOT" && bash "$REPO_ROOT/scripts/cron/run_magazine.sh"
3 9 * * * cd "$REPO_ROOT" && bash "$REPO_ROOT/scripts/cron/run_repo_watch.sh"
EOF

# Append only if these exact lines are not already present.
if crontab -l 2>/dev/null | grep -qF "$REPO_ROOT/scripts/cron/run_magazine.sh"; then
    echo "M7 cron jobs are already installed."
else
    (crontab -l 2>/dev/null || true; cat "$CRON_FILE") | crontab -
    echo "M7 cron jobs installed."
fi

rm -f "$CRON_FILE"
echo "Schedule:"
echo "  research-magazine at 8:57 AM local time"
echo "  repo-watch at 9:03 AM local time"
echo "Logs: $REPO_ROOT/knowledge/50-Sessions/cron.log"

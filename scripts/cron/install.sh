#!/usr/bin/env bash
# Install the M7 launchd jobs. Usage:
#   bash scripts/cron/install.sh                 # magazine + repo-watch (daily)
#   bash scripts/cron/install.sh --with-improver # also the improver (opt-in, after hand-tuning)
#
# For the simpler cron-based schedule (no Claude Code binary or API key required),
# see scripts/cron/install-cron.sh.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LA="$HOME/Library/LaunchAgents"
mkdir -p "$LA" "$HERE/logs"

if [ ! -f "$HERE/env.sh" ]; then
  echo "ERROR: $HERE/env.sh not found." >&2
  echo "  cp $HERE/env.sh.example $HERE/env.sh   and fill in ANTHROPIC_API_KEY, CLAUDE_BIN, REPO." >&2
  exit 1
fi

WITH_IMPROVER=0
[ "${1:-}" = "--with-improver" ] && WITH_IMPROVER=1

PLISTS=(com.m7.magazine com.m7.repo-watch)
[ "$WITH_IMPROVER" -eq 1 ] && PLISTS+=(com.m7.improver)

for p in "${PLISTS[@]}"; do
  sed "s|__REPO__|$REPO|g" "$HERE/$p.plist" > "$LA/$p.plist"
  launchctl unload -w "$LA/$p.plist" 2>/dev/null || true
  launchctl load -w "$LA/$p.plist"
  echo "loaded $p"
done

echo
echo "Done. Verify with:  launchctl list | grep com.m7"
if [ "$WITH_IMPROVER" -eq 0 ]; then
  echo "Note: the improver is NOT scheduled. Run /improve-repo by hand a few times to tune its"
  echo "signal/noise, then re-run:  bash scripts/cron/install.sh --with-improver"
fi
echo
echo "Reminder: approve the arxiv MCP server once in an interactive session before the first run,"
echo "or magazine runs will lack it (WebSearch/WebFetch still work). See README.md."

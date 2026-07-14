#!/usr/bin/env bash
# Daily research magazine — invoked by launchd (com.m7.magazine). See scripts/cron/README.md.
# NON-BARE on purpose: bare mode skips skills + hooks, so the harness's guards wouldn't run.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/env.sh"
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
DATE="$(date +%F)"

"$CLAUDE_BIN" -p "/research-magazine" \
  --allowedTools "Skill,Workflow,Agent,Task,Read,Grep,Glob,WebSearch,WebFetch,Write(knowledge/**),Edit(knowledge/**),Bash" \
  --permission-mode dontAsk \
  --output-format json >>"$LOGDIR/magazine-$DATE.json" 2>>"$LOGDIR/magazine-$DATE.err"
STATUS=$?

# Backstop vault backup (the skill also pushes; this is idempotent). Only if a remote exists.
if git -C "$REPO/knowledge" remote get-url origin >/dev/null 2>&1; then
  git -C "$REPO/knowledge" add -A
  git -C "$REPO/knowledge" commit -m "magazine $DATE" >/dev/null 2>&1 || true
  git -C "$REPO/knowledge" push >/dev/null 2>&1 || true
fi

if [ $STATUS -eq 0 ]; then
  osascript -e 'display notification "Today'\''s issue is ready in knowledge/" with title "M7 Research Magazine"' 2>/dev/null || true
else
  osascript -e 'display notification "Magazine run failed — see scripts/cron/logs" with title "M7 Research Magazine (error)"' 2>/dev/null || true
fi
exit $STATUS

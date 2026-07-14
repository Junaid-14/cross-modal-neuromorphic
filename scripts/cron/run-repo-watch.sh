#!/usr/bin/env bash
# Daily upstream repo watch — invoked by launchd (com.m7.repo-watch). See scripts/cron/README.md.
# NON-BARE on purpose (loads skills + hooks). Needs `gh` authenticated for GitHub API reads.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/env.sh"
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
DATE="$(date +%F)"

"$CLAUDE_BIN" -p "/repo-watch" \
  --allowedTools "Skill,Workflow,Agent,Task,Read,Grep,Glob,WebFetch,Write(knowledge/**),Edit(knowledge/**),Bash" \
  --permission-mode dontAsk \
  --output-format json >>"$LOGDIR/repo-watch-$DATE.json" 2>>"$LOGDIR/repo-watch-$DATE.err"
STATUS=$?

if git -C "$REPO/knowledge" remote get-url origin >/dev/null 2>&1; then
  git -C "$REPO/knowledge" add -A
  git -C "$REPO/knowledge" commit -m "repo-watch $DATE" >/dev/null 2>&1 || true
  git -C "$REPO/knowledge" push >/dev/null 2>&1 || true
fi

if [ $STATUS -eq 0 ]; then
  osascript -e 'display notification "Upstream digest ready in knowledge/" with title "M7 Repo Watch"' 2>/dev/null || true
else
  osascript -e 'display notification "Repo-watch run failed — see scripts/cron/logs" with title "M7 Repo Watch (error)"' 2>/dev/null || true
fi
exit $STATUS

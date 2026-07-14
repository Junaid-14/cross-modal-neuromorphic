#!/usr/bin/env bash
# Daily repo improver — invoked by launchd (com.m7.improver). See scripts/cron/README.md.
# REPORT MODE ONLY: runs /improve-repo WITHOUT "apply", so the working tree is never touched
# unattended — it writes a findings report + proposed patches into the vault. NON-BARE on purpose.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/env.sh"
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
DATE="$(date +%F)"

"$CLAUDE_BIN" -p "/improve-repo" \
  --allowedTools "Skill,Workflow,Agent,Task,Read,Grep,Glob,Write(knowledge/**),Edit(knowledge/**),Bash(git rev-parse *),Bash(git log *),Bash(git diff *),Bash(git -C knowledge *),Bash(date *),Bash(find *)" \
  --permission-mode dontAsk \
  --output-format json >>"$LOGDIR/improver-$DATE.json" 2>>"$LOGDIR/improver-$DATE.err"
STATUS=$?

if git -C "$REPO/knowledge" remote get-url origin >/dev/null 2>&1; then
  git -C "$REPO/knowledge" add -A
  git -C "$REPO/knowledge" commit -m "improver $DATE" >/dev/null 2>&1 || true
  git -C "$REPO/knowledge" push >/dev/null 2>&1 || true
fi

if [ $STATUS -eq 0 ]; then
  osascript -e 'display notification "Findings report ready in knowledge/" with title "M7 Improver"' 2>/dev/null || true
else
  osascript -e 'display notification "Improver run failed — see scripts/cron/logs" with title "M7 Improver (error)"' 2>/dev/null || true
fi
exit $STATUS

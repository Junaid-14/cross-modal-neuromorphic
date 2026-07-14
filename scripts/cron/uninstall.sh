#!/usr/bin/env bash
# Remove all M7 launchd jobs.  Usage: bash scripts/cron/uninstall.sh
set -uo pipefail
LA="$HOME/Library/LaunchAgents"
for p in com.m7.magazine com.m7.repo-watch com.m7.improver; do
  if [ -f "$LA/$p.plist" ]; then
    launchctl unload -w "$LA/$p.plist" 2>/dev/null || true
    rm -f "$LA/$p.plist"
    echo "removed $p"
  fi
done
echo "Done. Verify with:  launchctl list | grep com.m7   (should be empty)"

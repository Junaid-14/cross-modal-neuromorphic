# Scheduling — macOS launchd

Unattended scheduling for the recurring workflows. Claude Code has **no native scheduler**, so
`launchd` runs a wrapper script that calls headless `claude -p`, which writes into the `knowledge/`
vault and backs it up.

| Job | Skill | When | Loaded by default? |
|---|---|---|---|
| `com.m7.magazine` | `/research-magazine` | daily 06:00 | ✅ |
| `com.m7.repo-watch` | `/repo-watch` | daily 06:15 | ✅ |
| `com.m7.improver` | `/improve-repo` (report mode) | daily 06:30 | ❌ opt-in |

## Setup (one time)

```bash
# 1. Fill in your environment (API key, claude path, repo path).
cp scripts/cron/env.sh.example scripts/cron/env.sh
$EDITOR scripts/cron/env.sh

# 2. Approve the arxiv MCP server ONCE in an interactive session (project .mcp.json servers need
#    approval; a headless run can't answer the prompt). Just run claude in the repo and use /mcp,
#    or run /research-magazine once interactively. Reset later with: claude mcp reset-project-choices

# 3. Create + push the private vault repo (backup — see knowledge/README.md), then:
bash scripts/cron/install.sh                 # magazine + repo-watch
# later, once /improve-repo is tuned:
bash scripts/cron/install.sh --with-improver

# Verify / remove:
launchctl list | grep com.m7
bash scripts/cron/uninstall.sh
```

## Why non-bare (`claude -p`, not `claude --bare -p`)

`--bare` skips auto-discovery of **skills, hooks, MCP, and auto-memory**. That would mean the
scheduled run couldn't find `/research-magazine`, and — worse — would run with `guard_boundary.py`
and the other guard hooks **disabled**. The whole point of this harness is those guards, so we run
**non-bare**: slightly slower startup, full safety. Each wrapper still pins `--allowedTools` and
`--permission-mode dontAsk` so nothing outside the allowlist can run, and `Write` is scoped to
`knowledge/**` — a scheduled job can only ever write into the vault.

`dontAsk` auto-denies anything not explicitly allowed. This is a feature: it means a scheduled run
physically cannot launch the 4–5 hour training (`guard_long_runs.py` would `ask`, and an `ask`
under `dontAsk` is denied).

## launchd realities

- `StartCalendarInterval` does **not** fire while the Mac is asleep. A missed daily run coalesces
  into a single run on wake — so the magazine is ready when you open the lid. That's intended.
- Auth: `ANTHROPIC_API_KEY` comes from `env.sh` (interactive OAuth won't run under launchd).
- Logs: JSON result + stderr per run in `scripts/cron/logs/` (gitignored). The JSON captures
  `result`, `session_id`, and `total_cost_usd` — that's your cost/audit trail.

## What each run does

1. `claude -p "/<skill>"` → the skill runs its workflow and writes notes into `knowledge/`.
2. Backstop `git -C knowledge add/commit/push` (the skill also pushes; idempotent).
3. `osascript` desktop notification when the output lands (or on failure).

The improver wrapper runs **report mode only** (no `apply`) — it never touches the working tree
unattended; it writes a findings report + proposed patches into `knowledge/60-Patches/`.

#!/usr/bin/env python3
"""PreToolUse(Bash) hook — enforce the repo boundary rule for shell commands.

CLAUDE.md's boundary rule: operate only inside this repo. Never write to sibling
experiment repos (`../spinnaker-mapping/`) or the personal vault (`../research/`)
— they are separate repos with their own instructions and their own history.

`permissions.deny` already blocks the Edit/Write tools from escaping. It cannot
see *inside* a shell command, so `echo x > ../research/notes.md`, `cp -r . ../`,
`sed -i ... ../foo`, and `git -C ../research commit` all slip past it. This hook
closes that gap.

Design: deny only when BOTH conditions hold —
  (a) the command mutates something (a write verb, or a `>` / `>>` redirect), and
  (b) the target resolves outside the repo root and isn't a scratch/temp path.
Read-only commands that reference outside paths (grep, ls, cat, diff) are allowed
through; `permissions.ask` on `Read(../**)` covers the interactive case.

Contract: stdout JSON with hookSpecificOutput.permissionDecision. Exit 0.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env and Path(env).is_dir():
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()

# Commands that create, modify, move or delete files.
WRITE_VERBS = {
    "rm", "rmdir", "mv", "cp", "tee", "touch", "mkdir", "dd", "truncate",
    "ln", "install", "rsync", "shred", "chmod", "chown", "unlink",
}

# Writing outside the repo is fine for these — they're scratch, not content.
SAFE_PREFIXES = tuple(
    p for p in (
        "/dev/null", "/dev/stdout", "/dev/stderr", "/dev/fd",
        "/tmp/", "/private/tmp/", "/var/folders/", "/var/tmp/",
        os.environ.get("TMPDIR") or "\0",
        os.environ.get("CLAUDE_PROJECT_DIR", "\0") + "/",
    )
    if p and p != "\0"
)


def is_outside(token: str) -> bool:
    """True if `token` looks like a path resolving outside the repo root."""
    t = token.strip().strip("\"'")
    if not t or t.startswith("-"):
        return False
    # Only consider things that actually look like paths.
    if not (t.startswith(("/", "../", "~/")) or "/../" in t or t == ".."):
        return False
    if t.startswith(SAFE_PREFIXES):
        return False
    try:
        resolved = Path(os.path.expanduser(t))
        if not resolved.is_absolute():
            resolved = (ROOT / resolved)
        resolved = Path(os.path.normpath(str(resolved)))
    except Exception:
        return False
    if str(resolved).startswith(SAFE_PREFIXES):
        return False
    try:
        resolved.relative_to(ROOT)
        return False  # inside the repo
    except ValueError:
        return True  # outside


def quote_mask(cmd: str) -> list[bool]:
    """True for every character that sits INSIDE a quoted span.

    A `>` inside quotes is data, not a redirect:
        echo '{"cmd": "x > ../research/y"}'   <- not a redirect, must not be denied
        grep -r "../research" .               <- not a write, must not be denied
    But a quoted TARGET is still a real redirect, so we only use this to locate the
    operator; the target itself is parsed from the original string afterwards:
        echo hi > "../research/y"             <- IS a redirect, must be denied
    """
    inside = [False] * len(cmd)
    stack: list = []          # quote state saved across $( ... ) boundaries
    q = None                  # current quote char, or None
    i, n = 0, len(cmd)

    while i < n:
        c = cmd[i]

        # Inside single quotes EVERYTHING is literal: no escapes, no $(), no nesting.
        if q == "'":
            inside[i] = True
            if c == "'":
                q = None
            i += 1
            continue

        if c == "\\" and i + 1 < n:
            inside[i] = inside[i + 1] = q is not None
            i += 2
            continue

        # Command substitution restarts quoting. Without this, the `"` inside
        #   "$(cmd 'a "b > ../x" c')"
        # looks like it CLOSES the outer quote, and the `>` then reads as a redirect.
        if c == "$" and i + 1 < n and cmd[i + 1] == "(":
            inside[i] = inside[i + 1] = q is not None
            stack.append(q)
            q = None
            i += 2
            continue

        if c == ")" and stack:
            q = stack.pop()
            inside[i] = q is not None
            i += 1
            continue

        if c in ("'", '"'):
            inside[i] = True
            if q is None:
                q = c
            elif q == c:
                q = None
            i += 1
            continue

        inside[i] = q is not None
        i += 1

    return inside


def redirect_targets(cmd: str):
    """Yield the target of every UNQUOTED > / >> operator (target may itself be quoted)."""
    inside = quote_mask(cmd)
    i, n = 0, len(cmd)
    while i < n:
        if cmd[i] == ">" and not inside[i]:
            j = i + 1
            while j < n and cmd[j] == ">":
                j += 1
            while j < n and cmd[j] in " \t":
                j += 1
            if j < n and cmd[j] in ("'", '"'):          # quoted target
                q = cmd[j]
                k = cmd.find(q, j + 1)
                if k != -1:
                    yield cmd[j + 1:k]
                    i = k + 1
                    continue
            else:                                        # bare target
                k = j
                while k < n and cmd[k] not in " \t;|&()":
                    k += 1
                if k > j:
                    yield cmd[j:k]
                    i = k
                    continue
        i += 1


def deny(reason: str) -> None:
    sys.stdout.write(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    sys.exit(0)


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        return
    if payload.get("tool_name") != "Bash":
        return

    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not cmd.strip():
        return

    try:
        tokens = shlex.split(cmd, comments=False, posix=True)
    except ValueError:
        tokens = re.split(r"\s+", cmd)  # unbalanced quotes — degrade, don't crash

    # (a) Redirects: `> path` / `>> path`, ignoring any `>` that is merely quoted text.
    for target in redirect_targets(cmd):
        if is_outside(target):
            deny(
                f"Boundary rule: this command redirects output to `{target}`, which is "
                f"OUTSIDE this repository ({ROOT}).\n\n"
                f"CLAUDE.md forbids writing to sibling repos (../spinnaker-mapping/) or the "
                f"personal vault (../research/). Write inside the repo, or use a temp path."
            )

    # (b) Write verbs with an out-of-tree target.
    verb_positions = [i for i, t in enumerate(tokens) if Path(t).name in WRITE_VERBS]
    for i in verb_positions:
        for target in tokens[i + 1:]:
            if target in ("&&", "||", ";", "|"):
                break
            if is_outside(target):
                deny(
                    f"Boundary rule: `{Path(tokens[i]).name}` targets `{target}`, which is "
                    f"OUTSIDE this repository ({ROOT}).\n\n"
                    f"CLAUDE.md forbids operating on sibling repos (../spinnaker-mapping/) or "
                    f"the personal vault (../research/)."
                )

    # (c) `git -C <outside>` — a git write aimed at another repo.
    for i, t in enumerate(tokens[:-1]):
        if t == "-C" and i > 0 and Path(tokens[i - 1]).name == "git":
            if is_outside(tokens[i + 1]):
                deny(
                    f"Boundary rule: `git -C {tokens[i + 1]}` operates on a repository OUTSIDE "
                    f"this one ({ROOT}). Sibling repos have their own history and their own "
                    f"CLAUDE.md — do not touch them from here."
                )

    # (d) `sed -i` / in-place editors aimed out of tree.
    if re.search(r"\bsed\b[^|;&]*\s-i\b", cmd) or re.search(r"\bperl\b[^|;&]*\s-i\b", cmd):
        for t in tokens:
            if is_outside(t):
                deny(
                    f"Boundary rule: an in-place edit targets `{t}`, OUTSIDE this repository "
                    f"({ROOT})."
                )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)

---
description: Track upstream repos adjacent to M7 (snnTorch, Norse, SpikingJelly, BindsNET, Tonic, sPyNNaker/PyNN, …), diff each against our last-seen commit, classify every change mandatory/recommended/optional/ignore for our code, adversarially downgrade the positive verdicts, and publish a digest into the knowledge vault. Use daily (launchd) or to check whether an upstream release affects us.
argument-hint: "[optional: a specific repo to check, else the whole watch-list]"
allowed-tools: Bash(gh *), Bash(git ls-remote *), Bash(date *), Bash(git -C knowledge *), Workflow, WebFetch, Read, Write, Edit, Glob, Grep
---

# Repo watch: `$ARGUMENTS`

Runs the survey → classify → challenge workflow and writes a dated digest into the vault, then
advances each repo's last-seen SHA so tomorrow only sees what's new.

## 1. Run the workflow

```
Workflow(name: "repo-watch")
```

Returns `perRepo` (each repo's changes, classified and challenged), `actionItems` (the surviving
mandatory/recommended), `newLastSeen` (the SHA to record per repo), and a `summary`.

## 2. Get the date

```bash
date +%F
```

## 3. Publish the digest

Copy `knowledge/.templates/repo-watch.md`, fill frontmatter (`date`, `type: repo-watch`,
`source-workflow: repo-watch`, `status: new`), and fill the body from `perRepo` — one section per
repo (SHA range checked, then each relevant change as `[class] summary` with its commit URL and a
"for us:" line), then the action items. Write it to `knowledge/20-Repo-Watch/digest-<date>.md`.

**Every change must cite a real commit SHA/URL.** No changelog prose without a resolvable commit.

## 4. Advance the last-seen state

For each entry in `newLastSeen`, update (or create on first run) the tracking note
`knowledge/20-Repo-Watch/track-<repo>.md` with the repo's slug and the new `last-seen SHA`. This is
what makes tomorrow's run incremental — if you skip it, the next run re-reports everything.

## 5. Update the graph + back up

- Link the digest under `## Digests` in `knowledge/00-MOCs/repo-watch.md` (newest first).
- Surface any `mandatory` change in `00-MOCs/index.md` open actions.
- Write a session note to `knowledge/50-Sessions/repo-watch-<date>.md`.
- `git -C knowledge add -A && git -C knowledge commit -m "repo-watch <date>" && git -C knowledge push`

## Integrity

- Cite the SHA; a changelog is a claim, a commit is the source.
- A dependency change that could move our numerics is **mandatory to know** even if we don't upgrade —
  don't bury it. This is read-only intel: never bump a dependency or edit our code from here.

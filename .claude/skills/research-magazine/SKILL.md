---
description: Build today's research magazine — scan the field across beats, verify every source is real, dedupe against recent issues, curate a ranked digest adapting to your ratings — and publish it into the knowledge vault. Use each morning (or on the launchd schedule) for the latest papers / models / tools / articles relevant to M7.
argument-hint: "[optional: a beat or topic to emphasise today]"
allowed-tools: Bash(date *), Bash(git -C knowledge *), Workflow, WebSearch, WebFetch, Read, Write, Edit, Glob, Grep
---

# Research magazine: `$ARGUMENTS`

Runs the scan → verify → curate workflow and writes today's issue into the knowledge vault.

## 1. Run the workflow

```
Workflow(name: "research-magazine")
```

Returns `beats` (grouped, ranked items), `beatAdjustments`, `dropped`, and a `summary`. Every item
is already source-verified; anything unverifiable is in `dropped`, not the issue.

## 2. Get the date

```bash
date +%F        # e.g. 2026-07-15  → issue path 10-Research-Magazine/2026-07/15.md
```

## 3. Publish the issue

Copy `knowledge/.templates/magazine.md` and fill it from the workflow result:
- Frontmatter: `date` from step 2, `type: magazine`, `source-workflow: research-magazine`, `status: new`.
- Body: the overview line, the `beatAdjustments` note, then each beat with its items (title, source
  URL, "why it matters for M7", relevance, and an editable `- rating:` line for you to set), then the
  Dropped section.

Write it to `knowledge/10-Research-Magazine/<YYYY-MM>/<DD>.md`.

**Never add an item that isn't in the workflow's verified `beats`.** No source without a real URL.
Numbers from others' papers are attributed to them, never presented as ours.

## 4. Update the graph

- Add a newest-first link to the issue under `## Issues` in `knowledge/00-MOCs/research-magazine.md`.
- Refresh the `## Today` section of `knowledge/00-MOCs/index.md` to link today's issue.

## 5. Back up the vault

```bash
git -C knowledge add -A && git -C knowledge commit -m "magazine <date>" && git -C knowledge push
```

## The feedback loop

As you read, set each item's `rating:` to `keep | meh | skip`. Tomorrow's run reads the last 14 days
of those per-item ratings, grouped by beat, and boosts kept beats / trims skipped ones — it reports
what it changed in `beatAdjustments`. You don't do anything except rate.

The scheduled wrapper (`scripts/cron/run-magazine.sh`) runs this skill and then fires a macOS
notification when the issue lands.

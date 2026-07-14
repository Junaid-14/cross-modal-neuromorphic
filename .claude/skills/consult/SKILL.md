---
description: Statistics consultant for a completed M7 run. Analyses the CL results CSV (mean±std over unique seeds, effect sizes, seed-count sanity, gate trajectory), audits every number against the integrity rules, and publishes a report into the knowledge vault. Use after a model run, when asked about accuracy/forgetting/FWT/significance, or before any result number is written down.
argument-hint: "[optional: path to a specific results CSV, else auto-detected]"
allowed-tools: Bash(python3 *), Bash(find *), Bash(date *), Bash(git -C knowledge *), Workflow, Read, Write, Edit, Glob, Grep
---

# Consult the statistician: `$ARGUMENTS`

Runs the deep statistics + integrity audit and writes a dated report into the knowledge vault.
This layers on `/analyze-cl-results` (same canonical loader + dedup gates); the consultant adds the
statistics, the adversarial number-audit, and the vault write-up.

## 1. Run the analysis workflow

Invoke the `stats-consult` workflow via the **Workflow** tool:

```
Workflow(name: "stats-consult")
```

It returns either `dataState: "NO_RESULTS_YET"` (no CSV on disk — the expected state today) or a
full analysis with `perCondition`, `statistics`, `gateAnalysis`, `optimisationIdeas`,
`integrityFlags`, and `confirmedNumbers`.

## 2. Get today's date (workflows can't read the clock)

```bash
date +%F
```

## 3. Publish the report into the vault

- If `dataState` is `NO_RESULTS_YET`: still write a short report to
  `knowledge/30-Consultant/report-<date>.md` recording that there is no CSV and every number in
  `docs/`/`paper/` is therefore UNSOURCED. Do **not** invent numbers to fill the template.
- Otherwise: copy `knowledge/.templates/consultant.md`, fill the frontmatter (`date` from step 2,
  `type: consultant`, `source-workflow: stats-consult`, `status: new`), and fill the body from the
  workflow result. **The integrity gate goes first**, before any headline number. Every number in
  the results table must be one of `confirmedNumbers`; anything in `integrityFlags` is reported as a
  flag, never as a result.

Write it to `knowledge/30-Consultant/report-<date>.md`.

## 4. Update the graph

- Add a newest-first link to the report under `## Reports` in `knowledge/00-MOCs/results.md`.
- Refresh the "Today" / "Open actions" sections of `knowledge/00-MOCs/index.md` if any integrity
  flag needs your attention.
- Write a session note to `knowledge/50-Sessions/stats-consult-<date>.md` from
  `knowledge/.templates/session.md` (what ran, what it produced, any failed approaches).

## Integrity — non-negotiable

- **RULE 1** — every reported number is a `confirmedNumbers` entry (traces to a CSV column,
  aggregated over unique seeds). No estimates.
- **RULE 4** — test accuracy, never validation (SHD val/test gap is 15–19 pp).
- **RULE 5** — forgetting is ~0 pp; never "100 % reduction".
- **Accuracy never travels alone** — flag that the CSV has no spike-rate/sparsity/timestep column,
  so cost metrics are UNSOURCED by construction.

For a raw, quick load without the statistics layer, use `/analyze-cl-results`. For a whole-corpus
number trace across paper + docs, use the `results-audit` workflow.

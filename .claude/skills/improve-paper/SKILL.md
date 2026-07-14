---
description: Improve the correspondence paper — assess structure/clarity, trace every number to a CSV, verify every citation, hunt overclaims — then publish a report, refresh the citation ledger, and export a NotebookLM brief into the knowledge vault. Proposes edits; never edits the .tex unattended. Use when revising paper/cross_modal_correspondence.tex or before a submission.
argument-hint: "[optional: a section or concern to focus on]"
allowed-tools: Bash(find *), Bash(grep *), Bash(date *), Bash(git -C knowledge *), Workflow, WebSearch, WebFetch, Read, Write, Edit, Glob, Grep
---

# Improve the paper: `$ARGUMENTS`

Runs the deep paper audit and writes its output into the knowledge vault as proposals. **It never
edits `paper/cross_modal_correspondence.tex`** — you apply changes by hand after reading the report.

## 1. Run the workflow

```
Workflow(name: "paper-improver")
```

Returns `structureIssues`, `numberTrace`, `citeCheck`, `overclaimFlags`, and a `summary`.

## 2. Get today's date

```bash
date +%F
```

## 3. Publish into the vault

Write three artifacts under `knowledge/40-Paper/`:

1. **`report-<date>.md`** — from `knowledge/.templates/paper.md`. Number trace first (expect
   UNSOURCED across the board — the correspondence `results/` tree is absent), then citations, then
   overclaim flags, then the proposed edits (section · current · proposed · rationale). Fill
   frontmatter (`date`, `type: paper`, `source-workflow: paper-improver`, `status: new`).

2. **`citation-ledger.md`** — one row per `\cite` key: REAL / UNVERIFIABLE / FABRICATED, BibTeX if
   real, and whether it still carries `\todo{verify}`. Update in place if it already exists.

3. **`notebooklm-brief-<date>.md`** — a clean, self-contained markdown synthesis of the paper's
   argument + current status, suitable to drop into NotebookLM. **Anonymised** — no author names,
   no institutions (RULE 3). This is a reading aid, not a results source: mark every number with its
   trace verdict (UNSOURCED/SOURCED), never launder an unsourced number into it as fact.

## 4. Update the graph

- Link `report-<date>.md` under `## Reports` in `knowledge/00-MOCs/paper.md` (newest first).
- Surface any `overclaim-anonymity-leak` or `FABRICATED` citation in `00-MOCs/index.md` open actions.
- Write a session note to `knowledge/50-Sessions/paper-improver-<date>.md`.

## Integrity — non-negotiable

- **RULE 1** — no number without a CSV; the correspondence results are absent, so say UNSOURCED.
- **Anonymity** — self-cite `\citep{anon2026modality}`; never fill author `\todo{}`s; keep names
  out of the brief.
- **Two narratives** — never imply an emergent modality-invariant semantic space; this paper's data
  refutes it. Keep it separate from the FWY≈75 pp forward-transfer story.
- Surface the **IEEEtran-vs-NeurIPS** class mismatch and the inline-`thebibliography` (no `.bib`)
  reality as proposals — do not silently change either.

For a whole-corpus number sweep across paper + docs, use the `results-audit` workflow. For citation
verification alone, the `literature-scout` and `paper-fact-checker` agents.

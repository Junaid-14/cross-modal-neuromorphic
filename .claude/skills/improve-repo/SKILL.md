---
description: Sweep the repo for bugs, quality issues, and architecture problems and PROPOSE fixes — fan out by dimension, adversarially verify, dedupe against prior runs, then publish a findings report + proposed patches into the knowledge vault. Never edits code unattended; never proposes a silent numeric change. Add "apply" to stage confirmed non-numeric fixes on a draft branch for review.
argument-hint: "[apply] — omit for report+patches (safe/default); 'apply' to stage fixes on a draft branch"
allowed-tools: Bash(git rev-parse *), Bash(git log *), Bash(git worktree *), Bash(git switch *), Bash(git checkout *), Bash(git branch *), Bash(git diff *), Bash(git -C knowledge *), Bash(date *), Bash(find *), Workflow, Read, Write, Edit, Glob, Grep
---

# Improve the repo: `$ARGUMENTS`

Finds real issues and proposes fixes. **Default mode is propose-only** — it writes a report and
proposed patches into the vault and does not touch the working tree. Pass `apply` to stage the
confirmed, numeric-neutral fixes on a draft branch for your review.

## 1. Run the workflow

```
Workflow(name: "repo-improver")
```

Returns `baseSha`, a `summary`, ranked `findings` (each with a required `failureScenario` and a
`numericRisk`), and `numericMoversNeedExplicitReview`.

## 2. Get today's date

```bash
date +%F
```

## 3. Publish (always — both modes)

- **Findings report** → `knowledge/50-Sessions/repo-improver-<date>.md` from
  `knowledge/.templates/session.md`: ranked findings (most-severe first), each with file:line,
  category, severity, failure scenario, and proposed fix. Call out
  `numericMoversNeedExplicitReview` in their own section — these are **never** auto-applied.
- **Proposed patches** → one file per confirmed finding in `knowledge/60-Patches/<slug>-<date>.patch`.
  Put the **base SHA** (`baseSha` from the workflow) in a header comment at the top of each patch, so
  a later run can tell if the base has moved. Show the change as a clear before/after diff.
- Surface any `blocking`/`high` finding in the `## Open actions` of `knowledge/00-MOCs/index.md`.

## 4. Apply mode (only if `$ARGUMENTS` contains `apply`)

Do **not** touch the current working tree. Use a worktree **inside** the repo (an outside path like
`../.improver-worktree` is blocked by `guard_boundary.py`). `.worktrees/` is gitignored:

```bash
git worktree add .worktrees/improver-<date> -b improver/<date>
```

Apply **only** findings whose `numericRisk` is `none` or `flagged-safe` inside that worktree (the
`py_syntax_check` hook validates each edit automatically). **Never apply a `flagged-changes-numerics`
finding** — leave those in the report for a human. Then:

```bash
git -C .worktrees/improver-<date> diff
```

Report the branch name and the diff for review. Never commit to `main`; never merge.

## 5. Back up the vault

```bash
git -C knowledge add -A && git -C knowledge commit -m "improver <date>" && git -C knowledge push
```

## Integrity — non-negotiable

- **Never silently change numerics.** A fix that could move a result is flagged and reported, never
  applied. This is the whole reason the improver is propose-first.
- **Never propose touching** `*.pt`, `**/results/**/*.csv`, or `ICONS_M7/**` (also deny-blocked).
- A finding without a concrete failure scenario is a quality note at most, not a bug.

For silent-numeric-drift review of a specific change you're about to commit, use the
`snn-code-reviewer` agent. For doc-vs-code drift alone, the `doc-code-reconcile` workflow.

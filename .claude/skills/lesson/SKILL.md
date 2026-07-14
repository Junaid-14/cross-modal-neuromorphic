---
description: Record a correction into the self-correction ledger so a workflow never repeats a mistake. Use whenever a workflow (magazine, repo-watch, improver, consultant, paper-writer) got something wrong and you've told it the fix — this makes the fix stick across future runs. Also use when the user says "don't do that again" / "that was wrong because…".
argument-hint: "<what went wrong → the correction>  [workflow: research-magazine|repo-watch|repo-improver|stats-consult|paper-improver|shared]"
allowed-tools: Bash(date *), Bash(git -C knowledge *), Read, Write, Edit, Glob, Grep
---

# Record a lesson: `$ARGUMENTS`

Turns a one-off correction into a **binding** rule that every future run of the affected workflow
will read and apply. This is the mechanism that makes the automation self-correcting.

## 1. Decide the scope

From `$ARGUMENTS` (and the recent conversation), determine which ledger this belongs in:
- A specific workflow → `knowledge/01-Lessons/<workflow>.md`
  (`research-magazine` · `repo-watch` · `repo-improver` · `stats-consult` · `paper-improver`).
- Applies to all workflows → `knowledge/01-Lessons/_shared.md`.

If unsure, ask which workflow it binds; don't guess it into the wrong ledger.

## 2. Get the date

```bash
date +%F
```

## 3. Append the entry (newest first)

If the target ledger doesn't exist yet, create it with lessons frontmatter (`type: lessons`,
`source-workflow: <workflow>`, `status: new`, `links: ["[[index]]"]`). Then add, **above** any
existing entries:

```markdown
## <date> — <short title>
- **Mistake:** <what the run did wrong, concretely>
- **Correction:** <what it should do instead>
- **How to apply:** <the specific, checkable rule a future run can follow>
- **Scope:** <which workflow(s)>
```

Write a real, specific rule — "reject arXiv IDs not matching NNNN.NNNNN", not "be careful".

## 4. Integrity check before saving

A lesson may only **tighten** behaviour. If the correction would weaken an integrity rule
(RULE 1–5, anonymity, never-silently-change-numerics, cite-only-real-sources), **do not record it** —
say why instead. Those rules are not overridable by the ledger.

## 5. Back up

```bash
git -C knowledge add -A && git -C knowledge commit -m "lesson: <short title>" && git -C knowledge push
```

The next run of the affected workflow reads this automatically (its agents load `_shared.md` +
`<workflow>.md` before starting), so the fix is applied without any further action from you.

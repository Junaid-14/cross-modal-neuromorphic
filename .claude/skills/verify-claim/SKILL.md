---
description: Trace a specific number or factual claim back to the file that produced it, or declare it unsourced. Use before writing any number into the paper, docs, a figure caption, a commit message or a summary — and whenever someone asks "where does this number come from?", "is this real?", or "can we cite this?".
argument-hint: "<the number or claim, e.g. 82.83 or 'FWT 75.11 pp'>"
allowed-tools: Bash(python3 *), Bash(find *), Bash(grep *), Read, Grep, Glob
---

# Verifying a claim: `$ARGUMENTS`

This enforces **RULE 1 — never put a number anywhere that is not in a CSV. No estimates.**

## Start from what data exists

```bash
find . -name '*.csv' -not -path './.git/*' -not -path '*/.venv/*'
find . \( -name '*.pt' -o -name '*.pth' \) -not -path './.git/*'
```

**As of the last check, both return nothing.** There is no results CSV and there are no
checkpoints anywhere in this repo.

If that is still true, then the verdict for `$ARGUMENTS` — and for every other number quoted
in `docs/` and `paper/` — is **UNSOURCED**, and you should say so plainly. Do not soften it,
and do not go hunting for a rationalisation.

## Then find where the claim is asserted

```bash
grep -rn "$ARGUMENTS" --include='*.md' --include='*.tex' --include='*.py' . \
  | grep -v '/.git/' | grep -v '/.venv/'
```

**Appearing in several documents is not corroboration.** It is the same unsourced number,
copied. Only a CSV counts as a source.

## If a CSV does exist

Use the canonical loader — never read the CSV raw:

```bash
python3 .claude/skills/analyze-cl-results/scripts/load_results.py
```

Then check that the claim actually matches: the right **condition** (`replay` /
`no_replay`), the right **seeds**, the right **split** (test, not validation), and the right
**experiment**. Three different result sets circulate in the docs — tri-modal 3-seed,
two-task from the onboarding brief, and single-task from the correspondence paper. Quoting
one as another is a **CONTRADICTED** verdict, not a rounding difference.

## Verdicts

| Verdict | Meaning |
|---|---|
| **SOURCED** | Found in a CSV. Cite the file, column, row filter, and which seeds/conditions were aggregated. |
| **UNSOURCED** | Nothing on disk produces this number. Say so. Do not speculate about where it "probably" came from. |
| **CONTRADICTED** | A file on disk produces a **different** value. Report both. |

## Claims that are unsourced by construction

- **Anything resting on a figure generated before 2026-07-14.** Figures 3 and 4 were hardcoded
  from literal arrays then, and Figure 1 carried reference lines lifted from the legacy
  bi-modal paper. Fixed since — but a figure not regenerated is still fabricated.
- **Any "3-seed" number produced before 2026-07-14.** The code ran `SEEDS = [123, 456]` until
  then, so seed 42 never ran. Such a claim is a 2-seed number regardless of its label, and is
  **CONTRADICTED** unless the result was regenerated after the fix.
- **Any SpiNNaker energy or hardware-performance number.** Nothing has run on a board;
  `docs/HARDWARE_ACCESS_CHECKLIST.md` is entirely unchecked. A simulator cannot produce
  energy figures.

## Output

State the verdict first, then the evidence. If it's unsourced, say what would have to be run
to source it. Do not end on reassurance — an honest "we cannot currently support this number"
is the useful answer.

For a full sweep of every claim across the paper and docs, use the workflow instead:
`.claude/workflows/results-audit.js`.

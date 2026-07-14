# Research integrity — the five critical rules

These come from the team onboarding brief. They are not style preferences. They exist
because this group has already been burned by each of them.

## RULE 1 — Never put a number anywhere that is not in a CSV. No estimates.

Applies to the paper, the docs, figures, commit messages, and **anything you say in
chat**. If you cannot point to the file and column a number came from, you do not have
the number. Say "unsourced" or "I don't know."

This is the rule that makes the whole project trustworthy. Violating it once poisons
every other number by association.

> Right now there is **no results CSV in this repo at all**. So every number currently
> quoted in `docs/` and `paper/` is, in-repo, unsourced. Treat all of them as claims to
> be verified, not facts to be repeated. Use `/verify-claim` or the `results-auditor`
> agent before repeating any of them.

## RULE 2 — Back up to HuggingFace after every seed.

They have already lost data once to a terminated cloud instance. Checkpoints and results
are gitignored, so git will not save you. A run that finishes and isn't backed up is a
run that might as well not have happened.

## RULE 3 — Anonymous submission.

NeurIPS 2026 is double-blind. Self-cite as `\citep{anon2026modality}`. Do not put author
names, institutions, or identifying URLs into `paper/`. Do not "helpfully" fill in the
`\todo{Author names}` placeholders.

## RULE 4 — Report **test** accuracy, never validation.

SHD's validation/test gap runs **15–19 percentage points**. Quoting a val number as if it
were a test number would inflate the headline result by more than the effect being
claimed. When you read an accuracy out of code or a log, confirm which split it came from
before it goes anywhere.

## RULE 5 — The forgetting claim is ~0 pp. It is NOT "100 % reduction."

The honest claim is: **replay enables gate specialisation.** Forgetting is near zero in
absolute terms. "Eliminates catastrophic forgetting" and "100 % reduction" are
overstatements that will not survive review.

---

## Hypotheses are not results

Especially hardware-similarity claims. "This should map to SpiNNaker" is a hypothesis.
"This runs on SpiNNaker within ±1 % of the PyTorch baseline" is a result — and requires
having actually run it. Never promote the former to the latter, in prose or in tone.

Mark hypotheses as hypotheses. `\todo{verify}` exists for this. Use it.

## Accuracy never travels alone

Any accuracy figure must be reported alongside its **spike rate / sparsity** and
**timestep count**. An SNN accuracy number without its cost metrics is not a meaningful
result — it's the part of the result that flatters us.

## Never fabricate

Numbers, citations, hardware specs, dataset statistics. If you don't know, say so. A
plausible-sounding invented citation is worse than an admitted gap, because it will be
caught by a reviewer instead of by us.

## Never silently change numerics

Refactors must be numerically inert unless the change *is* the point — and then it must
be called out explicitly, with before/after. Reordering operations, changing a dtype,
swapping a reduction, or "cleaning up" a seed call can all move results. If a change
could move a number, say so before making it.

## Show a diff before overwriting

Never overwrite an existing file without showing the diff and waiting for confirmation.

## Boundary rule

Operate only inside this repository. Never write to sibling repos
(`../spinnaker-mapping/`) or the personal vault (`../research/`). They have their own
history and their own instructions. This is enforced by `permissions.deny` and by
`.claude/hooks/guard_boundary.py`, but the rule matters more than the enforcement.

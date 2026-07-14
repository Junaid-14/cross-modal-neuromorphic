---
paths:
  - "paper/**"
  - "docs/**/*.md"
  - "**/*.tex"
  - "**/*.bib"
---

# Writing about this work

## Every number traces to a CSV, or it does not go in

RULE 1, restated because this is where it gets violated: **never put a number in the paper
that is not in a CSV file. No estimates.**

Before writing any figure, table entry, or in-text number:
1. Name the CSV and the column it came from.
2. If you cannot, write `\todo{verify}` — do not write the number.

Right now **there is no results CSV in this repo**, so every number currently in `docs/`
and `paper/` is unsourced in-repo. Do not propagate them into new text as though they were
established. Run `/verify-claim <number>` or dispatch the `results-auditor` agent.

`paper/cross_modal_correspondence.tex` already has the right mechanism: a per-number
provenance block at the top and a `\todo{}` macro that renders in red. **Use it.** An
honest red `[TODO: verify]` is infinitely better than a confident wrong number.

## Never fabricate a citation

Not the authors, not the venue, not the year, not the arXiv ID. If a claim needs a citation
you don't have, mark it `\todo{cite}`. Use the `literature-scout` agent or the arXiv MCP
server to find a *real* one and verify it resolves.

There is **no `.bib` file in this repo.** `paper/cross_modal_correspondence.tex` uses an
inline `thebibliography` with most entries tagged `\todo{verify}`. Do not invent BibTeX to
fill it.

## Anonymity (RULE 3)

NeurIPS 2026 is double-blind.
- Self-cite as `\citep{anon2026modality}`.
- Leave `\todo{Author names}` alone. Do not "helpfully" fill in author names, institutions,
  emails, or a GitHub URL that identifies the group.

## Calibrate the claims

| Don't write | Write |
|---|---|
| "eliminates catastrophic forgetting" | "forgetting is ~0 pp" |
| "100 % reduction in forgetting" | "replay enables gate specialisation" (the honest claim) |
| "validated on SpiNNaker" | "validated on the sPyNNaker simulator" — unless it ran on a board |
| "emergent shared semantic space" | never — the correspondence result says the opposite |
| "97 % accuracy" | "97 % test accuracy at T=25 timesteps, spike rate X" |

- **Hypotheses are not results.** Hardware-similarity claims are hypotheses until a board
  has run the model. Phrase them that way.
- **Accuracy always travels with cost.** Spike rate / sparsity + timestep count, every time.
- **Test accuracy, not validation.** SHD's val/test gap is 15–19 pp.

## Keep the two narratives distinct

The onboarding brief sells forward transfer as positive (FWT ≈ 75 pp). The correspondence
paper concludes cross-modal correspondence is **engineered, not emergent** (held-out at
chance, Procrustes 0 %, Mantel r = 0.051). Both can be true — they measure different things
— but a summary that blurs them into "the network learned a shared cross-modal
representation" is an overclaim the team's own data refutes. See
`.claude/rules/contested-facts.md` §6.

## Figures

`code/generate_figures_3seed.py` **used to hardcode Figures 3 and 4** from literal arrays, and
drew Fig 1 reference lines lifted from the legacy bi-modal paper. Fixed 2026-07-14: Fig 3 now
plots real data or is skipped; Fig 4 is derived from CONFIG with the arithmetic printed.
**Any figure not regenerated since may still be fabricated — do not build an argument on it.**

## Docs and code are edited together

If a hyperparameter changes in `code/`, update `docs/EXPERIMENTAL_MANIFEST.md` in the same
change. The drift between these two is what motivated this whole harness — don't restart it.

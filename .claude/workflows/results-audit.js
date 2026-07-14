export const meta = {
  name: 'results-audit',
  description: 'Trace every numeric claim in paper/ and docs/ back to a CSV, or mark it unsourced. Enforces RULE 1 at scale.',
  whenToUse: 'Before a submission, before sharing results, or whenever you need to know which numbers in this project are actually defensible.',
  phases: [
    { title: 'Inventory', detail: 'establish what data actually exists on disk' },
    { title: 'Extract', detail: 'pull every numeric claim out of each document' },
    { title: 'Trace', detail: 'one tracer per claim → SOURCED / UNSOURCED / CONTRADICTED' },
    { title: 'Refute', detail: 'adversarially attack every SOURCED verdict' },
  ],
}

// RULE 1: "Never put a number in the paper that is not in a CSV file. No estimates."
// This workflow executes that rule across the whole corpus.

const CLAIMS_SCHEMA = {
  type: 'object',
  properties: {
    claims: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          value: { type: 'string', description: 'the number exactly as written, e.g. "82.83 ± 1.96"' },
          metric: { type: 'string', description: 'what it claims to measure, e.g. "DVS-Gesture test accuracy, +replay"' },
          file: { type: 'string' },
          line: { type: 'integer' },
          quote: { type: 'string', description: 'the sentence it appears in' },
        },
        required: ['value', 'metric', 'file', 'quote'],
      },
    },
  },
  required: ['claims'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['SOURCED', 'UNSOURCED', 'CONTRADICTED'] },
    evidence: { type: 'string', description: 'the CSV path + column + row filter, OR why nothing on disk produces it' },
    onDiskValue: { type: 'string', description: 'if CONTRADICTED, what the data actually says' },
  },
  required: ['verdict', 'evidence'],
}

const REFUTATION_SCHEMA = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean', description: 'true if the SOURCED verdict does NOT hold up' },
    why: { type: 'string' },
  },
  required: ['refuted', 'why'],
}

const GROUND_RULES = `
This is the M7 tri-modal continual-learning repo (NeurIPS 2026). Its first rule is:
"Never put a number in the paper that is not in a CSV file. No estimates."

Facts you must apply, and must re-verify rather than assume:
- Results live in cl_trimodal_results.csv (typically under ICONS_M7/results/). LAST CHECK: NO
  SUCH FILE EXISTS anywhere in the repo, and there are ZERO .pt checkpoints. Re-check with
  \`find . -name '*.csv' -not -path './.git/*' -not -path '*/.venv/*'\`. If it is still empty,
  then EVERY number in docs/ and paper/ is UNSOURCED. Say so; do not soften it.
- The CSV holds TWO schemas: incremental rows have a \`step\`; summary rows do not. Reruns
  append duplicates. The only valid population is df[df['step'].isna()].drop_duplicates() —
  and even that is exact-match, so near-duplicate reruns survive and inflate n. Use the
  canonical loader: .claude/skills/analyze-cl-results/scripts/load_results.py
- code/generate_figures_3seed.py USED TO hardcode Figures 3 and 4 from literal arrays (the
  gate trajectory; memory [80,150,4] MB) and drew Fig 1 reference lines at y=97.68 / y=82.16
  lifted from the LEGACY bi-modal paper. All three were removed on 2026-07-14: Fig 3 now plots
  gate_trajectory.csv or is SKIPPED, Fig 4 is derived from CONFIG. **Any figure or claim
  produced before that date may still rest on fabricated values — treat as UNSOURCED until
  regenerated.**
- code/m7_trimodal_run.py SEEDS is now [42, 123, 456] (fixed 2026-07-14; was [123, 456]).
  RE-CHECK IT. Any pre-existing "3-seed" claim was produced by the OLD 2-seed code, so it is
  CONTRADICTED unless it was regenerated after the fix.
- Nothing has run on SpiNNaker hardware. Any energy or hardware-performance number is
  UNSOURCED; a simulator cannot produce energy figures.
- Appearing in several documents is NOT corroboration. It is the same unsourced number, copied.
  Only a CSV counts as a source.
`

// ---------------------------------------------------------------------------
phase('Inventory')

const inventory = await agent(
  `${GROUND_RULES}

Establish TWO things and report them plainly.

(1) What result data actually exists on disk. Run:
    find . -name '*.csv' -not -path './.git/*' -not -path '*/.venv/*'
    find . \\( -name '*.pt' -o -name '*.pth' \\) -not -path './.git/*'
    python3 .claude/skills/analyze-cl-results/scripts/load_results.py

(2) The list of documents that make numeric claims — every .md and .tex under docs/, paper/,
    and the repo root. Use: find docs paper . -maxdepth 2 \\( -name '*.md' -o -name '*.tex' \\) -not -path './.git/*'

Return a plain-text report: the data inventory first (be explicit if it is empty), then a
newline-separated list of the document paths worth auditing.`,
  { label: 'inventory', phase: 'Inventory' },
)

log('Data inventory established. Fanning out over documents.')

// One line per document path. Filter to things that look like real files.
const docs = String(inventory)
  .split('\n')
  .map((s) => s.trim())
  .filter((s) => /\.(md|tex)$/i.test(s) && !s.includes(' '))
  .map((s) => s.replace(/^\.\//, ''))

const targets = [...new Set(docs)]
log(`${targets.length} documents to audit.`)

// ---------------------------------------------------------------------------
// Pipeline, not a barrier: each document's claims start getting traced as soon as
// that document is extracted — no waiting for the slowest extractor.
const audited = await pipeline(
  targets,

  // Stage 1 — extract every numeric claim from one document.
  (path) =>
    agent(
      `${GROUND_RULES}

Read ${path} and extract EVERY numeric claim it makes about results: accuracies, forgetting,
FWT, spike rates, sparsity, parameter counts, memory footprints, core counts, energy figures,
timings, dataset sizes.

Include the number exactly as written, what it claims to measure, the file, the line, and the
sentence it sits in. Do NOT evaluate them yet — just extract. If the document makes no numeric
claims, return an empty list.`,
      { label: `extract:${path}`, phase: 'Extract', schema: CLAIMS_SCHEMA },
    ),

  // Stage 2 — trace each claim from that document, concurrently.
  (extracted, path) =>
    parallel(
      (extracted?.claims ?? []).map((claim) => () =>
        agent(
          `${GROUND_RULES}

Trace this single claim to its source, or establish that it has none.

  value:  ${claim.value}
  metric: ${claim.metric}
  where:  ${claim.file}${claim.line ? ':' + claim.line : ''}
  quote:  "${claim.quote}"

Search the repo for data that produces this number. Check that the condition (replay /
no_replay), the seeds, the split (TEST, not validation — SHD's val/test gap is 15-19pp), and
the experiment all actually match the claim.

Three different result sets circulate in these docs — tri-modal 3-seed, two-task (onboarding
brief), and single-task (correspondence paper). Quoting one as another is CONTRADICTED, not a
rounding difference.

Return SOURCED / UNSOURCED / CONTRADICTED with concrete evidence. Do not speculate about where
a number "probably" came from — if nothing on disk produces it, it is UNSOURCED.`,
          { label: `trace:${claim.value}`, phase: 'Trace', schema: VERDICT_SCHEMA },
        ).then((v) => ({ ...claim, ...v })),
      ),
    ),
)

const findings = audited.flat().filter(Boolean)
const sourced = findings.filter((f) => f.verdict === 'SOURCED')
log(`${findings.length} claims traced. ${sourced.length} came back SOURCED — now attacking those.`)

// ---------------------------------------------------------------------------
// Adversarial pass. A SOURCED verdict is the only one that can do damage if wrong:
// an incorrectly-UNSOURCED claim just gets re-checked, but an incorrectly-SOURCED claim
// ships. So we only spend refutation budget on SOURCED, and we prompt to REFUTE.
phase('Refute')

const refuted = await parallel(
  sourced.map((f) => () =>
    agent(
      `${GROUND_RULES}

A previous agent claims this number IS sourced. Your job is to REFUTE that. Default to
refuted=true unless the provenance is airtight.

  claim:    ${f.value} — ${f.metric}
  asserted: ${f.evidence}

Attack it:
- Does the cited file actually exist, and does it actually contain that value?
- Was the dedup applied correctly? An exact-match drop_duplicates() leaves near-duplicate
  rerun rows in place, which inflates n and corrupts the ± std.
- Do the seeds match the claim? Any claim predating the 2026-07-14 seed fix was generated
  with SEEDS = [123, 456] and is a 2-seed number, whatever it is labelled.
- Is it TEST accuracy, or validation quietly reported as test?
- Does it rest on a figure generated before 2026-07-14? Figs 3 & 4 were hardcoded then, and
  Fig 1 carried fabricated reference lines — automatically refuted unless regenerated.
- Is it from a DIFFERENT experiment than the sentence implies?

Return refuted=true if the sourcing does not hold up, with the specific reason.`,
      { label: `refute:${f.value}`, phase: 'Refute', schema: REFUTATION_SCHEMA },
    ).then((r) => ({ ...f, survived: !r.refuted, refutation: r.why })),
  ),
)

const confirmed = refuted.filter(Boolean).filter((f) => f.survived)
const overturned = refuted.filter(Boolean).filter((f) => !f.survived)

return {
  summary: {
    documentsAudited: targets.length,
    claimsFound: findings.length,
    confirmedSourced: confirmed.length,
    overturnedOnRefutation: overturned.length,
    unsourced: findings.filter((f) => f.verdict === 'UNSOURCED').length,
    contradicted: findings.filter((f) => f.verdict === 'CONTRADICTED').length,
  },
  confirmedSourced: confirmed,
  overturnedOnRefutation: overturned,
  unsourced: findings.filter((f) => f.verdict === 'UNSOURCED'),
  contradicted: findings.filter((f) => f.verdict === 'CONTRADICTED'),
  dataInventory: String(inventory).slice(0, 2000),
}

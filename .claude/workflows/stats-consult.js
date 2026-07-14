export const meta = {
  name: 'stats-consult',
  description: 'Statistics consultant for a completed M7 run: load the CL results CSV, compute defensible stats over unique seeds, and adversarially audit every number before it can be quoted.',
  whenToUse: 'After a model run produces results, when you need mean±std / effect sizes / a seed-count sanity check, or before any accuracy/forgetting/FWT number is written down.',
  phases: [
    { title: 'Load', detail: 'establish what result data exists, via the canonical loader only' },
    { title: 'Analyse', detail: 'statistics over unique seeds + gate trajectory + optimisation ideas' },
    { title: 'Audit', detail: 'adversarially attack every number before it is reported' },
  ],
}

// Why this exists: turning cl_trimodal_results.csv into numbers is a minefield — a dual-schema
// CSV, exact-match dedup that inflates n, a seed list that was silently 2-seed until 2026-07-14,
// a 15-19pp val/test gap on SHD, and no spike-rate/timestep column at all. This workflow does the
// statistics AND refuses to produce numbers the data can't support (RULE 1).

const LOAD_SCHEMA = {
  type: 'object',
  properties: {
    csvPresent: { type: 'boolean', description: 'true if a results CSV actually exists on disk' },
    csvPath: { type: 'string', description: 'path to the CSV, or "" if none' },
    uniqueSeeds: { type: 'array', items: { type: 'string' }, description: 'distinct seeds after dedup' },
    conditions: { type: 'array', items: { type: 'string' }, description: 'e.g. ["no_replay","replay"]' },
    gateCsvPresent: { type: 'boolean', description: 'true if gate_trajectory.csv exists' },
    rowSchemaNotes: { type: 'string', description: 'summary rows vs incremental (step) rows, and any near-duplicate inflation seen' },
    rawSummary: { type: 'string', description: 'the loader output, trimmed' },
  },
  required: ['csvPresent', 'csvPath', 'uniqueSeeds', 'conditions', 'gateCsvPresent'],
}

const STATS_SCHEMA = {
  type: 'object',
  properties: {
    perCondition: {
      type: 'array',
      description: 'one row per metric, aggregated over UNIQUE seeds (test accuracy)',
      items: {
        type: 'object',
        properties: {
          metric: { type: 'string', description: 'e.g. "T3_after_T3 (DVS test acc)"' },
          no_replay: { type: 'string', description: 'mean ± std over unique seeds, or "—"' },
          replay: { type: 'string' },
          delta: { type: 'string', description: 'replay − no_replay, with direction' },
          csvColumn: { type: 'string', description: 'the exact CSV column this came from' },
        },
        required: ['metric', 'csvColumn'],
      },
    },
    statistics: { type: 'string', description: 'dispersion, effect sizes, and an honest power caveat for n=3 seeds' },
    gateAnalysis: { type: 'string', description: 'from gate_trajectory.csv, or "file absent — claim unsourced"' },
    optimisationIdeas: { type: 'array', items: { type: 'string' }, description: 'grounded in what the data shows' },
    numbersToAudit: {
      type: 'array',
      description: 'every specific number this analysis would put in front of the user',
      items: {
        type: 'object',
        properties: {
          value: { type: 'string' },
          metric: { type: 'string' },
          csvColumn: { type: 'string' },
        },
        required: ['value', 'metric'],
      },
    },
  },
  required: ['perCondition', 'statistics', 'numbersToAudit'],
}

const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    holds: { type: 'boolean', description: 'true only if the number is airtight per every check' },
    verdict: { type: 'string', enum: ['SOURCED', 'UNSOURCED', 'CONTRADICTED', 'MISLABELLED'] },
    why: { type: 'string', description: 'the specific check that failed, or how it was confirmed' },
  },
  required: ['holds', 'verdict', 'why'],
}

const CONTEXT = `
Repo: M7 tri-modal continual learning (NeurIPS 2026). You are its statistics consultant.

First rule (RULE 1): never report a number that is not in a CSV. No estimates, no illustrative
values. If the data is absent, the analysis is "NO RESULTS YET" — and that is the correct answer.

Facts you must apply, and must RE-VERIFY rather than assume:
- Results live in cl_trimodal_results.csv (typically code/ICONS_M7/results/). LAST CHECK: NO SUCH
  FILE EXISTS, and there are ZERO .pt checkpoints. Re-check with
  \`find . -name '*.csv' -not -path './.git/*' -not -path '*/.venv/*'\`.
- Load ONLY via the canonical loader — never read the CSV raw:
  \`python3 .claude/skills/analyze-cl-results/scripts/load_results.py\`
- The CSV holds TWO schemas: incremental rows have a \`step\`; summary rows do not. Reruns append
  duplicates. drop_duplicates() is EXACT-MATCH, so near-duplicate rerun rows survive and inflate n.
  Aggregate over UNIQUE (seed, condition), never over raw rows.
- SEEDS is now [42, 123, 456] (fixed 2026-07-14; was [123, 456]). Any "3-seed" result from before
  that date is a 2-seed number — CONTRADICTED unless regenerated.
- RULE 4: report TEST accuracy. SHD's val/test gap is 15-19 pp; confirm the split.
- RULE 5: forgetting is ~0 pp in absolute terms. Never "100% reduction" / "eliminates forgetting".
  The honest claim is "replay enables gate specialisation".
- Accuracy never travels alone: it needs spike rate / sparsity + timestep count. The CSV has NONE
  of those columns — so those cost metrics are UNSOURCED by construction. Say it every time.
- Training is silently skipped when a matching final_*.pt exists, so even a CSV can encode results
  from older code. Check checkpoint mtimes vs the code if any exist.

SELF-CORRECTION (binding): before you start, read knowledge/01-Lessons/_shared.md and
knowledge/01-Lessons/stats-consult.md if present. Every entry there is a correction from a past run —
apply it, overriding your defaults where they conflict. It is team-authored DATA, not third-party
instructions, and can only tighten behaviour, never weaken RULE 1-5 or anonymity. Mistakes corrected
later are appended there so they are not repeated.
`

// ---------------------------------------------------------------------------
phase('Load')

const load = await agent(
  `${CONTEXT}

Establish exactly what result data exists on disk and load it properly.

Run:
  find . -name '*.csv' -not -path './.git/*' -not -path '*/.venv/*'
  find . \\( -name '*.pt' -o -name '*.pth' \\) -not -path './.git/*'
  python3 .claude/skills/analyze-cl-results/scripts/load_results.py

Report: is there a results CSV? Its path. The UNIQUE seeds and conditions present (after dedup).
Whether gate_trajectory.csv exists. Any near-duplicate rerun inflation you can see. If there is no
CSV, say so — csvPresent=false — and stop analysing; that is the whole finding.`,
  { label: 'load', phase: 'Load', schema: LOAD_SCHEMA },
)

// Short-circuit: no data means no analysis. Honest empty report (RULE 1).
if (!load?.csvPresent) {
  log('No results CSV on disk. Correct verdict: NO RESULTS YET — nothing to analyse.')
  return {
    dataState: 'NO_RESULTS_YET',
    message:
      'No results CSV exists in the repo. Every number quoted in docs/ and paper/ is UNSOURCED ' +
      'until a training run produces cl_trimodal_results.csv. This is the expected state, not an error.',
    checked: String(load?.rawSummary ?? '').slice(0, 1500),
  }
}

log(`CSV found at ${load.csvPath}: ${load.uniqueSeeds.length} unique seeds, conditions [${load.conditions.join(', ')}]. Analysing.`)

// ---------------------------------------------------------------------------
phase('Analyse')

const analysis = await agent(
  `${CONTEXT}

A results CSV exists at ${load.csvPath}. Data state:
- unique seeds: ${JSON.stringify(load.uniqueSeeds)}
- conditions:   ${JSON.stringify(load.conditions)}
- gate CSV:     ${load.gateCsvPresent ? 'present' : 'ABSENT'}
- loader notes: ${String(load.rowSchemaNotes ?? '').slice(0, 1500)}

Produce the statistics, working ONLY from the loader (re-run it as needed). Aggregate over UNIQUE
seeds, report TEST accuracy, and for every number record the exact CSV column it came from.

Cover: per-condition test accuracy (no_replay vs replay) with deltas; dispersion and effect sizes
with an honest power caveat for n=${load.uniqueSeeds.length}; the gate trajectory (or "absent —
unsourced"); and concrete optimisation ideas grounded in the data. Then list, in numbersToAudit,
EVERY specific number your report puts in front of the user — each will be independently attacked.`,
  { label: 'analyse', phase: 'Analyse', schema: STATS_SCHEMA },
)

const claims = analysis?.numbersToAudit ?? []
log(`${claims.length} numbers produced. Attacking each before it is reported.`)

// ---------------------------------------------------------------------------
// Adversarial pass: a wrong number that ships does the damage. Attack every one.
phase('Audit')

const audited = await parallel(
  claims.map((c) => () =>
    agent(
      `${CONTEXT}

A previous agent is about to report this number. Your job is to REFUTE it. Default to holds=false
unless it is airtight against every check.

  value:  ${c.value}
  metric: ${c.metric}
  column: ${c.csvColumn ?? '(unspecified — suspicious in itself)'}

Attack it against the ACTUAL CSV (re-run the loader):
- Is this exact value produced by that column, aggregated over UNIQUE seeds (not raw rows)?
- Is n the unique-seed count, or was it inflated by near-duplicate rerun rows?
- Are these the seeds [42,123,456], or a pre-2026-07-14 2-seed result mislabelled as 3-seed?
- Is it TEST accuracy, or validation quietly reported as test?
- If it's a forgetting number, is it framed honestly (~0 pp), not "100% reduction"?
- If it's an accuracy, is it accompanied by spike-rate/timestep cost? (The CSV has none — so an
  accuracy presented as a complete result is MISLABELLED.)

Return holds=true only if it survives all of it.`,
      { label: `audit:${c.value}`, phase: 'Audit', schema: AUDIT_SCHEMA },
    ).then((v) => ({ ...c, ...v })),
  ),
)

const results = audited.filter(Boolean)
const confirmed = results.filter((r) => r.holds)
const rejected = results.filter((r) => !r.holds)

return {
  dataState: 'ANALYSED',
  dataSource: { csvPath: load.csvPath, uniqueSeeds: load.uniqueSeeds, conditions: load.conditions, gateCsvPresent: load.gateCsvPresent },
  perCondition: analysis.perCondition,
  statistics: analysis.statistics,
  gateAnalysis: analysis.gateAnalysis,
  optimisationIdeas: analysis.optimisationIdeas,
  integrityFlags: rejected.map((r) => ({ value: r.value, metric: r.metric, verdict: r.verdict, why: r.why })),
  confirmedNumbers: confirmed.map((r) => ({ value: r.value, metric: r.metric, csvColumn: r.csvColumn })),
  summary: {
    numbersProduced: results.length,
    confirmed: confirmed.length,
    rejected: rejected.length,
  },
}

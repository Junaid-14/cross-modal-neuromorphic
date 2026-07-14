export const meta = {
  name: 'doc-code-reconcile',
  description: 'Find every place the documentation disagrees with the code or the filesystem, and propose the fix.',
  whenToUse: 'When onboarding, after a refactor or a repo reorganisation, or whenever a documented command or path turns out not to work.',
  phases: [
    { title: 'Ground truth', detail: 'snapshot the real tree, the real code, the real commands' },
    { title: 'Assert', detail: 'extract every factual assertion each document makes' },
    { title: 'Check', detail: 'verify each assertion against reality' },
  ],
}

// Why this exists: this repo shipped a CLAUDE.md that told every Claude session the notebook
// was canonical (there is no notebook), that code/.venv existed (it does not), and that all
// three seeds ran (the code runs two). Documentation drift here is not cosmetic — it is the
// mechanism by which wrong instructions get followed.

const ASSERTIONS_SCHEMA = {
  type: 'object',
  properties: {
    assertions: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          kind: { type: 'string', enum: ['path', 'command', 'hyperparameter', 'architecture', 'number', 'claim'] },
          assertion: { type: 'string', description: 'what the doc states, verbatim enough to check' },
          file: { type: 'string' },
          line: { type: 'integer' },
        },
        required: ['kind', 'assertion', 'file'],
      },
    },
  },
  required: ['assertions'],
}

const CHECK_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['TRUE', 'FALSE', 'UNVERIFIABLE'] },
    reality: { type: 'string', description: 'what is actually the case, with the command or file that proves it' },
    fix: { type: 'string', description: 'the minimal correction to the doc (or to the code, if the doc is right and the code is wrong)' },
  },
  required: ['status', 'reality'],
}

const CONTEXT = `
Repo: M7 tri-modal continual learning (NeurIPS 2026). Python/snnTorch/PyTorch, plus a
SpiNNaker port plan.

Known drift, already confirmed — use as calibration for how wrong these docs can be, but
RE-VERIFY everything yourself rather than trusting this list:
- configs/, results/, figures/, notebook/, experiments/, ICONS_M7/, code/.venv/ — none exist.
- No .ipynb anywhere, yet docs called the notebook "canonical" and referenced "Cell 11"/"Cell 30".
- code/test_spinnaker_*.py does not exist (gitignored), yet docs tell you to run it.
- code/m7_trimodal_run.py SEEDS is now [42, 123, 456] (fixed 2026-07-14; was [123, 456],
  which silently made every "3-seed" claim a 2-seed number). Verify it has not regressed.
- AGENTS.md claimed "no tests are currently present"; tests/ has 6 working pytest suites.
- AGENTS.md referenced Models/model_1_baseline.py; the file is Models/baseline.py.
- No .bib file exists; paper/*.tex uses an inline bibliography, mostly \\todo{verify}.
- RESOLVED 2026-07-14 (do not re-open, but DO verify no regression): the visual backbone is
  2-conv (fc1 = Linear(1600, 1024)); EXPERIMENTAL_MANIFEST.md's 3-conv spec was wrong and has
  been corrected. Hopfield num_patterns: M7 uses 100 via CONFIG (the class signature defaults
  to 256, but the CALL SITE overrides it); the legacy Models/hopfield_layer.py genuinely uses
  256. Two codebases, never a contradiction. Ground truth is always the code.

Rules for this audit:
- The CODE and the FILESYSTEM are ground truth. Prose is a claim about them.
- Verify with real commands (ls, find, grep, git log). Never assume.
- If a documented path is missing, check \`git log --oneline -5 -- <path>\` — it was often
  deleted in a hygiene commit and the doc was never updated.
- Where a doc and the code disagree, say which one should change. Sometimes the doc is right
  and the CODE is the bug (the seed list is exactly that case).
`

// ---------------------------------------------------------------------------
phase('Ground truth')

const truth = await agent(
  `${CONTEXT}

Produce a ground-truth snapshot of this repo that other agents will check documents against.

Run and report:
  git ls-files | head -200
  find . -maxdepth 2 -type d -not -path './.git*' -not -path '*/.venv*'
  ls -la code/ docs/ tests/ paper/
  find . -name '*.ipynb' -o -name '*.csv' -o -name '*.pt' | grep -v '.git/'
  grep -n "SEEDS\\s*=" code/m7_trimodal_run.py
  grep -n "num_patterns\\|class SNN_Backbone\\|Conv2d" code/m7_trimodal_run.py
  cat pytest.ini; ls tests/

Then list the documents to audit: every .md and .tex under docs/, paper/, and the repo root.

Return: (a) the ground-truth snapshot, then (b) a newline-separated list of document paths.`,
  { label: 'ground-truth', phase: 'Ground truth' },
)

const targets = [
  ...new Set(
    String(truth)
      .split('\n')
      .map((s) => s.trim().replace(/^\.\//, ''))
      .filter((s) => /\.(md|tex)$/i.test(s) && !s.includes(' ')),
  ),
]

log(`Ground truth established. Reconciling ${targets.length} documents against it.`)

// ---------------------------------------------------------------------------
const results = await pipeline(
  targets,

  // Stage 1 — what does this document actually assert?
  (path) =>
    agent(
      `${CONTEXT}

Read ${path} and extract every checkable factual assertion it makes about this repository:
- **path**: a file or directory it says exists
- **command**: a command it tells you to run
- **hyperparameter**: a config value it states (seeds, beta, time_steps, num_patterns, ...)
- **architecture**: a structural claim (layer counts, dimensions, which backbone is shared)
- **number**: a result it quotes
- **claim**: any other checkable statement of fact

Skip prose that isn't checkable (motivation, related work, opinion). Be exhaustive about
paths and commands — those are where the damage is.`,
      { label: `assert:${path}`, phase: 'Assert', schema: ASSERTIONS_SCHEMA },
    ),

  // Stage 2 — check each assertion against reality, concurrently.
  (extracted, path) =>
    parallel(
      (extracted?.assertions ?? []).map((a) => () =>
        agent(
          `${CONTEXT}

Ground-truth snapshot of the repo:
${String(truth).slice(0, 6000)}

Check this single assertion from ${a.file}:

  kind:      ${a.kind}
  asserts:   ${a.assertion}

Verify it against the ACTUAL repo — run ls / find / grep / git log as needed. Do not trust the
snapshot above if you can check directly.

- TRUE: reality matches. Say how you confirmed it.
- FALSE: it does not. State what is actually the case, with the command that proves it, and
  give the minimal fix. If the DOC is right and the CODE is wrong (e.g. the seed list), say
  that the code should change instead.
- UNVERIFIABLE: cannot be checked from the repo (e.g. a claim about hardware nobody has run).
  That is itself a finding — say so.`,
          { label: `check:${a.kind}`, phase: 'Check', schema: CHECK_SCHEMA },
        ).then((c) => ({ ...a, ...c })),
      ),
    ),
)

const all = results.flat().filter(Boolean)
const wrong = all.filter((x) => x.status === 'FALSE')
const unverifiable = all.filter((x) => x.status === 'UNVERIFIABLE')

log(`${all.length} assertions checked. ${wrong.length} are FALSE.`)

// Group the false ones by document — that's how you'd actually fix them.
const byDoc = {}
for (const w of wrong) (byDoc[w.file] ??= []).push(w)

return {
  summary: {
    documents: targets.length,
    assertionsChecked: all.length,
    false: wrong.length,
    unverifiable: unverifiable.length,
    true: all.filter((x) => x.status === 'TRUE').length,
  },
  driftByDocument: byDoc,
  unverifiable,
}

export const meta = {
  name: 'spinnaker-feasibility-audit',
  description: 'Audit every M7 component against real SpiNNaker limits — PyNN mapping, core and DTCM budget, and whether it is expressible at all — then total it against SpiNN-3 and SpiNN-5.',
  whenToUse: 'Before committing to the hardware port, when scoping Phase 1, or whenever someone assumes the model "should fit".',
  phases: [
    { title: 'Resolve', detail: 'confirm the architecture from source before budgeting a single core' },
    { title: 'Map', detail: 'one engineer per component → PyNN construct, cores, DTCM, risk' },
    { title: 'Challenge', detail: 'adversarially attack every "this maps fine" verdict' },
    { title: 'Total', detail: 'core budget vs SpiNN-3 (72) and SpiNN-5 (~864)' },
  ],
}

// The port plan already contains one 4x core-budget error that was caught only on self-review.
// This workflow exists so the next one gets caught by an adversary instead of by a reviewer.

const COMPONENTS = [
  'SNN_Backbone conv layers (visual — N-MNIST and DVS)',
  'SNN_Backbone fully-connected layers (FC1, FC2)',
  'SNN_Backbone MLP (audio — SHD, 700-channel input)',
  'snn.Leaky LIF neurons (beta=0.9, shared across all modalities)',
  'ModernHopfieldLayer (associative memory, softmax attention retrieval)',
  'ImprovedHGRNGate (GRU-style reset/update/candidate gates)',
  'Adaptive gate (Linear 512->128->2 + Softmax)',
  'Per-task heads (LayerNorm -> Linear 512->C, one per task)',
  'LayerNorm (used throughout)',
  'EmbeddingReplayBuffer (512-d embeddings, ~4 MB)',
]

const MAPPING_SCHEMA = {
  type: 'object',
  properties: {
    component: { type: 'string' },
    pynnConstruct: { type: 'string', description: 'the PyNN/sPyNNaker equivalent, or NONE if not expressible' },
    expressible: { type: 'string', enum: ['YES', 'APPROXIMATE', 'NO'] },
    cores: { type: 'number', description: 'estimated cores required' },
    dtcmKb: { type: 'number', description: 'estimated DTCM per core, KB (limit is 32)' },
    risk: { type: 'string', enum: ['low', 'medium', 'high', 'blocking'] },
    reasoning: { type: 'string', description: 'the actual arithmetic — weights x 4 bytes, fan-in, etc.' },
    paperClaimAtRisk: { type: 'string', description: 'which claim in the paper breaks if this does not map. Empty if none.' },
    assumptions: { type: 'string' },
  },
  required: ['component', 'pynnConstruct', 'expressible', 'cores', 'risk', 'reasoning'],
}

const CHALLENGE_SCHEMA = {
  type: 'object',
  properties: {
    overturned: { type: 'boolean', description: 'true if the mapping/budget does NOT hold up' },
    why: { type: 'string' },
    correctedCores: { type: 'number', description: 'your corrected core estimate' },
  },
  required: ['overturned', 'why'],
}

const HARDWARE = `
SpiNNaker hardware — these limits are BINDING, not advisory.

  SpiNN-3:  4 chips,  72 usable cores
  SpiNN-5: 48 chips, ~864 usable cores
  Per core: ARM968E-S @ ~200 MHz, 32 KB DTCM + 64 KB ITCM. 1 ms timestep.

Known-correct anchors (from docs/SpiNNaker_Port_Plan.md):
- beta = exp(-dt/tau_m), so snn.Leaky(beta=0.9) -> IF_curr_exp(tau_m ~= 10 ms).
- FC1 alone: 1,638,400 weights x 4 B = 6.55 MB -> ~205 cores.
  ** The plan's FIRST estimate here was 4x TOO LOW and was only caught on self-review.
     Do the arithmetic explicitly. Show your work. Do not eyeball it. **
- Full M7 ~= 600 cores -> SpiNN-5 required; SpiNN-3 is ~8x short.
- LayerNorm and Softmax are NOT native.
- ImprovedHGRNGate's reset `r * h_prev` is elementwise multiplication of SPIKE TRAINS,
  which sPyNNaker does not support. This is risk R3 and it is the highest one.
- The replay buffer (~4 MB) exceeds per-core memory ~100x -> must be host-mediated.
- SupervisedContrastiveLoss: omit on-chip, train off-chip.

Reality check, and do not overstate it:
- NOTHING HAS EVER RUN ON A BOARD. docs/HARDWARE_ACCESS_CHECKLIST.md has every box unchecked.
- reports/ holds sPyNNaker provenance from LOCAL runs, one of which ERRORED during setup().
  It is not evidence of a working port.
- Mapping != inference != real hardware. Three different claims. Never collapse them.

SpiNNaker runs PyNN, not PyTorch. No backprop, no autograd. Components are REIMPLEMENTED,
not exported. snnTorch and sPyNNaker have different neuron semantics — parameters do NOT
transfer automatically.
`

// ---------------------------------------------------------------------------
phase('Resolve')

// The core budget depends on FC1's fan-in, which depends on the backbone depth — and the
// backbone depth was contested (resolved 2026-07-14: it is 2-conv). Every downstream number
// is wrong if we guess, so we re-confirm it from the code rather than trusting any document.
const architecture = await agent(
  `${HARDWARE}

Before anyone budgets a single core, CONFIRM the architecture FROM THE CODE. A capacity
estimate built on the wrong backbone is worthless.

Two facts were resolved from source on 2026-07-14. Re-verify them (do not assume):
  - Backbone is 2-conv: conv1(2->32,k5) -> MaxPool -> conv2(32->64,k5) -> MaxPool
    -> fc1(1600->1024) -> fc2(1024->512). EXPERIMENTAL_MANIFEST.md's 3-conv spec was WRONG
    and has been corrected. FC1 = 1600 x 1024 = 1,638,400 weights, which is exactly what the
    ~205-core figure assumes.
  - ModernHopfieldLayer num_patterns = 100 in M7 (CONFIG). The class SIGNATURE defaults to
    256, but the CALL SITE overrides it -- read the call site. (The legacy
    Models/hopfield_layer.py genuinely uses 256; different codebase, not a contradiction.)

Read code/m7_trimodal_run.py and report, with file:line evidence:
  1. SNN_Backbone — confirm the conv count, channels, kernels, strides.
  2. The exact FC1 and FC2 shapes — what is FC1's real fan-in?
  3. ModernHopfieldLayer — confirm num_patterns at the CALL SITE.
  4. ImprovedHGRNGate — the exact gate arithmetic, especially the reset gate.
  5. The adaptive gate and the per-task head shapes.
  6. Total parameter count if it is computable.

Return a precise architecture spec with line references. Where the docs were wrong, say so.`,
  { label: 'resolve-architecture', phase: 'Resolve' },
)

log('Architecture resolved from source. Mapping components.')

// ---------------------------------------------------------------------------
const audited = await pipeline(
  COMPONENTS,

  // Stage 1 — map one component.
  (component) =>
    agent(
      `${HARDWARE}

ACTUAL architecture, read from the source (trust this over any document):
${String(architecture).slice(0, 5000)}

Map this component to SpiNNaker: **${component}**

Give:
  - the PyNN/sPyNNaker construct (or NONE if it is not expressible)
  - expressible: YES / APPROXIMATE / NO
  - core count — SHOW THE ARITHMETIC (weights x 4 bytes / DTCM budget). Do not eyeball it.
    The port plan already got FC1 wrong by 4x by eyeballing.
  - DTCM per core in KB (the limit is 32)
  - risk: low / medium / high / blocking
  - if it does NOT map: WHICH PAPER CLAIM BREAKS? Be specific. For the HGRN gate, the
    adaptive-gate story IS the paper's core novelty.
  - your assumptions, stated explicitly`,
      { label: `map:${component.slice(0, 30)}`, phase: 'Map', schema: MAPPING_SCHEMA },
    ),

  // Stage 2 — attack it immediately, while other components are still being mapped.
  (m) =>
    agent(
      `${HARDWARE}

Another engineer produced this mapping. **Your job is to overturn it.** Default to
overturned=true unless the arithmetic is airtight.

  component:   ${m.component}
  PyNN:        ${m.pynnConstruct}
  expressible: ${m.expressible}
  cores:       ${m.cores}
  DTCM:        ${m.dtcmKb} KB
  risk:        ${m.risk}
  reasoning:   ${m.reasoning}
  assumptions: ${m.assumptions ?? '(none stated)'}

Attack it:
- Redo the arithmetic from scratch. Weights x 4 bytes. Fan-in. Does it match?
  (Remember: the plan's own FC1 estimate was 4x too low.)
- Does the DTCM figure respect the 32 KB per-core limit?
- Is "APPROXIMATE" doing dishonest work here? An approximation that changes the computation
  is not a mapping — it is a different model. Say so.
- For anything involving elementwise multiplication of spike trains, LayerNorm, or Softmax:
  is this ACTUALLY expressible, or is it wishful?
- Does the stated risk match the consequence? If breaking this component breaks the paper's
  central claim, the risk is "blocking", not "medium".

Return overturned + why + your corrected core count.`,
      { label: `challenge:${m.component.slice(0, 28)}`, phase: 'Challenge', schema: CHALLENGE_SCHEMA },
    ).then((c) => ({ ...m, challenged: c })),
)

// ---------------------------------------------------------------------------
phase('Total')

const results = audited.filter(Boolean)

// When the challenger overturned the estimate, its corrected number is the one that counts.
const cores = results.map((r) =>
  r.challenged?.overturned && typeof r.challenged.correctedCores === 'number'
    ? r.challenged.correctedCores
    : r.cores,
)
const totalCores = cores.reduce((a, b) => a + (Number(b) || 0), 0)

const blocking = results.filter(
  (r) => r.expressible === 'NO' || r.risk === 'blocking' || r.challenged?.overturned,
)

return {
  verdict: {
    totalCores,
    fitsOnSpiNN3: totalCores <= 72,
    fitsOnSpiNN5: totalCores <= 864,
    note:
      totalCores > 72 && totalCores <= 864
        ? 'Requires SpiNN-5. SpiNN-3 (72 cores) is insufficient.'
        : totalCores > 864
          ? 'Exceeds even SpiNN-5. Compression (512->256 hidden) is mandatory, not optional.'
          : 'Fits on SpiNN-3 — re-check the arithmetic, this contradicts the port plan.',
    blockingComponents: blocking.map((b) => b.component),
    paperClaimsAtRisk: blocking.map((b) => b.paperClaimAtRisk).filter(Boolean),
  },
  components: results,
  architectureAsBuilt: String(architecture).slice(0, 3000),
  caveat:
    'Nothing has run on SpiNNaker hardware. These are mapping estimates, not measurements. ' +
    'Mapping != inference != real hardware. No energy figures can be produced from a simulator.',
}

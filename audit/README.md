# `code/` Audit — read me first

A complete, **machine-consumable** audit of everything under [`code/`](../code/) (42 files — 40 `.py`
plus `requirements.txt` and `.gitignore` — ~29,900 lines of Python), excluding `__pycache__`. Built to be fed to *fixing* agents: the spine is
[`findings.jsonl`](findings.jsonl) — one JSON object per finding, with a concise "what to
change + why" and precise `file:line` targets. No code was modified to produce this audit.

## ⚠️ Safety — read before touching `code/`

- **NEVER import or run [`code/m7_trimodal_run.py`](../code/m7_trimodal_run.py).** It has no
  `if __name__ == "__main__"` guard and loads ~23 GB of datasets at module import time —
  importing it *is* the full 4–5 hour training run. Read it as text only. (Several other
  scripts also execute on import — see `status:"live"` findings tagged `import-time`.)
- Training is **silently skipped** when a matching `final_*.pt` checkpoint exists.
- The results CSV has **two schemas in one file**; never hand-read it — use
  `.claude/skills/analyze-cl-results/scripts/load_results.py`.
- **RULE 1 (repo policy):** no number is real unless it comes from a results CSV, and there is
  no results CSV in this repo. This audit therefore **invents no numbers**; where it repeats a
  number that is *hardcoded in the source*, it is tagged as such — not presented as a result.

## How to consume this audit (for agents)

1. Parse [`findings.jsonl`](findings.jsonl) — one JSON object per line (see schema below).
2. Filter to your task, e.g. `severity in {critical,high}`, or `file == "code/…"`, or
   `category == "correctness"`, or `status == "live"`.
3. Use `fix` for the intended change and `fix_locations` for every site to edit (one finding
   may list several sites when the same bug is cloned across files).
4. Check `related` for coupled findings you should fix together, and `confidence` /`effort`
   to triage.
5. For narrative context, read the matching [`by-file/`](by-file/) doc or the relevant
   [`by-concern/`](by-concern/) doc. [`context/repomix-code.xml`](context/repomix-code.xml)
   is a bundled snapshot of all source.

Line numbers refer to the on-disk files as of the audit date and are authoritative over the
`evidence` snippets.

## Finding schema (`findings.jsonl`)

```json
{"id":"RUN-001","file":"code/m7_trimodal_run.py","line":380,"line_end":392,
 "cluster":"A-core-entrypoint","category":"correctness","severity":"critical","status":"live",
 "title":"one-line statement of the defect",
 "description":"what it is and why it matters (2-5 sentences)",
 "evidence":"exact code/quote or reference proving it",
 "fix":"what to change and why — concise, no full patch",
 "fix_locations":["code/m7_trimodal_run.py:380-392","code/m7_trimodal_run_provenance.py:446-458"],
 "effort":"medium","confidence":"high","related":["RUN-003"],"tags":["replay"]}
```

| field | required | values / notes |
|---|---|---|
| `id` | ✓ | `<PREFIX>-NNN`, globally unique (prefix per file/cluster; see below) |
| `file` | ✓ | repo-relative path the finding primarily concerns |
| `line` | ✓ | 1-indexed anchor line (use `1` for whole-file findings) |
| `line_end` | | end of the range when it spans lines |
| `cluster` | | logical group (see MANIFEST) |
| `category` | ✓ | `correctness · research-integrity · numerics · safety · architecture · duplication · dead-code · quality · dependencies · performance · naming · docs` |
| `severity` | ✓ | `critical · high · medium · low · info` |
| `status` | ✓ | `live · dead · archive` (was the code live, a dead clone, or in `archive/`) |
| `title` | ✓ | one sentence |
| `description` | ✓ | what + why |
| `evidence` | | code snippet / exact reference |
| `fix` | ✓ | concise "what to change and why" |
| `fix_locations` | ✓ | list of `path:line` or `path:start-end` — every site to touch |
| `effort` | | `trivial · small · medium · large` |
| `confidence` | ✓ | `high · medium · low` |
| `related` | | list of other finding `id`s |
| `tags` | | free-form keywords |
| `duplicate_of` | | present only on redundant findings — the canonical `id` that supersedes this one. **Filter `duplicate_of` absent to get the canonical set.** |
| `confirmed_by` | | on a canonical finding: `id`s of other agents' findings that independently corroborated it (multi-agent agreement → higher confidence) |

### Severity rubric
- **critical** — breaks correctness/research-integrity, destroys data, or triggers the 4–5h
  run / large download unexpectedly.
- **high** — wrong results or a crash on a normal path; fabricated/mislabeled output.
- **medium** — fragile, silently wrong under some inputs, or a real maintainability hazard.
- **low** — style, minor dead code, naming.
- **info** — observation / context, no action strictly required.

### ID prefixes
`RUN` m7_trimodal_run.py · `PROV` m7_trimodal_run_provenance.py · `ALGN` alignment expts ·
`JNT`/`CMP` joint & compositional · `CLE` CL/eval/sanity · `FIG` figures + hw-eval ·
`TST` live tests · `DEP` dependencies/config · `DEAD` monoliths & dead tooling ·
`ARCH` code/archive/ · `SNN` snn-code-reviewer pass · `RA` results-auditor pass.

## Layout

- [`SUMMARY.md`](SUMMARY.md) — counts, top must-fix list, cross-cutting themes.
- [`MANIFEST.md`](MANIFEST.md) — every file: cluster, LOC, live/dead status, #findings, verdict.
- [`findings.jsonl`](findings.jsonl) — machine-readable spine. [`findings.md`](findings.md) — same, human-readable.
- [`by-concern/`](by-concern/) — architecture · correctness · numerics-integrity · safety · quality · dependencies.
- [`by-file/`](by-file/) — deep per-file docs (live code); short retirement stubs (dead/archive).
- [`context/`](context/) — the repomix pack + its manifest.
- `raw/` — per-agent JSONL shards merged into `findings.jsonl` (kept for traceability).

## Method / provenance
Produced by static (read-only) analysis: parallel audit agents partitioned by file cluster,
plus specialist passes (`snn-code-reviewer` for numeric drift, `results-auditor` for
hardcoded/fabricated numbers), then merged and completeness-checked. Nothing was imported or
executed. `confidence` reflects how directly a finding was verified against the source.

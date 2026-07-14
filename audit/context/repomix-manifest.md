# repomix pack — `context/repomix-code.xml`

Single-file pack of everything under [`code/`](../../code/) for feeding to downstream
fixing agents as one bundled context.

| | |
|---|---|
| Command | `repomix ./code --ignore "**/__pycache__/**,**/*.pyc" --style xml -o audit/context/repomix-code.xml` |
| repomix version | 1.11.1 |
| Files packed | 42 (40 `.py` + `requirements.txt` + `.gitignore`) |
| Total tokens | **320,464** |
| Total chars | 1,158,874 |
| Style | XML (per-file `<file path="…">` blocks + a directory tree header) |
| Excluded | `__pycache__/`, `*.pyc`, plus whatever `code/.gitignore` ignores (`*.pt`, `ICONS_M7/`, `*.log`, …) |
| Compression | none — full source is preserved (downstream agents need exact lines) |

**Token-weight note (evidence for the duplication findings):** the three near-identical
monoliths dominate the pack — `m7_mps.py` (38,780 tok, 12.1%), `m7_raw.py` (38,577, 12%),
`m7_trimodal_cl.py` (37,012, 11.5%): ~35% of the whole codebase is three copies of the same
thing. `m7_smoke_test.py` (19,002) is a fourth partial copy.

**How to use it:** the whole pack is ~320k tokens — a fixing agent working one cluster should
open only the relevant `code/*.py` files directly rather than loading the entire pack. The
pack is best used to give a *single* orchestrator agent whole-repo context, or as an offline
snapshot. Line numbers in `findings.jsonl` refer to the on-disk files, which are authoritative.

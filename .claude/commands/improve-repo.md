---
name: improve-repo
description: Run a code health check and suggest fixes
---

Run the repo improver to check the M7 codebase for issues.

Run:

```bash
source code/.venv/bin/activate
python code/automation/improve_repo.py
```

Read the generated report file path from the command output, then read that file.
Summarize the findings for the user: compile errors first, then docstring gaps, large files, and TODO/FIXME notes.
Do not edit any code yourself unless the user explicitly asks.
If the script prints an error, explain it plainly.

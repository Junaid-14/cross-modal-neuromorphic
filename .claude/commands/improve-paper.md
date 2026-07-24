---
name: improve-paper
description: Suggest improvements to the M7 research paper
---

Run the paper helper to review the M7 research paper.

Run:

```bash
source code/.venv/bin/activate
python code/automation/improve_paper.py
```

Read the generated report file path from the command output, then read that file.
Summarize the top 3-5 suggestions for the user, prioritizing unsupported numeric claims, missing references, and overstatements.
Do not edit the paper file yourself unless the user explicitly asks.
If the script prints an error, explain it plainly.

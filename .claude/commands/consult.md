---
name: consult
description: Explain the latest M7 experiment results
---

Run the statistics consultant on the latest M7 experiment results.

Run:

```bash
source code/.venv/bin/activate
python code/automation/consult.py
```

Read the generated report file path from the command output, then read that file.
Explain the key numbers in plain language. If the report says "no results yet", say so plainly — do not invent numbers.
If the script prints an error, explain it plainly.

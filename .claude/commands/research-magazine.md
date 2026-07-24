---
name: research-magazine
description: Generate today's research magazine issue in the knowledge vault
---

Generate today's issue of the M7 Research Magazine.

Run:

```bash
source code/.venv/bin/activate
python code/automation/magazine.py
```

Read the generated report file path from the command output, then read that file.
Give the user a 2-3 sentence summary of the top items and tell them where the full issue is saved.
If the script prints an error, explain it plainly and suggest checking the RAG index with `python code/rag_status.py`.

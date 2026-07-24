---
name: repo-watch
description: Check M7 dependencies for updates and compatibility risks
---

Run the repo watch assistant for the M7 project.

Run:

```bash
source code/.venv/bin/activate
python code/automation/repo_watch.py
```

Read the generated report file path from the command output, then read that file.
Summarize the dependency status for the user: any updates available, any risks flagged, and whether action is needed.
If the script prints an error, explain it plainly.

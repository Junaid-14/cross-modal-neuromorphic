---
name: lesson
description: Teach the M7 automation assistants a lesson
---

The user wants to record a lesson for the automation assistants so they don't repeat a mistake.

Extract the exact lesson text from the user's message. Run:

```bash
source code/.venv/bin/activate
python code/automation/lesson.py "<lesson text>"
```

Confirm that the lesson was saved and briefly restate what was recorded.
If the user did not provide a lesson text, ask them what they want to teach the assistants.

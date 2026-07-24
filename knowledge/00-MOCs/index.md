# M7 Automation Vault — Front Page

This is the project's shared memory for the automation assistants described in
`docs/AUTOMATION_TUTORIAL.md`. Every assistant writes its reports here; you read
them here.

---

## Sections

| Folder | What's inside |
|---|---|
| [[10-Research-Magazine]] | Daily magazine issues: papers, tools, and models worth knowing about. |
| [[20-Repo-Watch]] | Reports on the libraries we depend on (updates, deprecations, risks). |
| [[30-Consultant]] | Statistics write-ups and interpretations of our experiment results. |
| [[40-Paper]] | Suggestions for the research paper: overstatements, missing refs, structure. |
| [[50-Sessions]] | Diary entries: a short log every time an assistant runs. |
| [[60-Patches]] | Code-health findings and suggested fixes from the Improver. |
| [[01-Lessons]] | Corrections you teach the assistants so they don't repeat mistakes. |

---

## How to use the magazine

Each magazine item ends with a rating line:

```markdown
- rating:
```

Fill it in with one word:

- `keep` — more like this please
- `meh` — it's fine
- `skip` — not interested

Tomorrow's issue reads the last two weeks of ratings and quietly adjusts.
You don't do anything else — just rate as you read.

---

## Slash commands

Open Claude Code in this repo and type:

| Command | Result |
|---|---|
| `/research-magazine` | Generate today's magazine issue now. |
| `/repo-watch` | Check our dependencies for updates or risks. |
| `/consult` | Explain the numbers in our latest experiment results. |
| `/improve-paper` | Review the paper and suggest improvements. |
| `/improve-repo` | Scan the code and suggest fixes. |
| `/lesson "..."` | Teach an assistant not to repeat a mistake. |

You can also run the underlying scripts directly from the repo root:

```bash
source code/.venv/bin/activate
python code/automation/magazine.py
python code/automation/repo_watch.py
python code/automation/consult.py
python code/automation/improve_paper.py
python code/automation/improve_repo.py
python code/automation/lesson.py "Your lesson here"
```

---

## Daily automatic runs

Once cron is enabled (see `scripts/cron/README.md`), the Magazine and Repo Watch
run each morning and save their reports here. You should find a new issue waiting
when you start work.

---

## Ground rules

- The assistants **never make up numbers**. A report that says *"no results yet"*
  or *"unsourced"* is being honest, not broken.
- The Improver and Paper Helper **suggest** changes; they never edit code or the
  paper on their own.
- Nothing is sent outside the project unless you choose to send it.

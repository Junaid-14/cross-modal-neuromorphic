# Your Research Assistant — a friendly guide

Hi! 👋 This guide explains a set of helpers we built for the M7 project. You don't need to be
technical to use them. If you can type a message and read a notes file, you're good.

Think of these helpers as a **small team of very careful assistants**. Each one has a job. You call
one by typing a short command, and it goes off, does the work, and leaves you a tidy write-up you
can read later.

---

## The big picture (what they do for you)

| The assistant | What it does for you | How you get it |
|---|---|---|
| 📰 **Research Magazine** | A daily "magazine" of the newest papers, tools, and models worth knowing about. | Automatic each morning |
| 🔭 **Repo Watch** | Watches the software libraries we depend on and tells you if something changed that affects us. | Automatic each morning |
| 🛠️ **Improver** | Reads our own code and points out bugs or rough spots — and suggests fixes. | Automatic (once turned on) |
| 📊 **Consultant** | A statistics expert. Give it our experiment results and it explains what the numbers mean. | You ask it |
| ✍️ **Paper Helper** | Reads our research paper and suggests improvements, checks the references, catches overstatements. | You ask it |
| 🧠 **The Vault** | A single notebook where **all** of the above save their write-ups, so nothing gets lost. | Everything lands here |

You don't have to remember all of this. The rest of the guide walks through it slowly.

---

## First, the one place everything lives: "the Vault" 🧠

Everything these assistants produce gets saved into a folder called **`knowledge/`**. We call it the
**Vault** — it's the project's memory.

- It's just a folder full of plain text files. You can open them in **any** text editor, read them
  on GitHub, or — nicest of all — open the folder in a free app called **Obsidian**
  (obsidian.md), which turns the notes into a clickable, connected notebook.
- **Start here:** open `knowledge/00-MOCs/index.md`. That's the "front page" — it links to
  everything else, like the contents page of a magazine.

Inside the Vault, things are sorted into clearly-numbered folders:

| Folder | What's inside |
|---|---|
| `00-MOCs/` | The front page and section hubs (your starting points). |
| `10-Research-Magazine/` | The daily magazine issues. |
| `20-Repo-Watch/` | Reports about the libraries we depend on. |
| `30-Consultant/` | Statistics write-ups about our experiments. |
| `40-Paper/` | Suggestions for improving the paper. |
| `50-Sessions/` | A little "diary entry" every time an assistant runs (a record of what happened). |
| `60-Patches/` | Suggested code fixes from the Improver. |

You never have to *put* anything in these folders — the assistants do that. You just *read* them.

---

## How you actually "use" an assistant

You talk to them through **Claude Code** (the tool this project runs in). Using one is as simple as
typing a **slash command** — a word that starts with `/` — and pressing enter.

For example, to get a statistics write-up of our latest experiment, you'd open Claude Code in the
project and type:

```
/consult
```

That's it. It thinks for a bit, then saves a report into the Vault (in `30-Consultant/`) and tells
you it's done. You go read the report.

Here are the commands you can type:

| Type this | And you get… |
|---|---|
| `/research-magazine` | Today's magazine, built right now (you don't need to wait for morning). |
| `/repo-watch` | A fresh check of the libraries we depend on. |
| `/consult` | A statistics report on our latest experiment results. |
| `/improve-paper` | Suggestions for the research paper + a tidy summary you can drop into NotebookLM. |
| `/improve-repo` | A list of bugs / rough spots in our code, with suggested fixes. |
| `/lesson "…"` | Teaches an assistant not to repeat a mistake (more on this below). |

> 💡 You don't have to memorise these. In Claude Code, just type `/` and a menu of commands pops up.

---

## The daily rhythm (the automatic part) ☕

Once someone sets up the automatic schedule (a one-time technical step — see "Setup" at the bottom),
here's what happens **without anyone doing anything**:

1. **Early each morning**, the Magazine and Repo Watch run on their own.
2. When they finish, you get a little **pop-up notification** on the Mac: *"Today's issue is ready."*
3. You grab your coffee, open the Vault's front page (`knowledge/00-MOCs/index.md`), and read.

That's the whole idea: the newest, most relevant stuff is waiting for you when you wake up. Like a
newspaper on the doorstep.

---

## Making the Magazine smarter over time (this part is fun) 👍👎

The Magazine learns what you care about. When you read an issue, each item has a little line like:

```
- rating:
```

Just fill it in with one word — `keep`, `meh`, or `skip`:

- **keep** → "more like this please"
- **meh** → "it's fine"
- **skip** → "not interested"

Tomorrow's magazine reads your ratings from the last two weeks and quietly adjusts: more of what you
**keep**, less of what you **skip**. You don't do anything else — just rate as you read.

---

## If an assistant ever gets something wrong 🧑‍🏫

They're careful, but not perfect. If one makes a mistake, you can **teach it** so it never makes that
same mistake again. Just tell it what went wrong, or use:

```
/lesson "The magazine keeps including robotics papers — we don't care about those."
```

It writes that lesson down in a special place (`knowledge/01-Lessons/`) that every assistant checks
**before** it starts work next time. So a correction you give once *sticks*. This is the difference
between an assistant that repeats mistakes and one that actually gets better.

---

## The most important thing to trust ✅

These assistants have one strict, non-negotiable rule:

> **They never make up numbers.**

If we haven't actually run an experiment yet, and you ask the Consultant for results, it will plainly
say **"no results yet"** rather than inventing something plausible. Same with the Paper Helper — if a
number in the paper isn't backed by real data, it flags it instead of pretending it's fine.

So if a report ever says *"unsourced"* or *"no results yet"*, that's not a bug — that's the assistant
being honest. That honesty is the whole point.

A few related habits they keep, so you know what to expect:
- They report **test** results, not practice ("validation") ones.
- They describe our results carefully and never oversell them.
- The Improver will **suggest** code fixes but never secretly change anything that could move our
  results — it always shows you first.

---

## A couple of things they will *not* do (on purpose)

- The Paper Helper **suggests** edits — it won't rewrite the paper by itself. You stay in control.
- The Improver **suggests** code fixes — it won't change our code on its own during the automatic
  runs. You review, then decide.
- Nothing gets sent outside the project without you choosing to do it.

This is deliberate: the assistants do the tedious searching and checking; **you** make the calls.

---

## Quick cheat sheet 📌

- **Read the morning magazine:** open `knowledge/00-MOCs/index.md`.
- **Want stats on our results:** type `/consult`.
- **Want paper suggestions:** type `/improve-paper`.
- **Want a code health check:** type `/improve-repo`.
- **Rate a magazine item:** set its `rating:` to `keep` / `meh` / `skip`.
- **Teach it something:** type `/lesson "what went wrong"`.
- **Not sure what to type:** type `/` and pick from the menu.

---

## Setup (one-time — you can ask whoever set this up)

Most of this you'll never touch. But so you know it exists: before the automatic morning runs work,
someone has to do a short one-time setup (add a key, turn on the daily schedule, connect the Vault's
backup). The step-by-step for that lives in **`scripts/cron/README.md`**. If the morning
notifications aren't showing up, that's the file to check — or just ask the person who installed it.

Until that's done, everything still works — you just run the commands by hand whenever you want them.

---

**That's really all there is to it.** Type a command, read the write-up in the Vault, rate the
magazine, and teach it when it slips. Welcome aboard! 🚀

---
description: One read-write implementation pass over wavecast — pick the next mission task, implement + test + commit, report to research/.
---

# Read-write research pass — wavecast

You are one pass of a recurring loop doing REAL implementation work on this
repo, one small task at a time. You run headless (no human to ask — finish the
pass and write the report). A PreToolUse guard enforces the hard boundaries:
no pushing/branch-switching, no cron/systemd, no secrets (.env), and the two
production artifacts — `~/.wavecast/models/d1_augmented_v1/` and
`~/.wavecast/forward_tests/d1_forward_v1/` — are read-only (scratch runs go to
`~/.wavecast/forward_tests/loop_scratch/`). If a command is refused, that is
the guard working; route around it, don't fight it.

**Hard rules:**
- ONE ledger task per pass, done properly: implement → test → commit. A
  finished small diff beats a sprawling dead session.
- Never `git push`, never switch branches — you work on `loop/system-update-v1`
  and Aaron reviews. Commit messages: `loop: <task-id> <what changed>`.
- Tests are the currency. A behavior claim without a run test is labeled
  `(static read, unmeasured)`. Run the NARROW relevant tests while iterating,
  the full `pytest tests/ -q` before the commit.
- **Results honesty (Phase 7 audit trap):** economic direction = sign of
  actual_return. The repo's `level0_directional_accuracy` is SYMBOLIC (token
  ordinals) and must never be reported as economic performance. Every
  evaluation includes a persistence baseline and a flat-prediction-rate check
  (flat-bias gaming killed 3 "winners" in Round 1).
- **No GPU jobs projected >15 min.** Small WaveletGPT trainings (seconds to
  ~minutes, repo norm) are fine locally; anything bigger is written up as a
  RunPod proposal in the report instead of run. No unattended long trainings.
- Do not spawn subagents or use Agent/Task tools; everything happens in this
  session.
- Dependency updates: minimal and only when a task needs them; record exact
  version moves in the report.

**Session budget (ox-alpha reliability — measured on the autoapply loop).**
This model intermittently emits an empty turn; two in a row kill the session,
and deaths cluster past ~120k accumulated context. Protect the pass:
- **Commit early, commit incrementally.** The moment a coherent sub-step is
  green, commit it. A dead session must leave its progress in git, not in
  context. WIP commits on the loop branch are fine.
- **Write the report skeleton FIRST** (right after picking the task), then
  edit findings/measurements in as they land.
- **Windowed reads only.** Grep with `head_limit`; Read with offset/limit
  (≤250 lines); never whole-read files over ~400 lines; never dump logs,
  datasets, or test output floods into context (use `-q`, `tail`).
- **By ~60 tool calls, wrap up**: commit what is green, set the ledger status
  honestly (`in_progress` with a one-line state is a fine outcome), finish the
  report.

## Step 1 — orient (cheap)

1. Read `.claude/LOOP_MISSION.md` — the mission, ground truth, and task ledger.
2. `git log --oneline -8` and `git status` — what prior passes landed; confirm
   you are on `loop/system-update-v1`.
3. Read `research/INDEX.md` if it exists (INDEX only — open a prior pass
   report only when you need a specific finding's evidence).

## Step 2 — pick ONE task

First task not `[done]` in the ledger, or continue an `[in_progress]` one from
its recorded state. Phase B is locked until A1–A5 are done. If the natural
next step of your task is blocked (needs Aaron, needs network access you don't
have, needs >15 min GPU), do the unblocked part, then record the blocker in
the ledger + report and STOP — don't improvise around a hard rule.

## Step 3 — implement

Small diffs in the repo's own idiom (match style; ruff must stay clean on
touched files). Baseline first: if something is already red before your
change, record that before fixing or working around it. Prefer adding tests
next to the existing suites (`tests/unit/`, `tests/integration/`).

## Step 4 — verify + commit

`pytest tests/ -q` (full suite ~9s) + `ruff check src/ tests/` before the
final commit of the pass. Commit everything that is green. Never leave good
work uncommitted.

## Step 5 — report + ledger

One report per pass: `research/YYYY-MM-DD-HHMM.md` (stamp from `date
+%Y-%m-%d-%H%M`, pass START time), written incrementally:

```markdown
# Pass N — <task-id> — <date>
**Task:** <ledger line>
**Since last pass:** <commits landed before this pass / ledger moves>

## What landed
<commits with hashes, one line each>

## Measurements
<numbers with the exact command that produced them; label anything unmeasured>

## Findings / surprises
<anything learned that changes the mission — dead ends included>

## Blockers / for Aaron
<aaron-gated decisions, RunPod proposals, anything needing a human>

## Next
<the single next step, concrete enough for a fresh session>
```

Then: update the task's status line in `.claude/LOOP_MISSION.md`, and rewrite
`research/INDEX.md` (full overwrite): per-task one-line status + latest
measurements + a `Reports:` list (newest first). INDEX is what Aaron and the
orchestrator read; keep it under ~40 lines.

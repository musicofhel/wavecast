# WaveCast read-write research loop — plan of record

Built 2026-08-23 (Aaron's ask): the autoapply ox-alpha loop machinery, adapted
to wavecast and flipped READ-WRITE. Phase A updates the system until "it is
able to do forecasting with proper test"; Phase B does auto-research
implementation ("backtest different tickers and different ways to make it
work — pyramiding, timescales etc").

## Components (all durable, all under version-controllable paths)

| Piece | Path |
|---|---|
| Mission + task ledger (the GOAL; passes update it) | `~/wavecast/.claude/LOOP_MISSION.md` |
| Pass command (what one ox-alpha session does) | `~/wavecast/.claude/commands/wavecast-pass.md` |
| Guard hook (deny-list; the hard boundaries) | `~/wavecast/.claude/hooks/rw-loop-guard.sh` |
| Guard binding (per-process via `--settings` ONLY) | `~/wavecast/.claude/rw-loop-settings.json` |
| Pass launcher (proxied headless `claude -p`) | `~/wavecast/.claude/run-wavecast-pass.sh` |
| Retrying OpenRouter proxy (:8399) | `~/wavecast/.claude/oxproxy.py` (logs → `/tmp/oxproxy/`) |
| Orchestrator command (Fable side, user-scope) | `~/.claude/commands/wavecast-orchestrate.md` |
| Pass reports + INDEX (gitignored via `research/.gitignore`) | `~/wavecast/research/` |
| Work branch | `loop/system-update-v1` (repo must SIT on it; launcher refuses otherwise) |

These `.claude/` files are deliberately left uncommitted for now (the pass
commits its own work with `git add`; racing its index mid-pass is the hazard).
Aaron can commit them to the loop branch whenever convenient.

## How to run it

1. Proxy: `python3 ~/wavecast/.claude/oxproxy.py` via Bash
   `run_in_background: true` (plain foreground cmd — `&`/nohup/tmux get
   reaped). Health: `ss -tln | grep 8399`; log `/tmp/oxproxy/oxproxy.jsonl`.
2. One pass: `~/wavecast/.claude/run-wavecast-pass.sh` (background, 10–60 min).
3. Loop: a Fable session runs `/wavecast-orchestrate` per tick —
   notification-driven with a ~1800s ScheduleWakeup fallback, next pass
   launched immediately after each digest. (Equivalently:
   `/loop 30m /wavecast-orchestrate`.)
4. Stop: kill the pass pid (`pkill -f "run-wavecast-pass"`), pkill the proxy,
   cancel wakeups.

## CI/CD posture (remote: github.com/musicofhel/wavecast)

Three gates, then delivery via a draft PR:

- **Per-pass gate (inside the pass):** narrow tests while iterating; full
  `pytest tests/ -q` (424 tests, ~9s) + `ruff check src/ tests/` before the
  final commit. Commit early/incrementally — a dead session must leave its
  progress in git.
- **Independent gate (orchestrator, every tick):** re-runs pytest + ruff
  itself and never trusts the pass's claim. Red suite after a committed pass
  is flagged to Aaron and queued as the next pass's first job.
- **Boundary audit (orchestrator, every tick):** branch unchanged, master
  unmoved, production ledger prefix-identical to the pre-pass snapshot
  (`~/.wavecast/loop-snapshots/ledger-pre-pass.jsonl`), live model dir
  untouched, crontab unchanged. Violation = stop the loop, tell Aaron.
- **Remote CI (orchestrator-side push, PASS-side never):** when the two gates
  and the audit are clean, the ORCHESTRATOR pushes
  `loop/system-update-v1` and keeps a **draft PR → master** open; the
  existing `.github/workflows/ci.yml` `pull_request` trigger runs
  ruff+pytest on ubuntu py3.11/3.12 (CPU fallback — the suite auto-detects)
  on every push. Verdict checked next tick via `gh pr checks`. The guard
  denies the ox-alpha pass `git push`/`gh` — the trusted side owns the
  remote, and only pushes verified states.
- **Delivery:** Aaron reviews the draft PR and merges when he wants;
  promotion of any scratch forward-test data into production stays a
  documented manual step, never automatic.

## Hard boundaries (guard-enforced; smoke-tested 18/18 on 2026-08-23)

Deny: `.env`/rc/.ssh/memory reads · writes outside `~/wavecast` +
`~/.wavecast` + `/tmp` · any write naming `models/d1_augmented_v1` or
`forward_tests/d1_forward_v1` (production; scratch runs go to
`~/.wavecast/forward_tests/loop_scratch/`) · git push/pull/fetch/checkout/
switch/reset/rebase/merge/stash/clean/remote · crontab/systemctl/sudo ·
ssh/scp/rsync · gh/claude/docker (no nested agents) · curl/wget.
Allow: normal dev (python/pytest/make/maturin/pip), git add/commit/restore on
the loop branch, reads everywhere non-secret.
Never widen `--allowedTools` to `--dangerously-skip-permissions`; never load
the hooks via project `settings.json` (binds every session — the autoapply
lesson).

## Known failure modes

- **Empty-turn deaths (ox-alpha):** intermittent empty/thinking-only turns;
  two in a row kill a session; deaths cluster past ~120k context. The proxy
  retries empties upstream (8×, spaced) — zero pass failures across 47 proxied
  autoapply passes. Recovery for a dead pass: partial work is already in
  git/report; retry once, then 30-min backoff.
- **Stalls:** live pid, no commit AND no report growth for ~45 min → kill,
  keep partial work, relaunch once.
- **Massive API 429s:** the forward cron's current failure (task A2 fixes the
  fetcher). Pass-run python may hit the same; that is data for A2, not a loop
  fault.
- **Model wall on the orchestrator side:** if the Fable session dies, the
  running pass finishes unattended and its work sits in git + research/ — the
  next orchestrator tick picks it up from disk (everything needed is in this
  file + LOOP_MISSION.md + INDEX.md).

## Ground truth at build time (2026-08-23)

Suite env verified (torch 2.10.0+cu128, CUDA true). Forward ledger: 1880 rows,
1860 resolved; unfiltered acc 32.6% (random-level, expected); **A2i-filtered
55.5% (167/301) vs archived claim 66.7%** — adjudicating that gap is task A4.
Cron failing on 429s since ~08-19 (JPM MaxRetryError kills whole run).

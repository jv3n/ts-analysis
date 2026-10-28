# CLAUDE.md

Source of truth for the conventions of `ts-analysis`. Read this first.

## Project

`ts-analysis` is part of **Ticker Story**, the user's multi-repo trading platform (`ts-` is the
platform prefix, not TypeScript). It is a **POC** that measures **how accurate a forecasting model's
predictions are** on US small caps. If the model says a ticker will drop 30 %, did it actually
drop 30 %? For now, that is the whole goal. Scope and open questions:
[#1](https://github.com/jv3n/ts-analysis/issues/1).

- **Independent** — it runs alongside [`trade`](https://github.com/jv3n/trade) (the platform's
  tracker app) and [`ts-scanner`](https://github.com/jv3n/ts-scanner) (the D-1 GUS / double-top
  scanner), but reads none of their data, pushes nothing to them and shares no code with them.
- **Full local** — nothing is deployed. The model runs on the user's machine.
- **The model is Timestral**, Mistral's family of pretrained time-series forecasting models,
  released on 2026-10-15. It comes in four sizes (5M · 28M · 123M · 209M), with open weights on
  Hugging Face under a modified MIT license. It runs on CPU and takes up to 32k past steps, past
  covariates (volume) and known-future covariates. It returns **probabilistic forecasts on 21
  quantiles**, not a single number. Mistral reports ~10 ms a forecast on 5M and ~140 ms on 209M
  (8 CPU threads); its benchmarks are vendor-reported. Start with 28M or 123M and keep 209M as the
  ceiling.
- **Market data comes from Massive** (formerly Polygon.io) on the **free Basic plan**: end-of-day
  data (published the next day), REST only, **5 requests a minute**, no snapshots, no WebSocket.
  Every call is batched with that quota in mind. The plan is said to keep about **2 years of 1-min
  history**, to be checked on the account.
- **No trading, no signal** — it only measures.

## How the POC works

1. **Pick the tickers** — names in a **big gap**, e.g. up **≥ 40 %** on the previous close at some
   point of the day (40 % is an example, the threshold is open). One grouped-daily request finds the
   candidates from the day's high, among **common stocks only**: warrants, rights and units gap 40 %
   on a few cents and move like options. Take every candidate, or **a random ~10 of them**, never the top
   10 by the day's high: that would only keep the ones that went on running and bias every figure.
2. **Fetch** — the day's 1-min bars of each picked ticker, about 18 requests (~3.6 min of quota)
   a day with the common-stock list, stored locally. **Backfill from the start**: past days are fetched right away instead of
   waiting for them to pile up. A month of sessions is ~75 min of quota, and 6 months a few hours,
   which can run overnight. That gives thousands of forecasts to score in the first week.
3. **Replay with simulated time** — each stored day is played back from the open, minute by minute.
   At each step the model **only sees the bars up to that minute** and forecasts what follows. A
   replay must never read a bar from its own future.
4. **Score** — since the real future is known, each forecast, recorded as (ticker, day, cut minute,
   horizon, predicted move), is set against the real move over the same horizon:
   - **direction**: it said down, did it go down?
   - **magnitude**: predicted −30 %, real −x %, and the gap between them;
   - **calibration**: the real price should fall inside the 10–90 % band about 80 % of the time;
   - every figure is set against a **naive baseline**, the last price (flat or as a random walk). A
     model that does not beat it predicts nothing.

## Pitfalls to keep in mind

- **What counts as a call is a rule to settle.** The quantiles' median sits near zero most of the
  time. A call might be "the median says −10 % or more", or "90 % sure the price ends below X". That
  rule is what makes "it said −30 %, did it drop 30 %?" measurable.
- **Small-cap bars have holes.** Many minutes have no trade, so no bar. Fill the holes (last price
  carried forward, volume 0) before feeding the model, and prefer log prices to raw prices.
- **Small caps are noisy, thin and driven by news** the model never sees. A generic forecaster may
  only get the volatility right. The user built this repo precisely to test that on real data over
  a long period: **the figures decide**, a negative result is a valid outcome, and neither the
  model's reputation nor an intuition settles it.

## Later, not in the POC

- **A replay view** — a web page that plays a day back as if it were live: the candles minute by
  minute, the forecast band drawn ahead of them, then the real price showing whether it was right.
- **Recording the KPIs** — likely a database to keep each prediction and its outcome and follow
  them over time. Its shape comes after the POC; a flat file is enough until then.
- **A real-time mode**, only if the figures hold up, and on a paid real-time plan. The model runs
  during the session and starts on a ticker **as soon as it enters the top list**: absent at 9:30,
  showing up at 10:00 at +40 %, it is picked up at 10:00 and forecast from that minute on.

## Stack

Python, unlike the rest of the platform (Kotlin, Angular), because Timestral's tooling is Python.
It runs in **WSL**, with the repo on the Linux filesystem (`~/dev`, never `/mnt/c`).

| Layer        | Tech                                                         |
| ------------ | ------------------------------------------------------------ |
| Python       | 3.13, managed by **uv** (`uv.lock` committed) |
| Market data  | `httpx` against Massive's REST API                           |
| Storage      | `projects/analysis/data/` (git-ignored): JSON for grouped bars, splits and picks, parquet (`pandas` + `pyarrow`) for raw 1-min bars |
| Tests        | `pytest`, no network; GitHub Actions on every PR and push to `master` |

```
projects/analysis/          # the Python project: pyproject.toml, uv.lock, data/ (git-ignored)
├── src/ts_analysis/
│   ├── selection.py        # gap rule + random sample (pure)
│   ├── massive.py          # Massive client + rate limiter
│   ├── store.py            # data/ layout
│   ├── fetch.py            # backfill orchestration + `ts-fetch` CLI
│   ├── series.py           # minute grid: holes filled, log prices, what a cut sees (pure)
│   └── replay.py           # cuts, baselines, Timestral + `ts-replay` CLI
└── tests/
devops/                     # README (index) + docs/github.md: merge rules, master protection, hook, CI
```

From `projects/analysis/` (`.env` stays at the repo root, found from there):

```bash
uv sync                                               # install, creates .venv
uv run pytest                                         # tests
uv run ts-fetch --start 2026-10-01 --end 2026-10-27   # backfill; --end defaults to yesterday ET
uv run ts-replay --start 2026-10-01 --end 2026-10-27  # forecasts → data/forecasts/<model>/<day>.parquet
```

`ts-fetch` is resumable: rerun the same command after an interruption, it makes no request for
what is already in `data/`.

Pure functions for every rule (selection, replay cut, scoring), with the I/O wired around them, so
the rules are tested without the network.

## Conventions

### Comments — the strict minimum

Code says *what*. A comment is only worth its line when it says *why*, and only when the why isn't
obvious. Keep a non-obvious constraint (an API quirk, a value verified empirically, a warning about
what breaks). Drop anything that restates the code or narrates history: git holds the history and
the issue holds the rationale.

### Code

- Secrets never go in code or in a committed file: `.env` (git-ignored) locally, and `.env.example`
  documents the keys (the Massive API key first).
- **Never log PII.**

### Tests as documentation

- Test names are full sentences that describe the behaviour.
- Realistic fixtures: a real ticker and real prices beat `"x"`.
- The replay is tested against look-ahead: a forecast made at minute *t* sees no bar after *t*.

### Commits

- Conventional Commits in **English**, **title only**: one line, under 72 chars, no body, no footer.
- **No reference to Claude**: no `Co-Authored-By` and no « Generated with », in commits, PR titles
  or descriptions.
- **Every commit is linked to an issue**, its number prefixing the scope, e.g.
  `docs(1/repo): start the README`. No issue → ask the user which one to use.
- **Default = suggest, don't execute**: never run `git add/commit/push/branch/tag/rebase` or
  `gh pr/issue` unless the user asks in the current turn. Authorization does not carry forward to
  later turns. Never push to `master`: its protection lets admins bypass it and allows
  force-pushes, so it would not stop you. Its rules and the `.githooks/pre-push` guard are in
  [`devops/docs/github.md`](../devops/docs/github.md).
- See the [`git-commit`](./skills/git-commit/SKILL.md) skill.

### Pull requests

- **No work without an issue, one PR per issue.** The branch is named `<issue>-<slug>`, e.g.
  `3-fetch-gap-bars`, and is cut from an up-to-date `master`.
- **One commit per PR**: a fix after review is amended and force-pushed to the PR branch, never
  to `master`. Before pushing, check the PR is still open: a merged PR's branch is deleted, and a
  push would recreate it.
- The body opens with `Refs #N` (and the epic, e.g. `part of epic #1`), **never `Closes` /
  `Fixes` / `Resolves`**, then says what changed and what to run to check it.
- **Only the user merges**, by rebase: merge commits and squash are disabled, so `master` stays
  linear. Never run `gh pr merge`.
- GitHub deletes the remote branch on merge. Locally, suggest `git checkout master && git pull
  && git fetch --prune && git branch -D <branch>`.
- See the [`github-create-pull-request`](./skills/github-create-pull-request/SKILL.md) skill, and
  [`code-review`](./skills/code-review/SKILL.md) for a read-only review before the commit.

### Backlog

Issues live in this repo and on its own board, **TS Analysis Board** (project 3, owner `jv3n`),
separate from the TickerStory Board that `trade` and `ts-scanner` share.

- A new issue goes on the board with **Status** and **Priority** set, and carries a `prio:*`
  (`P0`–`P3`), a `scope:*` (`product` / `tech`) and a type label (`enhancement` / `bug` /
  `documentation` / `question`).
- The POC is epic **#1**. Its steps are **sub-issues** of it, titled `#1 <Area>: <what>`, with a
  body that opens with `Part of #1` and ends with an `## Acceptance` checklist.
- Statuses: Backlog → Ready → In progress → In review → Test → Done. The issue goes to
  **In review** when its PR opens. Once the PR is merged, suggest **Test** when the change still
  has to be tried on real data (a backfill on the account, a replay run), **Done** otherwise.
  Only the user moves an issue to Done.
- Creating an issue is fine when the user asks for one. Never close or reopen an issue, edit
  labels or move a card on your own: suggest it, and the user confirms in the current turn.

## Languages

| Where                                                 | Language   |
| ----------------------------------------------------- | ---------- |
| Talking to the user in the terminal                   | **French** |
| Source code, comments, docstrings, test names, logs   | **English** |
| GitHub — commits, PRs, issues, comments               | **English** |
| Docs — `README.md`, `.claude/`                        | **English** |

## Instructions for Claude

- **Line endings** — LF only. Add a `.gitattributes` (`* text=auto eol=lf`) with the first commit:
  IntelliJ on Windows tends to bring CRLF back.
- **Builds and tests** — never run them autonomously. Say what is worth running and let the user
  drive it, unless they ask in the current turn.

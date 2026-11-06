# ts-analysis

Part of **Ticker Story**, but independent of [`trade`](https://github.com/jv3n/trade) and
[`ts-scanner`](https://github.com/jv3n/ts-scanner): it reads none of their data and pushes nothing to them.

A **POC** to measure how accurate a forecasting model is on US small caps. The model is
[Timestral](https://huggingface.co/mistralai), Mistral's family of time-series forecasting models,
run locally on CPU. If the model says a ticker will drop 30 %, did it actually drop 30 %?

Scope, measures and open questions: [#1](https://github.com/jv3n/ts-analysis/issues/1).

## How it works

1. **Fetch**: the 1-min bars of the days to test, from **Massive** (formerly Polygon.io) on the free
   Basic plan (end of day, REST, 5 requests a minute). It takes about ten tickers a day in a big gap,
   for example up 40 % or more on the previous close, picked at random among the candidates. Past
   days are backfilled from the start, and the bars are stored locally.
2. **Replay**: each stored day is played back from the open, minute by minute. At each step the model
   only sees the bars up to that minute and forecasts what follows.
3. **Score**: each forecast is set against what really happened over the same horizon (direction,
   magnitude, calibration of its quantile band) and against a naive baseline, the last price. A model
   that does not beat the baseline predicts nothing.

## Setup

Python 3.13 with [uv](https://docs.astral.sh/uv/), in WSL.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once
cp .env.example .env                               # then fill MASSIVE_API_KEY
cd projects/analysis
uv sync
uv run pytest
```

[CI](.github/workflows/ci.yml) runs the same tests on every pull request and every push to
`master`, and fails if `uv.lock` is out of date with `pyproject.toml`. The CI, the merge rules and
the git hook are described in [`devops/`](devops/README.md).

## Fetch

From `projects/analysis/`, where the bars land in `data/`:

```bash
uv run ts-fetch --start 2026-10-01 --end 2026-10-27
```

For each session it picks the tickers whose high reached **+40 %** on the previous close (previous
close in **$0.30–$10**), among **common stocks** only (no warrants, rights or units), draws **10 of them at random**, and stores their 1-min bars. Prices are
**raw**, not split-adjusted, so every stored day stays consistent whenever it was fetched; a ticker
whose split executes that day is left out, since its raw gap is meaningless.

Each picked ticker also gets its **5 sessions before the day**
([#26](https://github.com/jv3n/ts-analysis/issues/26)), in one request over the 10 calendar days
before it. Those are fetched split-adjusted, so a reverse split inside the window does not read as a
×10 move, then brought back to the prices the day traded at: Massive adjusts for every split up to
the fetch, and the day's raw bars set against their adjusted copy undo the splits that came after.
Days fetched before this have their past sessions fetched on the next run, and nothing else.

`--rule scanner` ([#39](https://github.com/jv3n/ts-analysis/issues/39)) picks `ts-scanner`'s population
instead: the **premarket GUS** and the names that **pushed during the session**. The grouped daily
bars hold the regular session only, so the candidates are the names that opened 15 % up (what every
GUS shows there), or whose session high reached +25 % on the previous close, or whose session range
reached 40 %: about 45 a day, 10 drawn at random as with the default rule. Each pick then gets its
setup and entry minute from its bars, all known at the time:

- a **GUS** (previous close → premarket high ≥ +45 %, premarket open → premarket high ≥ +15 %,
  previous close → open ≥ +15 %, premarket high → open ≥ −55 %, the thresholds of
  `trade/docs/pattern/GUS.md`) enters on the session's first bar;
- any other name enters on the first session bar where its running high reaches +25 % on the
  previous close, or its running range 40 %: the minute a live scanner would list it.

A new rule means new picks, so it goes in its own `--data` folder. The grouped bars, splits, common
stocks, bars and past sessions do not depend on the rule and are shared by linking them:

```bash
mkdir -p data-scanner && for kind in grouped splits common bars past; do ln -s ../data/$kind data-scanner/$kind; done
uv run ts-fetch --rule scanner --data data-scanner --start 2026-05-01
```

A day costs 28 requests (grouped bars, splits, the day's common stocks over 6 pages, 10 tickers
twice): about 5.6 min at 5 requests a minute, so a month takes about 2 h. A day is only published after midnight ET; `--end` defaults to
yesterday ET, and the run stops cleanly at the first day not published yet. Rerun the same command
after an interruption: it resumes where it stopped. A rerun with other settings on days already
picked fails rather than mixing two rules. The thresholds are `--min-gap`, `--min-price`,
`--max-price` and `--per-day`.

```
data/
├── grouped/<day>.json            # every stock's daily bar
├── splits/<day>.json             # splits executed that day
├── common/<day>.json             # common stocks listed that day, delisted ones included
├── picks/<day>.json              # settings, candidates (prev close, high, gap), picked tickers
├── bars/<day>/<ticker>.parquet   # 1-min bars, UTC, extended hours included (04:00–20:00 ET)
└── past/<day>/<ticker>.parquet   # the same, over the 5 sessions before, in the day's prices
```

## Replay

```bash
uv run ts-replay --start 2026-10-01 --end 2026-10-27   # --model 28M by default, --step 5
```

Each stored day is played back from 9:30 to 15:55 ET, one cut every `--step` minutes. At a cut the
model sees the bars that started before it, from 4:00 ET, minutes with no trade filled with the last
close and volume 0, as log prices with the volume as a past covariate. It forecasts **+5, +15,
+30 min and the close**, on 21 quantiles. Two baselines forecast the same cuts: **flat** (the last
price) and a **random walk** at the last hour's volatility.

`--wrapping` sets how the prices reach Timestral, each stored under its own model name
([#21](https://github.com/jv3n/ts-analysis/issues/21)). The default is **`log-capped`** (log prices,
no non-negative clip, quantiles capped at 10 stds beyond the context's range,
`timestral-28M-log-capped`). Timestral inverts its scaling with `sinh`, so its uncapped tails blow
up: on October the original wrapping, `log` (`timestral-28M`), gave 1,252 forecasts beyond ×150,
`log-capped` 30. `raw` (raw prices) and `raw-nan` (raw prices, minutes with no trade left missing)
only move the blow-up below zero, where the clip floors it; a raw quantile clipped to 0 reads as a
price of $0.0001. They stay to reproduce the measure.

On top of `log-capped` ([#24](https://github.com/jv3n/ts-analysis/issues/24)): `clock` gives Timestral
the time of day as a known covariate (the share of the session elapsed, 0 at 9:30, 1 at 16:00), since
intraday volatility follows a U; `asym` turns its symmetric averaging off, which otherwise makes a
fading pump and a rebounding crash mirror images; `clock-asym` does both. On October none beat
`log-capped`: the clock changes nothing in any slot of the day, and without the symmetric averaging
every horizon gets worse. They stay to reproduce the measure.

`--past-days N` (0 to 5, 5 by default) puts the N sessions before the day in front of the context,
each from 4:00 to its close, the close carried over the night: without them Timestral never sees the
previous close, so not the gap the ticker was picked for. A name that traded on fewer sessions gets
the missing ones as NaN. The clock restarts every day. The model name gets `-pastN`, e.g.
`timestral-28M-log-capped-past3`; a day whose past sessions are not fetched yet is skipped. On
October 5 sessions beat 1 and 3 at every horizon, and the edge held on a held-out week
([#26](https://github.com/jv3n/ts-analysis/issues/26)), hence the default; `--past-days 0` gives the
day alone, as the earlier variants were measured.

`log-capped-mask` passes Timestral an `observation_mask`, False on the minutes without a trade: the
carried price stays in the context, but the model knows it was not traded. Over several sessions the
holes pile up (pre-market, overnight, thin names), which is what it tests.

Each ticker is forecast from its **entry** only ([#37](https://github.com/jv3n/ts-analysis/issues/37),
see Score): the cuts before it read the future through the pick, and the score drops them anyway,
so the replay does not spend time on them. `--all-cuts` forecasts every cut from the open.

The first run downloads the weights from Hugging Face. A day already replayed for a model is
skipped, so a rerun resumes.

```
data/forecasts/<model>/<day>.parquet   # model, ticker, day, cut, horizon, to_close, last, actual,
                                       # q01…q99 (log move: log(price / last)), ms_per_forecast
```

The quantiles are kept as log moves, the scale the model forecasts in: at long horizons the upper
tail is wide in log space, and turned into a price it reaches absurd values (×6,000 at q99).

## Calibrate

```bash
uv run ts-calibrate   # --model to pick, --min-days 5, --window N to learn from the last N days only
```

Corrects the stored forecasts after the fact, from the model's own past errors: Timestral stays frozen
and nothing is trained. Walk-forward, per model and horizon (+5, +15, +30, close), the correction of
day J is learned from the days before J only:

- **shift**: the median moves by the typical miss, so that half the past real moves fall below it;
- **band**: each side of the band is then scaled by one factor, so that 80 % of the past real moves
  fall inside the 10–90 % band.

The corrected forecasts go to `data/forecasts/<model>-cal/`, scored by `ts-score` like any model, so
the correction has to earn its place against the raw model and the baselines. Every stored model is
corrected except `flat`, which has no band, the random walk included: a calibrated random walk is the
fair baseline for a calibrated Timestral. Like the score, it learns from the cuts after the ticker
reached its gap only (`--all-cuts` to keep them all). A day with fewer than 5 days before it is left out, and the
`-cal` folders are rebuilt whole on each run.

## Score

```bash
uv run ts-score   # --call 0.05: the median move that makes a call; --big 0.05: the real move that makes a big one
uv run ts-score --start 2026-10-28 --end 2026-11-04   # only those days, e.g. a held-out week
```

Reads every stored forecast and writes `data/reports/score.md`, by horizon (+5, +15, +30, close) and
by model.

Only the cuts made **after the ticker reached its gap** are scored
([#37](https://github.com/jv3n/ts-analysis/issues/37)). A ticker is picked on the day's high, known at
the close, so at a cut before its first bar at +40 % the pick itself tells the future: on October,
82 % of those cuts ended higher at the close, by +13 % at the median, against a coin flip after. The
entry minute is that first bar's, from the picks and the stored bars, and a cut sees it once it has
closed. `--all-cuts` keeps every cut, to reproduce the earlier figures.

- **accuracy**: the pinball loss (mean quantile loss on log moves) and its **skill** against the
  random walk, above 0 beats it; the mean and median error of the median;
- **direction**: the median and the real move of the same sign, against the share of real moves
  down, the bar a call has to clear;
- **calls**: the median says 5 % or more, or the band sits wholly on one side of 0 (90 % sure), and
  how often the price went that way;
- **calibration**: the real price inside the 10–90 % band (80 % expected), and the share below each
  quantile;
- **big moves**: the cuts followed by a real move of 5 % or more, picked on the real move only: did
  the median lean the right way (also on the cuts where the move had not started yet), and was the
  band wider than the ticker-day's usual one just before?

The cuts of one ticker-day share its fate, so the forecasts are far from independent: hundreds of
thousands of them are closer to a few hundred samples. The skill, coverage, direction and call hit
rates get a **95 % interval** from a bootstrap over the ticker-days (`--resamples 1000`, `--seed 0`),
every model resampled on the same draws, so a `skill_95` wholly above 0 beats the random walk. The
calls also say how many ticker-days they came from.

Once `ts-calibrate` has run, `skill_cal` sets each model against the calibrated random walk, the fair
bar for a calibrated model: on October, calibrating the random walk alone gains it 1 to 2 points.
Every skill is computed on the ticker-days the model and its reference share. When the scored days
span several months, a **by month** section gives the skill and its interval month by month, to tell
an edge from a regime.

## Monitor

```bash
uv run ts-monitor   # http://localhost:8765, --port to change it
```

A local page to follow the runs without reading logs or parquet files, served from `projects/monitor/`
(plain HTML, CSS and JavaScript, no build step) on `localhost` only:

- **progress**: where a running `ts-fetch` or `ts-replay` is (day, ticker or cut), refreshed every
  2 s from `data/status/<job>.json`. A job killed without writing its end shows as **lost**. The
  fetch also shows its length, known at the start since the quota paces it: the requests still to
  make given what `data/` holds, 12 s each, with the time it should end;
- **forecast**: for a day, a model replayed on it and a ticker, the price through the session with a cut's 10–90 %
  and 25–75 % bands drawn ahead of it, the random walk's band next to it, and whether the real price
  landed inside;
- **replay**: the forecasts per day and model, and the time per forecast;
- **fetch**: the stored days, their candidates and picks, the bars per ticker, and the weekdays
  missing from the range.

## Later

- A web view that replays a day as if it were live: the candles minute by minute, the forecast band
  ahead of them, then the real price.
- If the figures hold up, a real-time mode on a paid plan: the model starts on a ticker as soon as it
  enters the top list.
- Context the model does not see yet, news, SEC filings and the float
  ([#30](https://github.com/jv3n/ts-analysis/issues/30)): where to get a dated float is in
  [`docs/float-sources.md`](docs/float-sources.md).

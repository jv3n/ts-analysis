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

A day costs 18 requests (grouped bars, splits, the day's common stocks over 6 pages, 10 tickers):
about 3.6 min at 5 requests a minute, so a month takes about 75 min. A day is only published after midnight ET; `--end` defaults to
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
└── bars/<day>/<ticker>.parquet   # 1-min bars, UTC, extended hours included (04:00–20:00 ET)
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

The first run downloads the weights from Hugging Face. A day already replayed for a model is
skipped, so a rerun resumes.

```
data/forecasts/<model>/<day>.parquet   # model, ticker, day, cut, horizon, to_close, last, actual,
                                       # q01…q99 (log move: log(price / last)), ms_per_forecast
```

The quantiles are kept as log moves, the scale the model forecasts in: at long horizons the upper
tail is wide in log space, and turned into a price it reaches absurd values (×6,000 at q99).

## Score

```bash
uv run ts-score   # --call 0.05: the median move that makes a call; --big 0.05: the real move that makes a big one
```

Reads every stored forecast and writes `data/reports/score.md`, by horizon (+5, +15, +30, close) and
by model:

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

## Monitor

```bash
uv run ts-monitor   # http://localhost:8765, --port to change it
```

A local page to follow the runs without reading logs or parquet files, served from `projects/monitor/`
(plain HTML, CSS and JavaScript, no build step) on `localhost` only:

- **progress**: where a running `ts-fetch` or `ts-replay` is (day, ticker or cut), refreshed every
  2 s from `data/status/<job>.json`. A job killed without writing its end shows as **lost**;
- **forecast**: for a model, a day and a ticker, the price through the session with a cut's 10–90 %
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

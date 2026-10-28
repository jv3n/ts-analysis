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

## Status

Nothing is built yet. Stack, setup and commands come with the first PR.

## Later

- A web view that replays a day as if it were live: the candles minute by minute, the forecast band
  ahead of them, then the real price.
- If the figures hold up, a real-time mode on a paid plan: the model starts on a ticker as soon as it
  enters the top list.

import argparse
import logging
import warnings
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from statistics import NormalDist
from time import perf_counter
from typing import Protocol

import numpy as np
import pandas as pd

from ts_analysis.fetch import last_published_day, weekdays
from ts_analysis.series import CLOSE, OPEN, Grid, clock_of, minute_grid
from ts_analysis.store import Store

log = logging.getLogger("ts_analysis.replay")

QUANTILES = (0.01, 0.05, *(round(0.10 + 0.05 * i, 2) for i in range(17)), 0.95, 0.99)
HORIZONS = (5, 15, 30)
VOL_WINDOW = 60


class Forecaster(Protocol):
    name: str

    def forecast(self, log_close: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
        """(series, time) in, (series, length, quantile) log prices out."""
        ...


class Flat:
    name = "flat"

    def forecast(self, log_close: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
        return np.broadcast_to(log_close[:, -1, None, None], (len(log_close), length, len(QUANTILES))).copy()


class RandomWalk:
    """Gaussian log-returns at the volatility of the last hour, so the band widens as sqrt(horizon)."""

    name = "random-walk"

    def forecast(self, log_close: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
        returns = np.diff(log_close[:, -VOL_WINDOW - 1 :], axis=1)
        with np.errstate(invalid="ignore"), warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            sigma = np.nan_to_num(np.nanstd(returns, axis=1))
        z = np.array([NormalDist().inv_cdf(q) for q in QUANTILES])
        steps = np.sqrt(np.arange(1, length + 1))
        return log_close[:, -1, None, None] + sigma[:, None, None] * steps[None, :, None] * z[None, None, :]


class Timestral:
    def __init__(self, size: str):
        from timestral import TimestralForecaster

        self.name = f"timestral-{size}"
        self._model = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}")

    def forecast(self, log_close: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
        out = self._model.forecast(
            log_close[:, None, :], prediction_length=length, past_covariates=np.log1p(volume)[:, None, :]
        )
        if not np.allclose(out.quantile_levels, QUANTILES):
            raise ValueError(f"Unexpected quantile levels {out.quantile_levels}")
        return np.asarray(out.quantiles)[:, 0]


def cuts(step: int) -> range:
    return range(OPEN, CLOSE, step)


def replay_day(day: date, grids: dict[str, Grid], models: Iterable[Forecaster], step: int) -> list[dict]:
    """Each model forecasts every ticker at every cut from what the cut sees; the grid is only read
    past the cut to score the forecast."""
    models = list(models)
    rows = []
    day_cuts = cuts(step)
    for done, cut in enumerate(day_cuts, 1):
        if cut > OPEN and (cut - OPEN) % 60 == 0:
            log.info(
                "%s   %s  %d/%d cuts, %d forecasts per model so far",
                day, clock_of(cut), done - 1, len(day_cuts), len(rows) // len(models),
            )
        seen = {ticker: grid.seen_at(cut) for ticker, grid in grids.items()}
        tickers = [ticker for ticker, grid in seen.items() if not np.isnan(grid.log_close[-1])]
        if not tickers:
            continue
        log_close = np.stack([seen[ticker].log_close for ticker in tickers])
        volume = np.stack([seen[ticker].volume for ticker in tickers])
        length = CLOSE - cut
        horizons = [h for h in HORIZONS if h < length] + [length]
        for model in models:
            started = perf_counter()
            predicted = model.forecast(log_close, volume, length)
            ms = (perf_counter() - started) * 1000 / len(tickers)
            for i, ticker in enumerate(tickers):
                last = log_close[i, -1]
                for h in horizons:
                    moves = np.expm1(predicted[i, h - 1] - last)
                    rows.append(
                        {
                            "model": model.name,
                            "ticker": ticker,
                            "day": day.isoformat(),
                            "cut": clock_of(cut),
                            "horizon": h,
                            "to_close": h == length,
                            "last": float(np.exp(last)),
                            "actual": grids[ticker].price_at(cut + h),
                            **{f"q{round(q * 100):02d}": float(m) for q, m in zip(QUANTILES, moves)},
                            "ms_per_forecast": ms,
                        }
                    )
    return rows


def day_grids(store: Store, day: date) -> dict[str, Grid]:
    picks = store.load_picks(day)
    if picks is None:
        return {}
    return {
        ticker: minute_grid(store.load_bars(day, ticker), day)
        for ticker in picks["picked"]
        if store.has_bars(day, ticker)
    }


def replay(store: Store, start: date, end: date, models: list[Forecaster], step: int) -> None:
    days = list(weekdays(start, end))
    started = perf_counter()
    for number, day in enumerate(days, 1):
        todo = [model for model in models if not store.has_forecasts(model.name, day)]
        if not todo:
            log.info("%s day %d/%d already replayed", day, number, len(days))
            continue
        grids = day_grids(store, day)
        if not grids:
            log.info("%s day %d/%d no stored bars", day, number, len(days))
            continue
        log.info("%s day %d/%d %d tickers: %s", day, number, len(days), len(grids), " ".join(grids))
        frame = pd.DataFrame(replay_day(day, grids, todo, step))
        for model in todo:
            rows = frame[frame["model"] == model.name]
            store.save_forecasts(model.name, day, rows)
            log.info(
                "%s   %-16s %5d forecasts, %.1f ms each", day, model.name, len(rows), rows["ms_per_forecast"].mean()
            )
        log.info("%s day %d/%d done, %.0f s elapsed", day, number, len(days), perf_counter() - started)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay the stored days and forecast them minute by minute.")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=last_published_day())
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", default="28M", choices=["5M", "28M", "123M", "209M"])
    parser.add_argument("--step", type=int, default=5, help="minutes between two cuts")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error(f"--start {args.start} is after --end {args.end}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info("loading Timestral-%s (downloaded from Hugging Face on the first run)", args.model)
    started = perf_counter()
    model = Timestral(args.model)
    log.info("loaded in %.1f s", perf_counter() - started)
    replay(Store(args.data), args.start, args.end, [Flat(), RandomWalk(), model], args.step)


if __name__ == "__main__":
    main()

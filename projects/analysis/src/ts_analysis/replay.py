import argparse
import logging
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import NormalDist
from time import perf_counter
from typing import Protocol

import numpy as np
import pandas as pd

from ts_analysis.fetch import last_published_day, weekdays
from ts_analysis.series import CLOSE, OPEN, Grid, clock_of, minute_grid
from ts_analysis.status import Status
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


# A raw-price quantile clipped to 0 still needs a log: below a hundredth of a cent the name is gone.
PRICE_FLOOR = 1e-4


@dataclass(frozen=True)
class Wrapping:
    """How the grid reaches Timestral. Timestral scales each series with asinh and inverts it with sinh,
    so on log prices its tails are exponential twice over; and its non-negative clip, meant for prices,
    floors a log price at 0, i.e. a price at $1."""

    raw: bool = False
    nan_holes: bool = False
    clip_nonnegative: bool = True
    cap_k: float = 1e4
    clock: bool = False
    symmetric: bool = True


WRAPPINGS = {
    "log": Wrapping(),
    "raw": Wrapping(raw=True),
    "raw-nan": Wrapping(raw=True, nan_holes=True),
    # The one kept after #21: on October it cut the forecasts beyond x150 from 1,252 to 30, at no cost.
    "log-capped": Wrapping(clip_nonnegative=False, cap_k=10.0),
    "clock": Wrapping(clip_nonnegative=False, cap_k=10.0, clock=True),
    # Symmetric averaging makes the forecast odd by construction: a fading pump and a rebounding
    # crash get mirrored answers, and that asymmetry is what gap-ups are about.
    "asym": Wrapping(clip_nonnegative=False, cap_k=10.0, symmetric=False),
    "clock-asym": Wrapping(clip_nonnegative=False, cap_k=10.0, clock=True, symmetric=False),
}


def session_clock(steps: int) -> np.ndarray:
    """The share of the session elapsed at each grid slot: 0 at 9:30, 1 at 16:00, negative before the
    open. The grid starts at 4:00 for every ticker, so the slot index is the clock."""
    return (np.arange(steps) - OPEN) / (CLOSE - OPEN)


class Timestral:
    def __init__(self, size: str, wrapping: str = "log", model=None):
        self.name = f"timestral-{size}" if wrapping == "log" else f"timestral-{size}-{wrapping}"
        self.wrapping = WRAPPINGS[wrapping]
        if model is None:
            from timestral import TimestralForecaster

            model = TimestralForecaster.from_pretrained(
                f"mistralai/Timestral-{size}", quantile_real_cap_k=self.wrapping.cap_k
            )
        self._model = model

    def forecast(self, log_close: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
        context = np.exp(log_close) if self.wrapping.raw else log_close.copy()
        if self.wrapping.nan_holes:
            context[volume == 0] = np.nan
        known = None
        if self.wrapping.clock:
            clock = session_clock(context.shape[1] + length)
            known = np.broadcast_to(clock, (len(context), 1, len(clock))).copy()
        out = self._model.forecast(
            context[:, None, :],
            prediction_length=length,
            past_covariates=np.log1p(volume)[:, None, :],
            known_covariates=known,
            clip_nonnegative=self.wrapping.clip_nonnegative,
            symmetric_averaging=self.wrapping.symmetric,
        )
        if not np.allclose(out.quantile_levels, QUANTILES):
            raise ValueError(f"Unexpected quantile levels {out.quantile_levels}")
        quantiles = np.asarray(out.quantiles, dtype="float64")[:, 0]
        return np.log(np.maximum(quantiles, PRICE_FLOOR)) if self.wrapping.raw else quantiles


def cuts(step: int) -> range:
    return range(OPEN, CLOSE, step)


def replay_day(
    day: date,
    grids: dict[str, Grid],
    models: Iterable[Forecaster],
    step: int,
    on_cut: Callable[[str, int, int], None] | None = None,
) -> list[dict]:
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
        if on_cut:
            on_cut(clock_of(cut), done, len(day_cuts))
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
                    # Log moves: the long-horizon upper tail is wide in log space, and exp blows it up.
                    moves = predicted[i, h - 1] - last
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
    info = {"start": start.isoformat(), "end": end.isoformat(), "days": len(days), "models": [m.name for m in models]}
    with Status(store.root, "replay", **info) as status:
        for number, day in enumerate(days, 1):
            status.update(day=day.isoformat(), day_number=number, cut=None, cut_number=0, cuts=0)
            todo = [model for model in models if not store.has_forecasts(model.name, day)]
            if not todo:
                log.info("%s day %d/%d already replayed", day, number, len(days))
                continue
            grids = day_grids(store, day)
            if not grids:
                log.info("%s day %d/%d no stored bars", day, number, len(days))
                continue
            log.info("%s day %d/%d %d tickers: %s", day, number, len(days), len(grids), " ".join(grids))

            def on_cut(cut: str, done: int, total: int) -> None:
                status.update(cut=cut, cut_number=done, cuts=total)

            frame = pd.DataFrame(replay_day(day, grids, todo, step, on_cut))
            for model in todo:
                rows = frame[frame["model"] == model.name]
                store.save_forecasts(model.name, day, rows)
                log.info(
                    "%s   %-16s %5d forecasts, %.1f ms each",
                    day, model.name, len(rows), rows["ms_per_forecast"].mean(),
                )
            log.info("%s day %d/%d done, %.0f s elapsed", day, number, len(days), perf_counter() - started)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay the stored days and forecast them minute by minute.")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=last_published_day())
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", default="28M", choices=["5M", "28M", "123M", "209M"])
    parser.add_argument("--wrapping", default="log-capped", choices=list(WRAPPINGS), help="how the prices reach Timestral")
    parser.add_argument("--step", type=int, default=5, help="minutes between two cuts")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error(f"--start {args.start} is after --end {args.end}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info("loading Timestral-%s, %s wrapping (downloaded from Hugging Face on the first run)", args.model, args.wrapping)
    started = perf_counter()
    model = Timestral(args.model, args.wrapping)
    log.info("loaded in %.1f s", perf_counter() - started)
    replay(Store(args.data), args.start, args.end, [Flat(), RandomWalk(), model], args.step)


if __name__ == "__main__":
    main()

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ts_analysis.entry import after_entry, entries, with_setup
from ts_analysis.score import COLUMNS, load_forecasts, with_outcome
from ts_analysis.store import Store

log = logging.getLogger("ts_analysis.calibrate")

SUFFIX = "-cal"
SETUP_SUFFIX = "-cal-setup"
COVERAGE = 0.80
MIN_DAYS = 5
# A flat forecast has no band to stretch, and a corrected model is never corrected again.
SKIPPED = ("flat",)


def correction(past: pd.DataFrame) -> tuple[float, float]:
    """The shift and the band factor learned from forecasts whose outcome is known. The shift puts half
    the real moves below the median. The factor then scales the band around the shifted median, each
    side by its own width, so that 80 % of the real moves land inside q10–q90."""
    shift = float(np.median(past["real"] - past["q50"]))
    gap = past["real"] - past["q50"] - shift
    width = np.where(gap >= 0, past["q90"] - past["q50"], past["q50"] - past["q10"])
    with np.errstate(divide="ignore", invalid="ignore"):
        reach = np.abs(gap) / width
    reach = reach[np.isfinite(reach)]
    factor = float(np.quantile(reach, COVERAGE)) if len(reach) else 1.0
    return shift, factor


def corrected(today: pd.DataFrame, shift: float, factor: float) -> pd.DataFrame:
    out = today.copy()
    median = out["q50"].to_numpy()[:, None]
    out[COLUMNS] = median + shift + factor * (out[COLUMNS].to_numpy() - median)
    return out


def is_corrected(model: str) -> bool:
    return model.endswith((SUFFIX, SETUP_SUFFIX))


def calibrate(
    forecasts: pd.DataFrame, min_days: int = MIN_DAYS, window: int | None = None, by_setup: bool = False
) -> pd.DataFrame:
    """Walk-forward, per model and horizon bucket: the correction applied to day J is learned from the
    model's days before J only (the last `window` of them if given), never from J or after. A day with
    fewer than `min_days` before it is left out. The corrected forecasts keep the stored columns, under
    the model's name plus `-cal`. With `by_setup`, also per setup the ticker entered on, read from the
    `setup` column: a GUS is corrected from the past GUS only and `min_days` counts the days the setup
    was seen on, under `-cal-setup`."""
    frame = with_outcome(forecasts)
    frame["day"] = frame["day"].astype(str)
    keys, suffix = (["model", "bucket", "setup"], SETUP_SUFFIX) if by_setup else (["model", "bucket"], SUFFIX)
    out = []
    for (model, *_), group in frame.groupby(keys):
        days = sorted(group["day"].unique())
        for i, day in enumerate(days):
            before = days[max(0, i - window) if window else 0 : i]
            if len(before) < min_days:
                continue
            shift, factor = correction(group[group["day"].isin(before)])
            out.append(corrected(group[group["day"] == day], shift, factor).assign(model=model + suffix))
    if not out:
        return pd.DataFrame(columns=forecasts.columns)
    return pd.concat(out)[list(forecasts.columns)].reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Correct the stored forecasts walk-forward from the past days' errors.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", action="append", help="model to correct, repeatable; every stored one by default")
    parser.add_argument("--min-days", type=int, default=MIN_DAYS, help="past days needed before a day is corrected")
    parser.add_argument("--window", type=int, help="learn from the last N days only; all past days by default")
    parser.add_argument(
        "--by-setup", action="store_true", help="learn the correction per setup (gus, push), under -cal-setup"
    )
    parser.add_argument(
        "--all-cuts", action="store_true", help="keep the cuts before the ticker reached its gap, which read the future"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    forecasts = load_forecasts(args.data)
    if forecasts.empty:
        parser.exit(1, "error: no forecasts in data/forecasts, run ts-replay first\n")
    stored = sorted(forecasts["model"].unique())
    models = args.model or [m for m in stored if m not in SKIPPED and not is_corrected(m)]
    if unknown := sorted(set(models) - set(stored)):
        parser.exit(1, f"error: no forecasts for {' '.join(unknown)}\n")

    store = Store(args.data)
    entry = entries(store)
    if not args.all_cuts:
        forecasts = after_entry(forecasts, entry)
    if args.by_setup:
        forecasts = with_setup(forecasts, entry)
    suffix = SETUP_SUFFIX if args.by_setup else SUFFIX
    for model in models:
        result = calibrate(forecasts[forecasts["model"] == model], args.min_days, args.window, args.by_setup)
        result = result.drop(columns="setup", errors="ignore")
        target = store.root / "forecasts" / (model + suffix)
        # Derived files, rebuilt whole: a day no longer corrected (other --min-days) must not linger.
        for stale in target.glob("*.parquet"):
            stale.unlink()
        for day, frame in result.groupby("day"):
            store.save_forecasts(model + suffix, date.fromisoformat(str(day)), frame)
        log.info("%s: %d days corrected into %s", model, result["day"].nunique(), model + suffix)


if __name__ == "__main__":
    main()

from datetime import time

import numpy as np
import pandas as pd

from ts_analysis.series import ET, minute_of
from ts_analysis.store import Store


def entry_minute(bars: pd.DataFrame, prev_close: float, min_gap: float) -> int | None:
    """The minute, from 4:00 ET, of the first bar whose high reaches the pick's threshold: the ticker
    was picked on the day's high, so before that minute the pick itself tells where the price goes."""
    hit = bars[bars["high"] >= prev_close * (1 + min_gap)]
    if hit.empty:
        return None
    return minute_of(hit["time"].iloc[0].tz_convert(ET).time())


def entries(store: Store) -> pd.DataFrame:
    """The entry minute of every picked ticker-day with stored bars; one that never reaches its
    threshold in the stored hours is left out."""
    rows = []
    for day in store.days("picks"):
        picks = store.load_picks(day)
        prev = {c["ticker"]: c["prev_close"] for c in picks["candidates"]}
        for ticker in picks["picked"]:
            if not store.has_bars(day, ticker):
                continue
            minute = entry_minute(store.load_bars(day, ticker), prev[ticker], picks["rule"]["min_gap"])
            if minute is not None:
                rows.append({"day": day.isoformat(), "ticker": ticker, "entry": minute})
    return pd.DataFrame(rows, columns=["day", "ticker", "entry"])


def after_entry(forecasts: pd.DataFrame, entry: pd.DataFrame) -> pd.DataFrame:
    """The cuts made once the ticker's entry bar has closed: a cut at minute m sees the bars before m,
    so the bar of the entry minute only from the next one."""
    minutes = {(str(d), t): m for d, t, m in entry[["day", "ticker", "entry"]].itertuples(index=False)}
    reached = np.array([minutes.get((str(d), t), np.inf) for d, t in zip(forecasts["day"], forecasts["ticker"])])
    clock = {c: minute_of(time.fromisoformat(c)) for c in forecasts["cut"].unique()}
    cut = forecasts["cut"].map(clock).to_numpy()
    return forecasts[cut > reached]

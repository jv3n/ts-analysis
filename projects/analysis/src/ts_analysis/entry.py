from datetime import date, time

import numpy as np
import pandas as pd

from ts_analysis.selection import ScannerRule
from ts_analysis.series import CLOSE, ET, OPEN, minute_of
from ts_analysis.store import Store


def entry_minute(bars: pd.DataFrame, prev_close: float, min_gap: float) -> int | None:
    """The minute, from 4:00 ET, of the first bar whose high reaches the pick's threshold: the ticker
    was picked on the day's high, so before that minute the pick itself tells where the price goes."""
    hit = bars[bars["high"] >= prev_close * (1 + min_gap)]
    if hit.empty:
        return None
    return minute_of(hit["time"].iloc[0].tz_convert(ET).time())


def _minutes(bars: pd.DataFrame) -> pd.Series:
    return bars["time"].dt.tz_convert(ET).map(lambda t: minute_of(t.time()))


def is_gus(bars: pd.DataFrame, prev_close: float, rule: ScannerRule = ScannerRule()) -> bool:
    """`ts-scanner`'s GUS, read off the premarket and the session's first bar only, so it is known
    once that bar has printed."""
    minutes = _minutes(bars)
    premarket, session = bars[minutes < OPEN], bars[(minutes >= OPEN) & (minutes < CLOSE)]
    if premarket.empty or session.empty:
        return False
    pm_open, pm_high, open_ = premarket["open"].iloc[0], premarket["high"].max(), session["open"].iloc[0]
    return (
        rule.min_open <= open_ <= rule.max_open
        and pm_high >= prev_close * (1 + rule.pm_gap)
        and pm_high >= pm_open * (1 + rule.pm_push)
        and open_ >= prev_close * (1 + rule.open_gap)
        and open_ >= pm_high * (1 + rule.pm_hold)
    )


def scanner_entry(bars: pd.DataFrame, prev_close: float, rule: ScannerRule = ScannerRule()) -> tuple[str, int] | None:
    """A GUS enters on the session's first bar. Any other name enters on the first session bar where
    its running high reaches `push_gap` over the previous close, or its running range `push_range`:
    the minute a live scanner would list it. None if neither happens in the stored session."""
    minutes = _minutes(bars)
    session = bars[(minutes >= OPEN) & (minutes < CLOSE)]
    if session.empty:
        return None
    if is_gus(bars, prev_close, rule):
        return "gus", int(minutes[session.index[0]])
    high, low = session["high"].cummax(), session["low"].cummin()
    reached = (high >= prev_close * (1 + rule.push_gap)) | ((low > 0) & (high >= low * (1 + rule.push_range)))
    if not reached.any():
        return None
    return "push", int(minutes[reached.idxmax()])


def day_entries(store: Store, day: date) -> dict[str, tuple[str, int]]:
    """The setup and entry minute of each of the day's picks with stored bars; one that never enters in
    the stored hours is left out. Days picked on the day's high (`gap`) enter at their threshold."""
    picks = store.load_picks(day)
    if picks is None:
        return {}
    prev = {c["ticker"]: c["prev_close"] for c in picks["candidates"]}
    scanner = picks["rule"].get("name") == "scanner"
    found = {}
    for ticker in picks["picked"]:
        if not store.has_bars(day, ticker):
            continue
        bars = store.load_bars(day, ticker)
        if scanner:
            entry = scanner_entry(bars, prev[ticker], ScannerRule(**picks["rule"]))
        else:
            minute = entry_minute(bars, prev[ticker], picks["rule"]["min_gap"])
            entry = None if minute is None else ("gap", minute)
        if entry is not None:
            found[ticker] = entry
    return found


def entries(store: Store) -> pd.DataFrame:
    rows = [
        {"day": day.isoformat(), "ticker": ticker, "setup": setup, "entry": minute}
        for day in store.days("picks")
        for ticker, (setup, minute) in day_entries(store, day).items()
    ]
    return pd.DataFrame(rows, columns=["day", "ticker", "setup", "entry"])


def after_entry(forecasts: pd.DataFrame, entry: pd.DataFrame) -> pd.DataFrame:
    """The cuts made once the ticker's entry bar has closed: a cut at minute m sees the bars before m,
    so the bar of the entry minute only from the next one."""
    minutes = {(str(d), t): m for d, t, m in entry[["day", "ticker", "entry"]].itertuples(index=False)}
    reached = np.array([minutes.get((str(d), t), np.inf) for d, t in zip(forecasts["day"], forecasts["ticker"])])
    clock = {c: minute_of(time.fromisoformat(c)) for c in forecasts["cut"].unique()}
    cut = forecasts["cut"].map(clock).to_numpy()
    return forecasts[cut > reached]


def with_setup(forecasts: pd.DataFrame, entry: pd.DataFrame) -> pd.DataFrame:
    """Each forecast with the setup its ticker entered on, missing for a ticker that never entered."""
    setups = {(str(d), t): s for d, t, s in entry[["day", "ticker", "setup"]].itertuples(index=False)}
    return forecasts.assign(setup=[setups.get((str(d), t)) for d, t in zip(forecasts["day"], forecasts["ticker"])])

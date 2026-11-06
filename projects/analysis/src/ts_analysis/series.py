from dataclasses import dataclass
from functools import reduce
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
PREMARKET = time(4, 0)


def minute_of(clock: time) -> int:
    return (clock.hour - PREMARKET.hour) * 60 + clock.minute - PREMARKET.minute


def clock_of(minute: int) -> str:
    hours, minutes = divmod(minute + PREMARKET.hour * 60 + PREMARKET.minute, 60)
    return f"{hours:02d}:{minutes:02d}"


OPEN = minute_of(time(9, 30))
CLOSE = minute_of(time(16, 0))


@dataclass(frozen=True)
class Grid:
    """One slot per minute from 4:00 ET up to the close. Slot i holds the bar that starts at minute i,
    so its close is only known at minute i + 1: a cut at minute m sees slots [0, m). The sessions
    before the day, if any, come first: `past` slots, so minute m of the day sits at slot past + m."""

    log_close: np.ndarray
    volume: np.ndarray
    past: int = 0

    def seen_at(self, cut: int) -> "Grid":
        return Grid(self.log_close[: self.past + cut], self.volume[: self.past + cut], self.past)

    def price_at(self, minute: int) -> float:
        return float(np.exp(self.log_close[self.past + minute - 1]))

    def traded_by(self, cut: int) -> bool:
        return bool(np.any(self.volume[self.past : self.past + cut] > 0))


def session_index(day: date) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(datetime.combine(day, PREMARKET), tz=ET), periods=CLOSE, freq="min")


def minute_grid(bars: pd.DataFrame, day: date, past: pd.DataFrame | None = None, past_days: int = 0) -> Grid:
    """Minutes with no trade carry the last close forward with volume 0, overnight included; before
    the first trade the price is NaN, which the model reads as missing. After-hours bars are dropped.
    The past sessions are the last `past_days` the name traded, each cut at its close like the day;
    a name with fewer gets the missing ones in front as NaN, so every grid of a day has one length."""
    sessions = []
    if past_days and past is not None and not past.empty:
        traded = past["time"].dt.tz_convert(ET).dt.date
        past = past[traded < day]
        sessions = sorted(set(traded[traded < day]))[-past_days:]
    index = reduce(pd.DatetimeIndex.append, [session_index(session) for session in [*sessions, day]])
    frames = pd.concat([past, bars]) if sessions else bars
    frame = frames.set_index(frames["time"].dt.tz_convert(ET)).reindex(index)
    padding = (past_days - len(sessions)) * CLOSE
    return Grid(
        np.concatenate([np.full(padding, np.nan), np.log(frame["close"]).ffill().to_numpy(dtype="float64")]),
        np.concatenate([np.zeros(padding), frame["volume"].fillna(0).to_numpy(dtype="float64")]),
        past_days * CLOSE,
    )

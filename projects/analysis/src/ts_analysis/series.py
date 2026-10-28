from dataclasses import dataclass
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
    so its close is only known at minute i + 1: a cut at minute m sees slots [0, m)."""

    log_close: np.ndarray
    volume: np.ndarray

    def seen_at(self, cut: int) -> "Grid":
        return Grid(self.log_close[:cut], self.volume[:cut])

    def price_at(self, minute: int) -> float:
        return float(np.exp(self.log_close[minute - 1]))


def minute_grid(bars: pd.DataFrame, day: date) -> Grid:
    """Minutes with no trade carry the last close forward with volume 0; before the first trade the
    price is NaN, which the model reads as missing. After-hours bars are dropped."""
    index = pd.date_range(pd.Timestamp(datetime.combine(day, PREMARKET), tz=ET), periods=CLOSE, freq="min")
    frame = bars.set_index(bars["time"].dt.tz_convert(ET)).reindex(index)
    return Grid(
        np.log(frame["close"]).ffill().to_numpy(dtype="float64"),
        frame["volume"].fillna(0).to_numpy(dtype="float64"),
    )

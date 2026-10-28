from datetime import date, datetime, time

import numpy as np
import pytest

from ts_analysis.series import CLOSE, ET, OPEN, clock_of, minute_grid, minute_of
from ts_analysis.store import bars_frame

MON = date(2026, 10, 26)


def bar(clock: str, close: float, volume: float) -> dict:
    hours, minutes = map(int, clock.split(":"))
    start = datetime.combine(MON, time(hours, minutes), tzinfo=ET)
    return {"t": int(start.timestamp() * 1000), "o": close, "h": close, "l": close, "c": close, "v": volume, "vw": close, "n": 3}


FFIE = bars_frame(
    [
        bar("07:12", 2.05, 1200),
        bar("09:30", 2.40, 85000),
        bar("09:33", 2.62, 41000),
        bar("16:05", 2.31, 900),
    ]
)


def test_the_open_and_the_close_sit_at_their_clock_time():
    assert (clock_of(OPEN), clock_of(CLOSE)) == ("09:30", "16:00")


def test_the_grid_has_one_slot_per_minute_from_4am_to_the_close():
    assert len(minute_grid(FFIE, MON).log_close) == CLOSE


def test_a_minute_without_a_trade_carries_the_last_close_forward_with_no_volume():
    grid = minute_grid(FFIE, MON)
    slot = minute_of(time(9, 32))
    assert (np.exp(grid.log_close[slot]), grid.volume[slot]) == (pytest.approx(2.40), 0)


def test_the_price_is_missing_before_the_first_trade():
    assert np.isnan(minute_grid(FFIE, MON).log_close[minute_of(time(7, 11))])


def test_after_hours_bars_are_left_out():
    grid = minute_grid(FFIE, MON)
    assert np.exp(grid.log_close[-1]) == pytest.approx(2.62)


def test_a_cut_at_9_33_sees_the_9_32_bar_but_not_the_9_33_one():
    seen = minute_grid(FFIE, MON).seen_at(minute_of(time(9, 33)))
    assert (len(seen.log_close), np.exp(seen.log_close[-1])) == (minute_of(time(9, 33)), pytest.approx(2.40))


def test_the_price_at_a_minute_is_the_close_of_the_bar_just_before_it():
    assert minute_grid(FFIE, MON).price_at(minute_of(time(9, 34))) == pytest.approx(2.62)

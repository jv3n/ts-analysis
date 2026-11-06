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


THU, FRI = date(2026, 10, 22), date(2026, 10, 23)


def bar_on(day: date, clock: str, close: float, volume: float) -> dict:
    start = datetime.combine(day, time.fromisoformat(clock), tzinfo=ET)
    return {"t": int(start.timestamp() * 1000), "o": close, "h": close, "l": close, "c": close, "v": volume, "vw": close, "n": 3}


FFIE_PAST = bars_frame(
    [
        bar_on(THU, "10:15", 1.79, 22000),
        bar_on(THU, "15:59", 1.81, 64000),
        bar_on(FRI, "11:02", 1.84, 15000),
        bar_on(FRI, "15:59", 1.85, 48000),
        bar_on(FRI, "17:30", 1.90, 2000),
    ]
)


def test_the_past_sessions_come_before_the_day_one_full_session_each():
    grid = minute_grid(FFIE, MON, FFIE_PAST, past_days=2)
    assert (grid.past, len(grid.log_close)) == (2 * CLOSE, 3 * CLOSE)


def test_a_cut_on_the_day_sees_the_previous_sessions_in_full_and_no_bar_of_the_day_after_the_cut():
    seen = minute_grid(FFIE, MON, FFIE_PAST, past_days=2).seen_at(minute_of(time(9, 33)))
    assert len(seen.log_close) == 2 * CLOSE + minute_of(time(9, 33))
    assert seen.volume.sum() == 22000 + 64000 + 15000 + 48000 + 1200 + 85000


def test_the_previous_close_carries_over_the_night_until_the_first_trade_of_the_day():
    grid = minute_grid(FFIE, MON, FFIE_PAST, past_days=2)
    assert np.exp(grid.log_close[grid.past + minute_of(time(7, 11))]) == pytest.approx(1.85)


def test_a_name_with_fewer_past_sessions_than_asked_gets_them_missing_in_front():
    grid = minute_grid(FFIE, MON, FFIE_PAST, past_days=3)
    assert len(grid.log_close) == 4 * CLOSE
    assert np.isnan(grid.log_close[:CLOSE]).all() and not grid.volume[:CLOSE].any()


def test_with_past_sessions_minutes_of_the_day_keep_their_meaning():
    with_past = minute_grid(FFIE, MON, FFIE_PAST, past_days=2)
    assert with_past.price_at(minute_of(time(9, 34))) == minute_grid(FFIE, MON).price_at(minute_of(time(9, 34)))


def test_a_name_has_not_traded_by_a_cut_before_its_first_trade_of_the_day_even_with_a_past():
    grid = minute_grid(FFIE, MON, FFIE_PAST, past_days=2)
    assert (grid.traded_by(minute_of(time(7, 12))), grid.traded_by(minute_of(time(7, 13)))) == (False, True)

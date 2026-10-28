from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from ts_analysis.replay import QUANTILES, Flat, RandomWalk, replay, replay_day
from ts_analysis.series import CLOSE, OPEN, Grid, minute_of
from ts_analysis.store import Store

MON = date(2026, 10, 26)


def gapper(open_price: float, drift: float, seed: int) -> Grid:
    """A name that trades from 7:00, opens at `open_price` and drifts a little every minute."""
    rng = np.random.default_rng(seed)
    log_close = np.full(CLOSE, np.nan)
    first = minute_of(time(7, 0))
    steps = drift + 0.01 * rng.standard_normal(CLOSE - first)
    log_close[first:] = np.log(open_price) + np.cumsum(steps) - np.sum(steps[: OPEN - first])
    return Grid(log_close, np.where(np.isnan(log_close), 0.0, 5000.0))


class Spy:
    name = "spy"

    def __init__(self):
        self.seen: list[np.ndarray] = []

    def forecast(self, log_close: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
        self.seen.append(log_close.copy())
        return Flat().forecast(log_close, volume, length)


def test_a_forecast_made_at_minute_t_sees_no_bar_after_t():
    ffie = gapper(2.40, -0.001, seed=1)
    rewritten = minute_of(time(11, 0))
    future = ffie.log_close.copy()
    future[rewritten:] += np.log(3)  # a 3x squeeze after 11:00
    honest, peeking = Spy(), Spy()

    replay_day(MON, {"FFIE": ffie}, [honest], step=5)
    replay_day(MON, {"FFIE": Grid(future, ffie.volume)}, [peeking], step=5)

    for cut, (before, after) in zip(range(OPEN, rewritten + 1, 5), zip(honest.seen, peeking.seen)):
        assert before.shape[1] == cut
        np.testing.assert_array_equal(before, after)


def test_the_flat_baseline_predicts_no_move():
    rows = replay_day(MON, {"FFIE": gapper(2.40, 0.0, seed=2)}, [Flat()], step=30)
    assert {row[f"q{round(q * 100):02d}"] for row in rows for q in QUANTILES} == {0.0}


def test_the_random_walk_band_widens_with_the_horizon_around_no_move():
    rows = replay_day(MON, {"FFIE": gapper(2.40, 0.0, seed=3)}, [RandomWalk()], step=390)
    by_horizon = {row["horizon"]: row for row in rows}
    assert by_horizon[5]["q50"] == pytest.approx(0.0)
    assert by_horizon[30]["q90"] - by_horizon[30]["q10"] > by_horizon[5]["q90"] - by_horizon[5]["q10"] > 0


def test_each_cut_forecasts_5_15_30_minutes_ahead_and_to_the_close():
    rows = replay_day(MON, {"FFIE": gapper(2.40, 0.0, seed=4)}, [Flat()], step=390)
    assert [(row["horizon"], row["to_close"]) for row in rows] == [(5, False), (15, False), (30, False), (390, True)]


def test_ten_minutes_before_the_close_only_the_5_minute_and_close_horizons_remain():
    rows = replay_day(MON, {"FFIE": gapper(2.40, 0.0, seed=5)}, [Flat()], step=5)
    last_cut = [row for row in rows if row["cut"] == "15:50"]
    assert [row["horizon"] for row in last_cut] == [5, 10]


def test_a_forecast_is_scored_against_the_price_at_the_end_of_its_horizon():
    ffie = gapper(2.40, -0.002, seed=6)
    row = next(r for r in replay_day(MON, {"FFIE": ffie}, [Flat()], step=390) if r["horizon"] == 15)
    assert (row["last"], row["actual"]) == (ffie.price_at(OPEN), ffie.price_at(OPEN + 15))


def test_a_ticker_that_has_not_traded_yet_is_not_forecast():
    halted = Grid(np.full(CLOSE, np.nan), np.zeros(CLOSE))
    rows = replay_day(MON, {"FFIE": gapper(2.40, 0.0, seed=7), "MULN": halted}, [Flat()], step=390)
    assert {row["ticker"] for row in rows} == {"FFIE"}


def test_a_replayed_day_is_not_replayed_again(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.save_picks(MON, {"picked": ["FFIE"]})
    store.save_forecasts("flat", MON, pd.DataFrame([{"model": "flat"}]))
    monkeypatch.setattr("ts_analysis.replay.day_grids", lambda *_: pytest.fail("bars were read again"))

    replay(store, MON, MON, [Flat()], step=5)

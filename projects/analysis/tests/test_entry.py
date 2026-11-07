from datetime import date

import pandas as pd

from ts_analysis.entry import after_entry, entries, entry_minute, is_gus, scanner_entry, with_setup
from ts_analysis.score import main
from ts_analysis.store import Store

DAY = "2026-10-27"


def bars(*minutes: tuple) -> pd.DataFrame:
    """1-min bars given as (ET clock, high) — open, low and close at the high — or (clock, open, high, low)."""
    rows = [(clock, *prices) if len(prices) == 3 else (clock, prices[0], prices[0], prices[0]) for clock, *prices in minutes]
    times = [pd.Timestamp(f"{DAY} {clock}", tz="America/New_York").tz_convert("UTC") for clock, *_ in rows]
    opens, highs, lows = ([row[i] for row in rows] for i in (1, 2, 3))
    return pd.DataFrame({"time": times, "open": opens, "high": highs, "low": lows, "close": highs, "volume": 1000.0})


def forecast(cut: str, ticker: str = "AIOK") -> dict:
    return {"model": "random-walk", "ticker": ticker, "day": DAY, "cut": cut, "horizon": 15, "to_close": False,
            "last": 12.0, "actual": 12.6, **{f"q{q:02d}": 0.0 for q in (1, 5, *range(10, 95, 5), 95, 99)}}


def test_the_entry_is_the_first_minute_whose_high_reaches_the_gap_threshold():
    # AIOK closed at $9.16: +40 % is $12.82, first touched at 10:12.
    frame = bars(("09:30", 11.40), ("10:11", 12.70), ("10:12", 12.95), ("10:40", 17.72))
    assert entry_minute(frame, prev_close=9.16, min_gap=0.40) == (10 - 4) * 60 + 12


def test_a_ticker_day_that_never_reaches_its_threshold_has_no_entry():
    assert entry_minute(bars(("09:30", 11.40), ("15:59", 12.10)), prev_close=9.16, min_gap=0.40) is None


def test_a_cut_at_the_entry_minute_is_dropped_since_its_bar_has_not_closed_and_the_next_one_kept():
    entry = pd.DataFrame([{"day": DAY, "ticker": "AIOK", "entry": (10 - 4) * 60 + 12}])
    kept = after_entry(pd.DataFrame([forecast("10:10"), forecast("10:12"), forecast("10:13")]), entry)
    assert list(kept["cut"]) == ["10:13"]


def test_a_ticker_day_without_an_entry_is_left_out():
    entry = pd.DataFrame([{"day": DAY, "ticker": "AIOK", "entry": 300}])
    kept = after_entry(pd.DataFrame([forecast("11:00"), forecast("11:00", ticker="AEYE")]), entry)
    assert list(kept["ticker"]) == ["AIOK"]


def test_each_forecast_carries_the_setup_its_ticker_entered_on():
    entry = pd.DataFrame([{"day": DAY, "ticker": "AIOK", "setup": "gus", "entry": 330},
                          {"day": DAY, "ticker": "AEYE", "setup": "push", "entry": 372}])
    frame = pd.DataFrame([forecast("10:00"), forecast("11:00", ticker="AEYE"), forecast("11:00", ticker="MULN")])
    setup = with_setup(frame, entry)["setup"]
    assert list(setup[:2]) == ["gus", "push"] and pd.isna(setup.iloc[2])


def stored(tmp_path) -> Store:
    store = Store(tmp_path)
    store.save_picks(
        date.fromisoformat(DAY),
        {"day": DAY, "rule": {"min_gap": 0.4}, "candidates": [{"ticker": "AIOK", "prev_close": 9.16, "high": 17.72, "gap": 0.93}],
         "picked": ["AIOK"]},
    )
    path = store.bars_path(date.fromisoformat(DAY), "AIOK")
    path.parent.mkdir(parents=True)
    bars(("09:30", 11.40), ("10:12", 12.95)).to_parquet(path, index=False)
    return store


def test_the_entries_are_read_from_the_picks_and_the_stored_bars(tmp_path):
    assert entries(stored(tmp_path)).to_dict("records") == [{"day": DAY, "ticker": "AIOK", "setup": "gap", "entry": 372}]


def test_the_score_keeps_the_cuts_after_entry_unless_asked_for_every_cut(tmp_path, capsys):
    store = stored(tmp_path)
    folder = store.root / "forecasts" / "random-walk"
    folder.mkdir(parents=True)
    pd.DataFrame([forecast("09:45"), forecast("10:30")]).to_parquet(folder / f"{DAY}.parquet", index=False)
    main(["--data", str(tmp_path), "--resamples", "10"])
    assert "1 forecasts, on the cuts after the ticker reached its gap, 1 of 2" in capsys.readouterr().out
    main(["--data", str(tmp_path), "--resamples", "10", "--all-cuts"])
    assert "2 forecasts, on every cut" in capsys.readouterr().out


# BNAI closed at $2.00: a premarket push from $2.10 to $3.10 (+55 %), an open at $2.60 (+30 %, 16 % off the top).
GUS_DAY = [("04:12", 2.10, 2.15, 2.08), ("07:45", 2.40, 3.10, 2.35), ("09:12", 2.70, 2.75, 2.55), ("09:30", 2.60, 2.85, 2.50)]


def test_a_premarket_push_still_alive_at_the_open_is_a_gus_entering_on_the_first_session_bar():
    assert scanner_entry(bars(*GUS_DAY, ("10:40", 2.10, 2.20, 1.90)), prev_close=2.00) == ("gus", (9 - 4) * 60 + 30)


def test_a_gus_reads_no_bar_after_the_session_s_first_one():
    calm, wild = bars(*GUS_DAY, ("11:00", 2.40, 2.45, 2.30)), bars(*GUS_DAY, ("11:00", 2.40, 9.00, 0.40))
    assert is_gus(calm, prev_close=2.00) and is_gus(wild, prev_close=2.00)


def test_a_premarket_that_gave_everything_back_before_the_bell_is_no_gus():
    # Open at $1.30: 58 % off the $3.10 top, and down on the previous close.
    faded = [*GUS_DAY[:3], ("09:30", 1.30, 1.35, 1.25)]
    assert not is_gus(bars(*faded), prev_close=2.00)


def test_a_session_push_enters_on_the_first_bar_whose_running_high_reaches_25_percent():
    # AIOK closed at $9.16: +25 % is $11.45, first touched at 10:12.
    day = bars(("09:30", 9.30, 9.40, 9.20), ("10:11", 11.00, 11.40, 10.90), ("10:12", 11.40, 11.60, 11.30))
    assert scanner_entry(day, prev_close=9.16) == ("push", (10 - 4) * 60 + 12)


def test_a_40_percent_session_range_enters_even_below_the_previous_close():
    # CLRO closed at $5.00, topped at $4.85 at the open, then sank: $3.40 at 11:00 completes a 40 % range.
    day = bars(("09:30", 4.80, 4.85, 4.70), ("10:15", 4.10, 4.15, 3.90), ("11:00", 3.50, 3.55, 3.40))
    assert scanner_entry(day, prev_close=5.00) == ("push", (11 - 4) * 60)


def test_a_name_that_never_pushes_in_the_session_does_not_enter():
    day = bars(("09:30", 9.30, 9.40, 9.20), ("15:59", 10.10, 10.20, 10.00))
    assert scanner_entry(day, prev_close=9.16) is None

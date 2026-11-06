from datetime import date

import pandas as pd

from ts_analysis.entry import after_entry, entries, entry_minute
from ts_analysis.score import main
from ts_analysis.store import Store

DAY = "2026-10-27"


def bars(*minutes: tuple[str, float]) -> pd.DataFrame:
    """AIOK's 1-min bars, given as (ET clock, high)."""
    times = [pd.Timestamp(f"{DAY} {clock}", tz="America/New_York").tz_convert("UTC") for clock, _ in minutes]
    highs = [high for _, high in minutes]
    return pd.DataFrame({"time": times, "open": highs, "high": highs, "low": highs, "close": highs, "volume": 1000.0})


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
    assert entries(stored(tmp_path)).to_dict("records") == [{"day": DAY, "ticker": "AIOK", "entry": 372}]


def test_the_score_keeps_the_cuts_after_entry_unless_asked_for_every_cut(tmp_path, capsys):
    store = stored(tmp_path)
    folder = store.root / "forecasts" / "random-walk"
    folder.mkdir(parents=True)
    pd.DataFrame([forecast("09:45"), forecast("10:30")]).to_parquet(folder / f"{DAY}.parquet", index=False)
    main(["--data", str(tmp_path), "--resamples", "10"])
    assert "1 forecasts, on the cuts after the ticker reached its gap, 1 of 2" in capsys.readouterr().out
    main(["--data", str(tmp_path), "--resamples", "10", "--all-cuts"])
    assert "2 forecasts, on every cut" in capsys.readouterr().out

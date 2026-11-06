import json
from datetime import date, datetime, time, timedelta

import pytest


from ts_analysis.fetch import (
    PAST_LOOKBACK,
    PicksMismatch,
    backfill,
    fetch_day,
    main,
    past_sessions,
    planned_requests,
    previous_session,
    weekdays,
)
from ts_analysis.massive import DayNotPublished
from ts_analysis.selection import GapRule
from ts_analysis.series import ET
from ts_analysis.store import Store, bars_frame

THU = date(2026, 10, 22)
FRI = date(2026, 10, 23)
MON = date(2026, 10, 26)
TUE = date(2026, 10, 27)
GROUPED = {
    THU: [
        {"T": "FFIE", "h": 1.88, "c": 1.81},
        {"T": "SNDL", "h": 2.12, "c": 2.08},
        {"T": "MULN", "h": 0.45, "c": 0.43},
    ],
    FRI: [
        {"T": "FFIE", "h": 1.90, "c": 1.85},
        {"T": "BBLGW", "h": 0.33, "c": 0.31},
        {"T": "SNDL", "h": 2.15, "c": 2.10},
        {"T": "MULN", "h": 0.44, "c": 0.42},
    ],
    MON: [
        {"T": "FFIE", "h": 2.95, "c": 2.40},
        {"T": "BBLGW", "h": 0.64, "c": 0.52},
        {"T": "SNDL", "h": 2.20, "c": 2.12},
        {"T": "MULN", "h": 4.60, "c": 4.31},
    ],
}
COMMON = ["FFIE", "MULN", "SNDL"]
SPLITS = {MON: [{"ticker": "MULN", "execution_date": "2026-10-26", "split_from": 10, "split_to": 1}]}
BAR = {"t": 1793021400000, "o": 2.10, "h": 2.30, "l": 2.05, "c": 2.25, "v": 184200, "vw": 2.19, "n": 512}


class FakeSource:
    def __init__(self, unpublished: frozenset[date] = frozenset()):
        self.calls: list[tuple] = []
        self.unpublished = unpublished

    def grouped_daily(self, day):
        self.calls.append(("grouped", day))
        if day in self.unpublished:
            raise DayNotPublished(day.isoformat())
        return GROUPED.get(day, [])

    def minute_bars(self, ticker, start, end, adjusted=False):
        self.calls.append(("bars", ticker, start, end, adjusted))
        return [BAR]

    def splits(self, day):
        self.calls.append(("splits", day))
        return SPLITS.get(day, [])

    def common_stocks(self, day):
        self.calls.append(("common", day))
        return COMMON


def fetch_monday(store: Store, source: FakeSource | None = None, **settings) -> dict | None:
    settings = {"rule": GapRule(), "per_day": 10, "seed": "s", **settings}
    return fetch_day(source or FakeSource(), store, MON, **settings)


def test_weekdays_skip_the_weekend():
    assert list(weekdays(FRI, MON)) == [FRI, MON]


def test_a_monday_is_compared_with_the_previous_friday_close(tmp_path):
    picks = fetch_monday(Store(tmp_path))
    assert picks["candidates"] == [{"ticker": "FFIE", "prev_close": 1.85, "high": 2.95, "gap": 0.5946}]


def test_a_warrant_gapping_more_than_40_percent_is_not_a_candidate(tmp_path):
    assert [c["ticker"] for c in fetch_monday(Store(tmp_path))["candidates"]] == ["FFIE"]


def test_with_every_type_allowed_the_warrant_is_a_candidate_and_no_ticker_list_is_fetched(tmp_path):
    source = FakeSource()
    picks = fetch_monday(Store(tmp_path), source, rule=GapRule(common_only=False))
    assert [c["ticker"] for c in picks["candidates"]] == ["BBLGW", "FFIE"]
    assert ("common", MON) not in source.calls


def test_a_ticker_whose_reverse_split_executes_that_day_is_not_picked(tmp_path):
    assert "MULN" not in fetch_monday(Store(tmp_path))["picked"]


def test_the_picks_record_the_settings_that_produced_them(tmp_path):
    picks = fetch_monday(Store(tmp_path), seed="ts-analysis")
    assert picks["rule"] == {"min_gap": 0.40, "min_prev_close": 0.30, "max_prev_close": 10.0, "common_only": True}
    assert (picks["per_day"], picks["seed"]) == (10, "ts-analysis")


def test_no_session_in_the_lookback_window_is_an_error(tmp_path):
    with pytest.raises(LookupError):
        previous_session(FakeSource(), Store(tmp_path), date(2026, 1, 5), max_lookback=3)


def test_the_picked_tickers_bars_are_stored_locally(tmp_path):
    store = Store(tmp_path)
    fetch_monday(store)

    bars = store.load_bars(MON, "FFIE")
    assert list(bars.columns) == ["time", "open", "high", "low", "close", "volume", "vwap", "trades"]
    assert str(bars["time"].iloc[0]) == "2026-10-26 13:30:00+00:00"


def test_a_ticker_with_no_bars_is_stored_empty_with_the_same_columns_and_types(tmp_path):
    store = Store(tmp_path)
    store.save_bars(MON, "FFIE", [])
    store.save_bars(MON, "SNDL", [BAR])

    empty = store.load_bars(MON, "FFIE")
    assert empty.empty
    assert empty.dtypes.equals(store.load_bars(MON, "SNDL").dtypes)


def test_a_bars_file_left_half_written_does_not_count_as_stored(tmp_path):
    store = Store(tmp_path)
    tmp = store.bars_path(MON, "FFIE").with_suffix(".tmp")
    tmp.parent.mkdir(parents=True)
    tmp.write_bytes(b"PAR1 truncated")
    assert not store.has_bars(MON, "FFIE")


def test_a_rerun_makes_no_request_for_a_day_already_fetched(tmp_path):
    store = Store(tmp_path)
    fetch_monday(store)
    rerun = FakeSource()
    fetch_monday(store, rerun)
    assert rerun.calls == []


def test_a_rerun_resumes_the_bars_missing_from_an_interrupted_day(tmp_path):
    store = Store(tmp_path)
    fetch_monday(store)
    store.bars_path(MON, "FFIE").unlink()
    rerun = FakeSource()
    fetch_monday(store, rerun)
    assert rerun.calls == [("bars", "FFIE", MON, MON, False)]


def test_the_past_sessions_come_in_one_adjusted_request_over_the_days_before(tmp_path):
    source = FakeSource()
    fetch_monday(Store(tmp_path), source)
    assert ("bars", "FFIE", MON - timedelta(days=PAST_LOOKBACK), MON, True) in source.calls


def test_a_rerun_fetches_the_past_sessions_of_a_day_stored_before_they_existed(tmp_path):
    store = Store(tmp_path)
    fetch_monday(store)
    store.past_path(MON, "FFIE").unlink()
    rerun = FakeSource()
    fetch_monday(store, rerun)
    assert rerun.calls == [("bars", "FFIE", MON - timedelta(days=PAST_LOOKBACK), MON, True)]


def minute(day: date, clock: str, close: float, volume: float) -> dict:
    start = datetime.combine(day, time.fromisoformat(clock), tzinfo=ET)
    return {"t": int(start.timestamp() * 1000), "o": close, "h": close, "l": close, "c": close, "v": volume, "vw": close, "n": 9}


# MULN reverse-splits 1:10 on Friday, between Thursday and the Monday it gaps, and again 1:4 the
# Wednesday after, before the fetch. Raw, it closes $0.42 Thursday and trades $4.40 Monday.
RAW_MONDAY = bars_frame([minute(MON, "09:30", 4.40, 50_000), minute(MON, "09:31", 4.52, 30_000)])
ADJUSTED = bars_frame(
    [
        minute(THU, "15:59", 0.42 * 10 * 4, 125_000 / 40),
        minute(FRI, "15:59", 4.30 * 4, 9_000 / 4),
        minute(MON, "09:30", 4.40 * 4, 50_000 / 4),
        minute(MON, "09:31", 4.52 * 4, 30_000 / 4),
    ]
)


def test_a_split_inside_the_window_does_not_show_up_as_a_price_jump():
    past = past_sessions(ADJUSTED, RAW_MONDAY, MON, sessions=5)
    assert past["close"].tolist() == pytest.approx([4.20, 4.30])


def test_a_split_after_the_day_is_undone_so_the_past_meets_the_raw_day_prices_and_volumes():
    past = past_sessions(ADJUSTED, RAW_MONDAY, MON, sessions=5)
    assert (past["close"].iloc[-1], past["volume"].iloc[-1]) == (pytest.approx(4.30), pytest.approx(9_000))


def test_only_the_last_sessions_before_the_day_are_kept():
    past = past_sessions(ADJUSTED, RAW_MONDAY, MON, sessions=1)
    assert past["time"].dt.tz_convert(ET).dt.date.tolist() == [FRI]


def test_past_sessions_that_cannot_be_matched_to_the_raw_day_are_not_guessed():
    assert past_sessions(ADJUSTED.iloc[:2], RAW_MONDAY, MON, sessions=5) is None


def test_the_planned_requests_count_what_a_picked_day_misses_and_a_full_day_for_a_new_one(tmp_path):
    store = Store(tmp_path)
    fetch_monday(store)
    store.past_path(MON, "FFIE").unlink()
    assert planned_requests(store, [MON, TUE], per_day=10) == 1 + 8 + 2 * 10


def test_a_day_known_to_be_a_holiday_plans_no_request(tmp_path):
    store = Store(tmp_path)
    thanksgiving = date(2026, 11, 26)
    store.save_grouped(thanksgiving, [])
    assert planned_requests(store, [thanksgiving], per_day=10) == 0


def test_the_backfill_records_its_planned_requests_and_their_time_at_the_quota(tmp_path):
    store = Store(tmp_path)
    backfill(FakeSource(), store, TUE, TUE, GapRule(), per_day=10, seed="s")
    status = json.loads((tmp_path / "status" / "fetch.json").read_text())
    assert (status["requests"], status["estimated_s"]) == (28, 28 * 12)


def test_a_rerun_with_other_settings_fails_instead_of_silently_keeping_the_old_picks(tmp_path):
    store = Store(tmp_path)
    fetch_monday(store)
    with pytest.raises(PicksMismatch):
        fetch_monday(store, rule=GapRule(min_gap=0.30))


def test_a_market_holiday_picks_nothing(tmp_path):
    thanksgiving = date(2026, 11, 26)
    assert fetch_day(FakeSource(), Store(tmp_path), thanksgiving, GapRule(), 10, "s") is None


def test_consecutive_days_reuse_the_grouped_bars_already_fetched(tmp_path):
    source = FakeSource()
    backfill(source, Store(tmp_path), FRI, MON, GapRule(), per_day=10, seed="s")
    assert source.calls.count(("grouped", FRI)) == 1


def test_the_backfill_stops_at_the_first_unpublished_day_and_resumes_later(tmp_path):
    store = Store(tmp_path)
    blocked = FakeSource(unpublished=frozenset({TUE}))
    backfill(blocked, store, MON, TUE, GapRule(), per_day=10, seed="s")
    assert store.load_picks(MON) is not None and store.load_picks(TUE) is None

    later = FakeSource()
    backfill(later, store, MON, TUE, GapRule(), per_day=10, seed="s")
    assert ("grouped", MON) not in later.calls and ("grouped", TUE) in later.calls


def test_the_cli_refuses_a_start_after_the_end(capsys):
    with pytest.raises(SystemExit):
        main(["--start", "2026-10-27", "--end", "2026-10-01"])
    assert "after --end" in capsys.readouterr().err


def test_the_cli_refuses_to_run_without_an_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(["--start", "2026-10-26", "--end", "2026-10-26"])
    assert "MASSIVE_API_KEY" in capsys.readouterr().err

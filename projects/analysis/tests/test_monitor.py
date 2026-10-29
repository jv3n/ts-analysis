import json
from datetime import date
from threading import Thread
from urllib.request import urlopen

import pytest

from ts_analysis.monitor import fetch_summary, forecast_view, replay_summary, route, server
from ts_analysis.replay import Flat, RandomWalk, replay
from ts_analysis.store import Store

FRI = date(2026, 10, 23)
MON = date(2026, 10, 26)
TUE = date(2026, 10, 27)
WED = date(2026, 10, 28)
OPEN_MS = 1793021400000  # 2026-10-26 09:30 ET


def ffie_bars(minutes: int = 390) -> list[dict]:
    """FFIE fading from $2.40 at the open, one cent every ten minutes, half a cent up every other minute."""
    rows = []
    for i in range(minutes):
        close = round(2.40 - 0.001 * i + 0.005 * (i % 2), 4)
        rows.append({"t": OPEN_MS + i * 60_000, "o": close, "h": close, "l": close, "c": close, "v": 1200, "vw": close, "n": 9})
    return rows


@pytest.fixture
def store(tmp_path) -> Store:
    store = Store(tmp_path)
    store.save_grouped(MON, [{"T": "FFIE", "h": 2.95, "c": 2.40}])
    store.save_picks(
        MON,
        {
            "candidates": [{"ticker": "FFIE", "prev_close": 1.85, "high": 2.95, "gap": 0.5946}],
            "picked": ["FFIE", "SNDL"],
        },
    )
    store.save_bars(MON, "FFIE", ffie_bars())
    replay(store, MON, MON, [Flat(), RandomWalk()], step=30)
    return store


def test_the_fetch_summary_counts_the_bars_stored_per_ticker_and_flags_the_ones_still_to_fetch(store):
    [day] = fetch_summary(store)["days"]
    assert day["day"] == "2026-10-26"
    assert day["bars"] == {"FFIE": 390, "SNDL": None}


def test_a_weekday_without_picks_is_a_holiday_when_the_market_was_closed_and_missing_otherwise(store):
    store.save_picks(WED, {"candidates": [], "picked": []})
    store.save_grouped(TUE, [])
    summary = fetch_summary(store)
    assert (summary["holidays"], summary["missing"]) == (["2026-10-27"], [])

    store.root.joinpath("grouped", "2026-10-27.json").unlink()
    assert fetch_summary(store)["missing"] == ["2026-10-27"]


def test_the_previous_session_fetched_for_the_gap_does_not_count_as_a_missing_day(store):
    store.save_grouped(FRI, [{"T": "FFIE", "h": 1.90, "c": 1.85}])
    assert fetch_summary(store)["missing"] == []


def test_the_replay_summary_lists_each_model_s_days_with_their_forecast_count_and_tickers(store):
    models = {m["model"]: m["days"] for m in replay_summary(store)["models"]}
    assert set(models) == {"flat", "random-walk"}
    [day] = models["flat"]
    # 12 cuts from 10:00, FFIE having no trade before the open: +5, +15, +30 and the close each,
    # except 15:30 where +30 is the close.
    assert (day["day"], day["forecasts"], day["tickers"]) == ("2026-10-26", 12 * 4 - 1, ["FFIE"])
    assert day["ms"] >= 0


def test_the_forecast_view_gives_the_price_at_every_minute_of_the_session(store):
    view = forecast_view(store, "flat", MON, "FFIE")
    assert len(view["prices"]) == 391
    assert view["prices"][0] is None  # no premarket trade: nothing known at 9:30
    assert view["prices"][1] == pytest.approx(2.40)
    assert view["prices"][390] == pytest.approx(2.016)


def test_a_cut_s_band_is_drawn_in_prices_from_the_last_price_at_the_cut(store):
    view = forecast_view(store, "random-walk", MON, "FFIE")
    cut = next(c for c in view["cuts"] if c["cut"] == "10:00")
    assert (cut["at"], cut["last"]) == (30, pytest.approx(2.376))
    plus_15 = next(h for h in cut["horizons"] if h["horizon"] == 15)
    assert plus_15["at"] == 45
    assert plus_15["q10"] < plus_15["q50"] < plus_15["q90"]
    assert plus_15["q50"] == pytest.approx(2.376)


def test_a_ticker_not_forecast_that_day_is_not_found(store):
    code, payload = route(store, "/api/forecast", {"model": "flat", "day": "2026-10-26", "ticker": "SNDL"})
    assert code == 404 and "SNDL" in payload["error"]


@pytest.mark.parametrize(
    "params", [{"model": "flat", "ticker": "FFIE"}, {"model": "flat", "day": "26/10/2026", "ticker": "FFIE"}]
)
def test_a_forecast_request_without_a_valid_day_is_a_bad_request(store, params):
    assert route(store, "/api/forecast", params)[0] == 400


def test_an_unknown_endpoint_is_not_found(store):
    assert route(store, "/api/scores", {})[0] == 404


def test_the_status_endpoint_reports_no_job_when_none_ran(tmp_path):
    statuses = route(Store(tmp_path), "/api/status", {})[1]
    assert statuses == {"fetch": None, "replay": None}


def test_the_page_and_the_api_are_served_on_localhost(store, tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    static.joinpath("index.html").write_text("<title>ts-analysis monitor</title>")
    httpd = server(store, port=0, static=static)
    Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert httpd.server_address[0] == "127.0.0.1"
        assert "ts-analysis monitor" in urlopen(f"{base}/").read().decode()
        replayed = json.loads(urlopen(f"{base}/api/replay").read())
        assert [m["model"] for m in replayed["models"]] == ["flat", "random-walk"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_forecast_payload_holds_no_nan_so_the_browser_can_parse_it(store):
    json.dumps(forecast_view(store, "random-walk", MON, "FFIE"), allow_nan=False)

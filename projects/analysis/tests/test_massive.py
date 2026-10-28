from datetime import date

import httpx
import pytest

from ts_analysis.massive import BASE_URL, DayNotPublished, MassiveClient, RateLimiter, auth_headers

MON = date(2026, 10, 26)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_the_limiter_spaces_calls_12_seconds_apart_for_5_requests_a_minute():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=5, clock=clock.time, sleep=clock.sleep)
    for _ in range(3):
        limiter.wait()
    assert clock.slept == [12.0, 12.0]


def test_the_limiter_does_not_wait_when_enough_time_has_passed():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=5, clock=clock.time, sleep=clock.sleep)
    limiter.wait()
    clock.now += 30
    limiter.wait()
    assert clock.slept == []


def client_for(handler, clock: FakeClock, api_key: str = "test-key") -> MassiveClient:
    http = httpx.Client(
        base_url=BASE_URL, headers=auth_headers(api_key), transport=httpx.MockTransport(handler)
    )
    limiter = RateLimiter(clock=clock.time, sleep=clock.sleep)
    return MassiveClient(api_key, http=http, limiter=limiter, sleep=clock.sleep)


def responding(*responses: httpx.Response):
    queue = iter(responses)
    return lambda request: next(queue)


def test_minute_bars_asks_one_day_of_raw_1_minute_bars():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": [{"t": 1793021400000, "c": 2.41}]})

    bars = client_for(handler, FakeClock()).minute_bars("FFIE", MON)

    assert bars == [{"t": 1793021400000, "c": 2.41}]
    assert seen[0].url.path == "/v2/aggs/ticker/FFIE/range/1/minute/2026-10-26/2026-10-26"
    assert seen[0].url.params["adjusted"] == "false"
    assert seen[0].url.params["limit"] == "50000"


def test_the_api_key_goes_in_a_header_and_never_in_the_url():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    client_for(handler, FakeClock(), api_key="sk-massive-123").splits(MON)

    assert seen[0].headers["Authorization"] == "Bearer sk-massive-123"
    assert "sk-massive-123" not in str(seen[0].url)


def test_two_calls_go_through_the_rate_limiter():
    clock = FakeClock()
    client = client_for(lambda r: httpx.Response(200, json={"results": []}), clock)
    client.splits(MON)
    client.splits(MON)
    assert clock.slept == [12.0]


def test_a_holiday_returns_no_grouped_bars():
    client = client_for(responding(httpx.Response(200, json={"status": "OK", "resultsCount": 0})), FakeClock())
    assert client.grouped_daily(date(2026, 7, 3)) == []


def test_a_grouped_response_without_results_or_count_is_not_taken_for_a_holiday():
    client = client_for(responding(httpx.Response(200, json={"status": "DELAYED"})), FakeClock())
    with pytest.raises(RuntimeError):
        client.grouped_daily(MON)


def test_a_403_means_the_day_is_not_published_yet():
    client = client_for(responding(httpx.Response(403)), FakeClock())
    with pytest.raises(DayNotPublished):
        client.grouped_daily(date(2026, 10, 28))


def test_a_429_waits_for_retry_after_then_retries():
    clock = FakeClock()
    client = client_for(
        responding(httpx.Response(429, headers={"Retry-After": "20"}), httpx.Response(200, json={"results": []})),
        clock,
    )
    assert client.splits(MON) == []
    assert 20.0 in clock.slept


def test_a_429_without_retry_after_waits_a_minute():
    clock = FakeClock()
    client = client_for(responding(httpx.Response(429), httpx.Response(200, json={"results": []})), clock)
    client.splits(MON)
    assert 60 in clock.slept


def test_a_429_on_every_attempt_fails_without_a_last_useless_wait():
    clock = FakeClock()
    client = client_for(lambda r: httpx.Response(429), clock)
    with pytest.raises(httpx.HTTPStatusError):
        client.splits(MON)
    assert clock.slept.count(60) == 3


def test_a_server_error_is_retried():
    client = client_for(responding(httpx.Response(502), httpx.Response(200, json={"results": []})), FakeClock())
    assert client.splits(MON) == []


def test_a_network_error_is_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json={"results": []})

    assert client_for(handler, FakeClock()).splits(MON) == []


def test_the_common_stocks_of_a_day_follow_every_page():
    seen = []
    pages = iter(
        [
            {"results": [{"ticker": "FFIE"}, {"ticker": "AISP"}], "next_url": f"{BASE_URL}/v3/reference/tickers?cursor=p2"},
            {"results": [{"ticker": "SNDL"}]},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=next(pages))

    assert client_for(handler, FakeClock()).common_stocks(MON) == ["AISP", "FFIE", "SNDL"]
    assert (seen[0].params["type"], seen[0].params["date"]) == ("CS", "2026-10-26")
    assert seen[1].params["cursor"] == "p2"

import logging
import time
from collections.abc import Callable
from datetime import date

import httpx

BASE_URL = "https://api.massive.com"
REQUESTS_PER_MINUTE = 5

log = logging.getLogger("ts_analysis.massive")


class DayNotPublished(Exception):
    """Basic serves a day only after midnight ET; before that the request gets a 403."""


class RateLimiter:
    """Spaces calls evenly instead of bursting 5 then waiting for the window to roll."""

    def __init__(
        self,
        per_minute: int = REQUESTS_PER_MINUTE,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._interval = 60 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._next = float("-inf")

    def wait(self) -> float:
        now = self._clock()
        waited = max(0.0, self._next - now)
        if waited:
            self._sleep(waited)
            now = self._next
        self._next = now + self._interval
        return waited


class MassiveClient:
    def __init__(
        self,
        api_key: str,
        http: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 4,
    ):
        # Key in a header, not the apiKey query param, so it never shows up in a logged URL.
        self._http = http or httpx.Client(base_url=BASE_URL, headers=auth_headers(api_key), timeout=30)
        self._limiter = limiter or RateLimiter()
        self._sleep = sleep
        self._max_attempts = max_attempts

    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(1, self._max_attempts + 1):
            waited = self._limiter.wait()
            last = attempt == self._max_attempts
            try:
                response = self._http.get(path, params=params)
            except httpx.TransportError as error:
                if last:
                    raise
                log.warning("GET %s failed (%s), retry %d in %d s", path, type(error).__name__, attempt, 30 * attempt)
                self._sleep(30 * attempt)
                continue
            log.info("GET %s -> %d (waited %.0f s for the quota)", path, response.status_code, waited)
            if response.status_code == 403:
                raise DayNotPublished(path)
            if response.status_code == 429 or response.status_code >= 500:
                if last:
                    response.raise_for_status()
                backoff = retry_after(response, default=60 if response.status_code == 429 else 30 * attempt)
                log.warning("GET %s -> %d, retry %d in %.0f s", path, response.status_code, attempt, backoff)
                self._sleep(backoff)
                continue
            response.raise_for_status()
            return response.json()
        raise AssertionError("unreachable")

    # Raw prices: adjusted ones are adjusted as of the fetch, so a split between two fetches would
    # make a day's high and the previous close disagree. Split days are excluded instead.
    def grouped_daily(self, day: date) -> list[dict]:
        """Daily bars of every US stock; empty on a market holiday."""
        payload = self._get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{day.isoformat()}", {"adjusted": "false"}
        )
        if "results" not in payload and payload.get("resultsCount") != 0:
            raise RuntimeError(f"Unexpected grouped daily response for {day}: {payload.get('status')}")
        return payload.get("results") or []

    def minute_bars(self, ticker: str, day: date) -> list[dict]:
        payload = self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/minute/{day.isoformat()}/{day.isoformat()}",
            {"adjusted": "false", "sort": "asc", "limit": 50000},
        )
        return payload.get("results") or []

    def splits(self, day: date) -> list[dict]:
        payload = self._get("/stocks/v1/splits", {"execution_date": day.isoformat(), "limit": 5000})
        return payload.get("results") or []


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def retry_after(response: httpx.Response, default: float) -> float:
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return default

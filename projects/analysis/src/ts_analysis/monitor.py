import argparse
import json
import logging
import math
from datetime import date, time
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pyarrow.parquet as pq

from ts_analysis.fetch import weekdays
from ts_analysis.series import CLOSE, OPEN, minute_grid, minute_of
from ts_analysis.status import load_status
from ts_analysis.store import Store

log = logging.getLogger("ts_analysis.monitor")

STATIC = Path(__file__).resolve().parents[3] / "monitor"
BAND = ("q10", "q25", "q50", "q75", "q90")


class NotFound(Exception):
    pass


def _number(value: float) -> float | None:
    return None if value is None or math.isnan(value) else round(float(value), 6)


def fetch_summary(store: Store) -> dict:
    """The picked days with their candidates and the bars stored per ticker, `None` for a ticker still to
    fetch. A weekday between the first and the last picked day with no picks is a holiday when its grouped
    bars came back empty, missing otherwise."""
    picked_days = store.days("picks")
    days = []
    for day in picked_days:
        picks = store.load_picks(day)
        bars = {
            ticker: pq.read_metadata(store.bars_path(day, ticker)).num_rows if store.has_bars(day, ticker) else None
            for ticker in picks["picked"]
        }
        days.append({"day": day.isoformat(), "candidates": picks["candidates"], "picked": picks["picked"], "bars": bars})
    holidays, missing = [], []
    if picked_days:
        for day in weekdays(picked_days[0], picked_days[-1]):
            if day in picked_days:
                continue
            (holidays if store.load_grouped(day) == [] else missing).append(day.isoformat())
    return {"days": days, "holidays": holidays, "missing": missing}


def replay_summary(store: Store) -> dict:
    models = []
    for model in store.models():
        days = []
        for day in store.forecast_days(model):
            frame = pq.read_table(store.forecasts_path(model, day), columns=["ticker", "ms_per_forecast"]).to_pandas()
            days.append(
                {
                    "day": day.isoformat(),
                    "forecasts": len(frame),
                    "tickers": sorted(frame["ticker"].unique().tolist()),
                    "ms": _number(frame["ms_per_forecast"].mean()) if len(frame) else None,
                }
            )
        models.append({"model": model, "days": days})
    return {"models": models}


def forecast_view(store: Store, model: str, day: date, ticker: str) -> dict:
    """The price at every minute from the open to the close, and for each cut the quantile band it
    forecast, as prices. Minutes count from the open: `at` 0 is 9:30."""
    if not store.has_forecasts(model, day) or not store.has_bars(day, ticker):
        raise NotFound(f"no forecasts of {model} for {ticker} on {day}")
    grid = minute_grid(store.load_bars(day, ticker), day)
    prices = [_number(grid.price_at(minute)) for minute in range(OPEN, CLOSE + 1)]
    frame = store.load_forecasts(model, day)
    frame = frame[frame["ticker"] == ticker]
    if frame.empty:
        raise NotFound(f"{ticker} was not forecast by {model} on {day}")
    cuts = []
    for cut, rows in frame.groupby("cut", sort=True):
        at = minute_of(time.fromisoformat(cut)) - OPEN
        last = float(rows["last"].iloc[0])
        horizons = [
            {"horizon": int(row.horizon), "at": at + int(row.horizon)}
            | {q: _number(last * np.exp(getattr(row, q))) for q in BAND}
            for row in rows.sort_values("horizon").itertuples()
        ]
        cuts.append({"cut": cut, "at": at, "last": _number(last), "horizons": horizons})
    return {"model": model, "day": day.isoformat(), "ticker": ticker, "prices": prices, "cuts": cuts}


def statuses(store: Store) -> dict:
    return {job: load_status(store.root, job) for job in ("fetch", "replay")}


def route(store: Store, path: str, params: dict[str, str]) -> tuple[int, dict]:
    try:
        match path:
            case "/api/status":
                return HTTPStatus.OK, statuses(store)
            case "/api/fetch":
                return HTTPStatus.OK, fetch_summary(store)
            case "/api/replay":
                return HTTPStatus.OK, replay_summary(store)
            case "/api/forecast":
                day = date.fromisoformat(params["day"])
                return HTTPStatus.OK, forecast_view(store, params["model"], day, params["ticker"])
    except (KeyError, ValueError) as error:
        return HTTPStatus.BAD_REQUEST, {"error": f"bad or missing parameter: {error}"}
    except NotFound as error:
        return HTTPStatus.NOT_FOUND, {"error": str(error)}
    return HTTPStatus.NOT_FOUND, {"error": f"unknown endpoint {path}"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, store: Store, **kwargs):
        self.store = store
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        params = {key: values[0] for key, values in parse_qs(url.query).items()}
        code, payload = route(self.store, url.path, params)
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        log.debug(format, *args)


def server(store: Store, port: int, static: Path = STATIC) -> ThreadingHTTPServer:
    # localhost only: the page reads data/ and nothing guards it.
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, store=store, directory=str(static)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve a local page to follow the fetch, the replay and the forecasts.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with server(Store(args.data), args.port) as httpd:
        log.info("monitor on http://localhost:%d, reading %s", httpd.server_address[1], args.data.resolve())
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

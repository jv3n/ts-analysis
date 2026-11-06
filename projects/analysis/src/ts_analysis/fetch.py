import argparse
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from ts_analysis.massive import REQUESTS_PER_MINUTE, DayNotPublished, MassiveClient
from ts_analysis.selection import GapRule, ScannerRule, gap_candidates, sample, scanner_candidates
from ts_analysis.series import ET
from ts_analysis.status import Status
from ts_analysis.store import Store, bars_frame

log = logging.getLogger("ts_analysis.fetch")

PAST_SESSIONS = 5
# Calendar days: 5 sessions back still fit with a weekend and a holiday in between.
PAST_LOOKBACK = 10
PRICE_COLUMNS = ["open", "high", "low", "close", "vwap"]
# Grouped bars, splits and the common stocks over 6 pages; the previous session's grouped bars are
# usually the day before's, already stored.
NEW_DAY_REQUESTS = 8


class Source(Protocol):
    def grouped_daily(self, day: date) -> list[dict]: ...

    def minute_bars(self, ticker: str, start: date, end: date, adjusted: bool = False) -> list[dict]: ...

    def splits(self, day: date) -> list[dict]: ...

    def common_stocks(self, day: date) -> list[str]: ...


class PicksMismatch(Exception):
    pass


def weekdays(start: date, end: date) -> Iterator[date]:
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def cached(
    load: Callable[[date], list | None],
    save: Callable[[date, list], None],
    fetch: Callable[[date], list],
    day: date,
) -> list:
    rows = load(day)
    if rows is None:
        rows = fetch(day)
        save(day, rows)
    return rows


def grouped(source: Source, store: Store, day: date) -> list[dict]:
    return cached(store.load_grouped, store.save_grouped, source.grouped_daily, day)


def previous_session(source: Source, store: Store, day: date, max_lookback: int = 10) -> list[dict]:
    for offset in range(1, max_lookback + 1):
        candidate = day - timedelta(days=offset)
        if candidate.weekday() < 5 and (rows := grouped(source, store, candidate)):
            return rows
    raise LookupError(f"No session found in the {max_lookback} days before {day}")


def pick(
    source: Source, store: Store, day: date, rule: GapRule | ScannerRule, per_day: int, seed: str
) -> dict | None:
    today = grouped(source, store, day)
    if not today:
        return None
    prev = {row["T"]: row["c"] for row in previous_session(source, store, day)}
    splits = frozenset(split["ticker"] for split in cached(store.load_splits, store.save_splits, source.splits, day))
    common = cached(store.load_common, store.save_common, source.common_stocks, day) if rule.common_only else None
    common = None if common is None else frozenset(common)
    if isinstance(rule, ScannerRule):
        candidates = scanner_candidates(today, prev, rule, splits, common)
    else:
        found = gap_candidates({row["T"]: row["h"] for row in today}, prev, rule, splits, common)
        candidates = [{**asdict(c), "gap": round(c.gap, 4)} for c in found]
    return {
        "day": day.isoformat(),
        "rule": asdict(rule),
        "per_day": per_day,
        "seed": seed,
        "candidates": candidates,
        "picked": sample([c["ticker"] for c in candidates], per_day, seed=f"{seed}:{day.isoformat()}"),
    }


def past_sessions(adjusted: pd.DataFrame, raw: pd.DataFrame, day: date, sessions: int) -> pd.DataFrame | None:
    """The last `sessions` sessions before `day`, in the prices `day` traded at. Massive adjusts for
    every split up to the fetch: that keeps a split inside the window from reading as a jump, and the
    raw day set against its adjusted copy undoes the splits after it. None when the two share no bar."""
    traded = adjusted["time"].dt.tz_convert(ET).dt.date
    matched = raw.merge(adjusted[traded == day], on="time", suffixes=("", "_adjusted"))
    if matched.empty:
        return None
    factor = float((matched["close"] / matched["close_adjusted"]).median())
    keep = sorted(set(traded[traded < day]))[-sessions:]
    past = adjusted[traded.isin(keep)].reset_index(drop=True)
    past[PRICE_COLUMNS] *= factor
    past["volume"] /= factor
    return past


def check_settings(picks: dict, rule: GapRule | ScannerRule, per_day: int, seed: str) -> None:
    stored = (picks["rule"], picks["per_day"], picks["seed"])
    if stored != (asdict(rule), per_day, seed):
        raise PicksMismatch(
            f"{picks['day']} was picked with rule={stored[0]} per_day={stored[1]} seed={stored[2]!r}; "
            "rerun with the same settings or use another --data folder"
        )


def fetch_day(
    source: Source,
    store: Store,
    day: date,
    rule: GapRule | ScannerRule,
    per_day: int,
    seed: str,
    status: Status | None = None,
) -> dict | None:
    """Picks the day's tickers once, then fetches whatever bars are missing, so a rerun resumes."""
    picks = store.load_picks(day)
    if picks is None:
        picks = pick(source, store, day, rule, per_day, seed)
        if picks is None:
            return None
        store.save_picks(day, picks)
    else:
        check_settings(picks, rule, per_day, seed)
    picked = picks["picked"]
    log.info("%s %d candidates, %d picked: %s", day, len(picks["candidates"]), len(picked), " ".join(picked))
    for number, ticker in enumerate(picked, 1):
        if status:
            status.update(ticker=ticker, ticker_number=number, tickers=len(picked))
        done = []
        if not store.has_bars(day, ticker):
            rows = source.minute_bars(ticker, day, day)
            store.save_bars(day, ticker, rows)
            done.append(f"{len(rows)} bars")
        if not store.has_past(day, ticker):
            adjusted = bars_frame(source.minute_bars(ticker, day - timedelta(days=PAST_LOOKBACK), day, adjusted=True))
            past = past_sessions(adjusted, store.load_bars(day, ticker), day, PAST_SESSIONS)
            if past is None:
                log.warning("%s   %-6s adjusted and raw bars share no minute, no past sessions", day, ticker)
                past = adjusted.iloc[0:0]
            store.save_past(day, ticker, past)
            done.append(f"{past['time'].dt.tz_convert(ET).dt.date.nunique()} past sessions")
        log.info("%s   %2d/%d %-6s %s", day, number, len(picked), ticker, ", ".join(done) or "already stored")
    return picks


def planned_requests(store: Store, days: list[date], per_day: int) -> int:
    """The requests a backfill still has to make, as far as the store tells: what a picked day misses,
    a full day for one not picked yet, none for a day known to be a holiday."""
    total = 0
    for day in days:
        picks = store.load_picks(day)
        if picks is None:
            total += 0 if store.load_grouped(day) == [] else NEW_DAY_REQUESTS + 2 * per_day
            continue
        for ticker in picks["picked"]:
            total += (not store.has_bars(day, ticker)) + (not store.has_past(day, ticker))
    return total


def backfill(
    source: Source, store: Store, start: date, end: date, rule: GapRule | ScannerRule, per_day: int, seed: str
) -> None:
    days = list(weekdays(start, end))
    started = time.monotonic()
    requests = planned_requests(store, days, per_day)
    info = {"start": start.isoformat(), "end": end.isoformat(), "days": len(days), "requests": requests}
    log.info("%d requests to make, about %.0f min of quota", requests, requests / REQUESTS_PER_MINUTE)
    with Status(store.root, "fetch", **info, estimated_s=round(requests * 60 / REQUESTS_PER_MINUTE)) as status:
        for number, day in enumerate(days, 1):
            log.info("%s day %d/%d", day, number, len(days))
            status.update(day=day.isoformat(), day_number=number, ticker=None, ticker_number=0, tickers=0)
            try:
                picks = fetch_day(source, store, day, rule, per_day, seed, status)
            except DayNotPublished:
                log.info("%s not published yet, it opens after midnight ET: stopping", day)
                status.update(note=f"{day} not published yet")
                return
            if picks is None:
                log.info("%s market holiday", day)
            else:
                log.info(
                    "%s day %d/%d done, %.0f min elapsed", day, number, len(days), (time.monotonic() - started) / 60
                )


def last_published_day() -> date:
    return datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch the 1-min bars of gap tickers from Massive.")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=last_published_day())
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--per-day", type=int, default=10)
    parser.add_argument("--seed", default="ts-analysis")
    parser.add_argument(
        "--rule",
        choices=["gap", "scanner"],
        default="gap",
        help="gap: the session high --min-gap up; scanner: ts-scanner's premarket GUS and intraday pushers",
    )
    parser.add_argument("--min-gap", type=float, default=GapRule.min_gap)
    parser.add_argument("--min-price", type=float, default=GapRule.min_prev_close)
    parser.add_argument("--max-price", type=float, default=GapRule.max_prev_close)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error(f"--start {args.start} is after --end {args.end}")

    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        parser.error("MASSIVE_API_KEY is not set (see .env.example)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    rule = ScannerRule() if args.rule == "scanner" else GapRule(args.min_gap, args.min_price, args.max_price)
    try:
        client, store = MassiveClient(api_key), Store(args.data)
        backfill(client, store, args.start, args.end, rule, args.per_day, args.seed)
    except PicksMismatch as mismatch:
        parser.exit(1, f"error: {mismatch}\n")


if __name__ == "__main__":
    main()

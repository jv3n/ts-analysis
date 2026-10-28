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

from dotenv import find_dotenv, load_dotenv

from ts_analysis.massive import DayNotPublished, MassiveClient
from ts_analysis.selection import GapRule, gap_candidates, sample
from ts_analysis.store import Store

log = logging.getLogger("ts_analysis.fetch")


class Source(Protocol):
    def grouped_daily(self, day: date) -> list[dict]: ...

    def minute_bars(self, ticker: str, day: date) -> list[dict]: ...

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


def pick(source: Source, store: Store, day: date, rule: GapRule, per_day: int, seed: str) -> dict | None:
    today = grouped(source, store, day)
    if not today:
        return None
    prev = previous_session(source, store, day)
    splits = cached(store.load_splits, store.save_splits, source.splits, day)
    common = cached(store.load_common, store.save_common, source.common_stocks, day) if rule.common_only else None
    candidates = gap_candidates(
        {row["T"]: row["h"] for row in today},
        {row["T"]: row["c"] for row in prev},
        rule,
        frozenset(split["ticker"] for split in splits),
        None if common is None else frozenset(common),
    )
    return {
        "day": day.isoformat(),
        "rule": asdict(rule),
        "per_day": per_day,
        "seed": seed,
        "candidates": [{**asdict(c), "gap": round(c.gap, 4)} for c in candidates],
        "picked": sample([c.ticker for c in candidates], per_day, seed=f"{seed}:{day.isoformat()}"),
    }


def check_settings(picks: dict, rule: GapRule, per_day: int, seed: str) -> None:
    stored = (picks["rule"], picks["per_day"], picks["seed"])
    if stored != (asdict(rule), per_day, seed):
        raise PicksMismatch(
            f"{picks['day']} was picked with rule={stored[0]} per_day={stored[1]} seed={stored[2]!r}; "
            "rerun with the same settings or use another --data folder"
        )


def fetch_day(
    source: Source, store: Store, day: date, rule: GapRule, per_day: int, seed: str
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
        if store.has_bars(day, ticker):
            log.info("%s   %2d/%d %-6s already stored", day, number, len(picked), ticker)
            continue
        rows = source.minute_bars(ticker, day)
        store.save_bars(day, ticker, rows)
        log.info("%s   %2d/%d %-6s %d bars", day, number, len(picked), ticker, len(rows))
    return picks


def backfill(
    source: Source, store: Store, start: date, end: date, rule: GapRule, per_day: int, seed: str
) -> None:
    days = list(weekdays(start, end))
    started = time.monotonic()
    for number, day in enumerate(days, 1):
        log.info("%s day %d/%d", day, number, len(days))
        try:
            picks = fetch_day(source, store, day, rule, per_day, seed)
        except DayNotPublished:
            log.info("%s not published yet, it opens after midnight ET: stopping", day)
            return
        if picks is None:
            log.info("%s market holiday", day)
        else:
            log.info("%s day %d/%d done, %.0f min elapsed", day, number, len(days), (time.monotonic() - started) / 60)


def last_published_day() -> date:
    return datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch the 1-min bars of gap tickers from Massive.")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=last_published_day())
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--per-day", type=int, default=10)
    parser.add_argument("--seed", default="ts-analysis")
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
    rule = GapRule(args.min_gap, args.min_price, args.max_price)
    try:
        client, store = MassiveClient(api_key), Store(args.data)
        backfill(client, store, args.start, args.end, rule, args.per_day, args.seed)
    except PicksMismatch as mismatch:
        parser.exit(1, f"error: {mismatch}\n")


if __name__ == "__main__":
    main()

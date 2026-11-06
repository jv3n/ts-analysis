import random
from dataclasses import dataclass


@dataclass(frozen=True)
class GapRule:
    min_gap: float = 0.40
    min_prev_close: float = 0.30
    max_prev_close: float = 10.0
    common_only: bool = True


@dataclass(frozen=True)
class Candidate:
    ticker: str
    prev_close: float
    high: float

    @property
    def gap(self) -> float:
        return self.high / self.prev_close - 1


def gap_candidates(
    day_highs: dict[str, float],
    prev_closes: dict[str, float],
    rule: GapRule = GapRule(),
    split_tickers: frozenset[str] = frozenset(),
    common_stocks: frozenset[str] | None = None,
) -> list[Candidate]:
    """On a split day the raw high and previous close are on two scales: a 1-for-10 looks like +900 %.
    Warrants, rights and units gap 40 % on a few cents and move like options, hence `common_stocks`."""
    return sorted(
        (
            Candidate(ticker, prev, high)
            for ticker, high in day_highs.items()
            if ticker not in split_tickers
            and (common_stocks is None or ticker in common_stocks)
            and (prev := prev_closes.get(ticker)) is not None
            and rule.min_prev_close <= prev <= rule.max_prev_close
            and high >= prev * (1 + rule.min_gap)
        ),
        key=lambda candidate: candidate.ticker,
    )


def sample(tickers: list[str], n: int, seed: str) -> list[str]:
    """Random pick, never the top n by gap: keeping the biggest runners would bias every figure."""
    if len(tickers) <= n:
        return sorted(tickers)
    return sorted(random.Random(seed).sample(sorted(tickers), n))


@dataclass(frozen=True)
class ScannerRule:
    """`ts-scanner`'s population (`trade/docs/pattern/GUS.md`): the premarket GUS and the names that
    pushed during the session. Candidates come from the grouped daily bars, which hold the regular
    session only: an open +15 % up is what every GUS shows there."""

    name: str = "scanner"
    min_open: float = 0.30
    max_open: float = 10.0
    open_gap: float = 0.15
    push_gap: float = 0.25
    push_range: float = 0.40
    pm_gap: float = 0.45
    pm_push: float = 0.15
    pm_hold: float = -0.55
    common_only: bool = True


def scanner_candidates(
    today: list[dict],
    prev_closes: dict[str, float],
    rule: ScannerRule = ScannerRule(),
    split_tickers: frozenset[str] = frozenset(),
    common_stocks: frozenset[str] | None = None,
) -> list[dict]:
    """Grouped daily rows (`T`, `o`, `h`, `l`) in, the names that opened `open_gap` up, or whose
    session high reached `push_gap` over the previous close, or whose range reached `push_range`."""
    out = []
    for row in today:
        ticker, open_, high, low = row["T"], row["o"], row["h"], row["l"]
        prev = prev_closes.get(ticker)
        if ticker in split_tickers or (common_stocks is not None and ticker not in common_stocks):
            continue
        if not prev or not rule.min_open <= open_ <= rule.max_open:
            continue
        opened = open_ >= prev * (1 + rule.open_gap)
        pushed = high >= prev * (1 + rule.push_gap) or (low > 0 and high >= low * (1 + rule.push_range))
        if opened or pushed:
            out.append({"ticker": ticker, "prev_close": prev, "open": open_, "high": high, "low": low,
                        "gap": round(high / prev - 1, 4)})
    return sorted(out, key=lambda candidate: candidate["ticker"])

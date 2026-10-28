import random
from dataclasses import dataclass


@dataclass(frozen=True)
class GapRule:
    min_gap: float = 0.40
    min_prev_close: float = 0.30
    max_prev_close: float = 10.0


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
) -> list[Candidate]:
    """On a split day the raw high and previous close are on two scales: a 1-for-10 looks like +900 %."""
    return sorted(
        (
            Candidate(ticker, prev, high)
            for ticker, high in day_highs.items()
            if ticker not in split_tickers
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

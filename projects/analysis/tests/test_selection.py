from ts_analysis.selection import Candidate, GapRule, gap_candidates, sample

PREV_CLOSES = {"MULN": 0.42, "FFIE": 2.50, "NVDA": 135.20, "SNDL": 2.10, "BBIG": 0.12}


def tickers(candidates: list[Candidate]) -> list[str]:
    return [candidate.ticker for candidate in candidates]


def test_a_high_exactly_40_percent_above_the_previous_close_makes_a_candidate():
    assert gap_candidates({"FFIE": 3.50}, PREV_CLOSES) == [Candidate("FFIE", 2.50, 3.50)]


def test_a_high_just_under_40_percent_is_not_a_candidate():
    assert gap_candidates({"FFIE": 3.49}, PREV_CLOSES) == []


def test_a_candidate_carries_its_gap():
    assert round(gap_candidates({"FFIE": 4.00}, PREV_CLOSES)[0].gap, 4) == 0.6


def test_the_price_range_includes_both_bounds():
    closes = {"CTXR": 0.30, "IMPP": 10.00}
    assert tickers(gap_candidates({"CTXR": 0.45, "IMPP": 15.00}, closes)) == ["CTXR", "IMPP"]


def test_a_previous_close_outside_the_price_range_excludes_the_ticker():
    closes = {"CTXR": 0.29, "IMPP": 10.01, **PREV_CLOSES}
    assert gap_candidates({"CTXR": 0.60, "IMPP": 20.00, "NVDA": 200.00, "BBIG": 0.30}, closes) == []


def test_a_ticker_without_a_previous_close_is_skipped():
    assert gap_candidates({"RDDT": 80.00}, PREV_CLOSES) == []


def test_a_warrant_is_not_a_candidate_when_only_common_stocks_count():
    closes = {"AISPW": 0.31, "AISP": 2.40}
    highs = {"AISPW": 0.62, "AISP": 3.50}
    assert tickers(gap_candidates(highs, closes, common_stocks=frozenset({"AISP"}))) == ["AISP"]


def test_a_reverse_split_on_the_day_is_not_taken_for_a_gap():
    # A 1-for-10 on raw prices: the previous close is pre-split, the day's high post-split.
    assert gap_candidates({"MULN": 4.60}, PREV_CLOSES, split_tickers=frozenset({"MULN"})) == []


def test_the_threshold_and_price_range_come_from_the_rule():
    rule = GapRule(min_gap=0.20, min_prev_close=1.00, max_prev_close=5.00)
    assert tickers(gap_candidates({"SNDL": 2.55, "MULN": 0.80}, PREV_CLOSES, rule)) == ["SNDL"]


def test_every_ticker_is_kept_when_there_are_fewer_than_asked():
    assert sample(["SNDL", "FFIE"], 10, seed="s") == ["FFIE", "SNDL"]


def test_the_sample_is_reproducible_for_a_given_seed_whatever_the_input_order():
    names = [f"T{i:02d}" for i in range(30)]
    seed = "ts-analysis:2026-10-27"
    assert sample(names, 10, seed) == sample(list(reversed(names)), 10, seed)


def test_different_days_draw_different_samples():
    names = [f"T{i:02d}" for i in range(30)]
    assert sample(names, 10, "ts-analysis:2026-10-26") != sample(names, 10, "ts-analysis:2026-10-27")

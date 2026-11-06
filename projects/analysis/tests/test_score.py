import numpy as np
import pandas as pd
import pytest

from ts_analysis.replay import QUANTILES
from ts_analysis.score import (
    COLUMNS,
    big_moves,
    intervals,
    main,
    pinball,
    quantile_coverage,
    report,
    resample_weights,
    scoreboard,
    with_outcome,
)


def forecast(model: str, actual: float, band: tuple[float, float, float], horizon: int = 15, to_close: bool = False):
    """FFIE at $2.40, a band given as simple moves (q10, q50, q90) and spread linearly in log space."""
    low, mid, high = (np.log1p(m) for m in band)
    levels = np.array(QUANTILES)
    moves = np.where(levels <= 0.5, low + (mid - low) * (levels - 0.1) / 0.4, mid + (high - mid) * (levels - 0.5) / 0.4)
    return {
        "model": model,
        "ticker": "FFIE",
        "day": "2026-10-26",
        "cut": "10:00",
        "horizon": horizon,
        "to_close": to_close,
        "last": 2.40,
        "actual": actual,
        **dict(zip(COLUMNS, moves)),
    }


def board_row(rows: list[dict], model: str = "timestral-28M", call: float = 0.05) -> pd.Series:
    board = scoreboard(pd.DataFrame(rows), call)
    return board[board["model"] == model].iloc[0]


def test_a_forecast_whose_every_quantile_is_the_real_move_scores_zero():
    row = forecast("timestral-28M", 2.16, (-0.10, -0.10, -0.10))
    assert pinball(with_outcome(pd.DataFrame([row]))).iloc[0] == pytest.approx(0, abs=1e-12)


def test_half_the_random_walk_loss_is_a_50_percent_skill():
    rows = [forecast("random-walk", 2.40, (-0.20, 0.0, 0.20)), forecast("timestral-28M", 2.40, (-0.10, 0.0, 0.10))]
    assert board_row(rows)["skill"] == pytest.approx(0.5, abs=0.05)


def test_the_real_price_inside_the_10_90_band_counts_toward_its_coverage():
    rows = [forecast("timestral-28M", 2.30, (-0.10, 0.0, 0.10)), forecast("timestral-28M", 1.80, (-0.10, 0.0, 0.10))]
    assert board_row(rows)["coverage_80"] == 0.5


def test_a_flat_median_says_no_direction_and_is_left_out_of_the_direction_rate():
    rows = [forecast("timestral-28M", 2.16, (-0.10, -0.02, 0.05)), forecast("timestral-28M", 2.16, (-0.10, 0.0, 0.10))]
    assert board_row(rows)["direction"] == 1.0


def test_a_median_5_percent_down_is_a_down_call_hit_when_the_price_fell():
    rows = [forecast("timestral-28M", 1.68, (-0.40, -0.30, -0.10)), forecast("timestral-28M", 2.50, (-0.10, -0.06, 0.02))]
    row = board_row(rows)
    assert (row["down_calls"], row["down_hit"], row["down_real"]) == (2, 0.5, pytest.approx((-0.30 + 0.0417) / 2, abs=1e-3))


def test_a_band_wholly_below_zero_is_a_90_percent_sure_call():
    rows = [forecast("timestral-28M", 2.10, (-0.20, -0.10, -0.01)), forecast("timestral-28M", 2.10, (-0.20, -0.10, 0.01))]
    assert (board_row(rows)["sure_calls"], board_row(rows)["sure_hit"]) == (1, 1.0)


def test_the_last_cut_of_the_day_is_scored_as_a_close_forecast():
    row = forecast("timestral-28M", 2.40, (-0.05, 0.0, 0.05), horizon=5, to_close=True)
    assert list(scoreboard(pd.DataFrame([row]))["bucket"]) == ["close"]


def test_a_calibrated_median_has_half_the_real_moves_below_it():
    rows = [forecast("timestral-28M", price, (-0.10, 0.0, 0.10)) for price in (2.30, 2.35, 2.45, 2.50)]
    assert quantile_coverage(pd.DataFrame(rows))["q50"].iloc[0] == 0.5


def test_the_report_shows_a_dash_where_a_figure_has_no_case():
    rows = [forecast("random-walk", 2.40, (-0.05, 0.0, 0.05)), forecast("timestral-28M", 2.45, (-0.05, 0.0, 0.05))]
    assert "| – |" in report(pd.DataFrame(rows), 0.05)


def test_the_cli_says_to_replay_first_when_there_is_no_forecast(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--data", str(tmp_path)])
    assert "run ts-replay first" in capsys.readouterr().err


def at(cut: str, last: float, actual: float, band: tuple[float, float, float], model: str = "timestral-28M") -> dict:
    return forecast(model, actual, band) | {"cut": cut, "last": last}


def big_row(rows: list[dict], model: str = "timestral-28M") -> pd.Series:
    board = big_moves(pd.DataFrame(rows), big=0.05)
    return board[board["model"] == model].iloc[0]


def test_a_big_move_is_picked_on_the_real_move_never_on_the_forecast():
    rows = [
        at("10:00", 2.40, 2.42, (-0.50, -0.30, 0.50)),  # the model screams, the price barely moves
        at("10:05", 2.42, 2.66, (-0.01, 0.0, 0.01)),  # the model shrugs, the price jumps 10 %
    ]
    row = big_row(rows)
    assert (row["cuts"], row["big"], row["big_share"]) == (2, 1, 0.5)


def test_a_big_drop_counts_as_much_as_a_big_rise():
    rows = [at("10:00", 2.40, 2.25, (-0.10, -0.02, 0.05)), at("10:05", 2.25, 2.40, (-0.05, 0.02, 0.10))]
    row = big_row(rows)
    assert (row["big"], row["big_down"]) == (2, 0.5)


def test_a_median_leaning_the_way_of_the_big_move_is_a_hit():
    rows = [at("10:00", 2.40, 2.16, (-0.15, -0.02, 0.05)), at("10:05", 2.16, 2.40, (-0.15, -0.02, 0.05))]
    assert big_row(rows)["lean_hit"] == 0.5


def test_a_flat_median_does_not_lean_so_a_baseline_has_no_lean_hit():
    rows = [at("10:00", 2.40, 2.16, (-0.05, 0.0, 0.05), model="random-walk")]
    assert np.isnan(big_row(rows, model="random-walk")["lean_hit"])


def test_a_big_move_already_under_way_at_the_cut_is_not_quiet():
    rows = [
        at("11:35", 2.40, 2.41, (-0.05, 0.01, 0.05)),
        at("11:40", 2.41, 2.46, (-0.05, 0.01, 0.05)),
        at("11:45", 2.46, 2.75, (-0.05, 0.03, 0.15)),  # +2.1 % since 11:40, under half of 5 %: still quiet
        at("11:50", 2.75, 3.05, (-0.05, 0.03, 0.15)),  # +11.8 % since 11:45: the move is under way
    ]
    row = big_row(rows)
    assert (row["big"], row["quiet"], row["quiet_hit"]) == (2, 1, 1.0)


def test_a_band_wider_than_the_ticker_day_s_usual_one_before_a_big_move_reads_above_1():
    usual = [at(f"10:{m:02d}", 2.40, 2.41, (-0.05, 0.0, 0.05)) for m in (0, 5, 10, 15)]
    wide = at("10:20", 2.40, 2.70, (-0.10, 0.01, 0.12))
    row = big_row([*usual, wide])
    assert row["width_big"] > 2 and row["width_rest"] == pytest.approx(1.0)


def test_the_report_has_a_big_moves_section():
    rows = [at("10:00", 2.40, 2.66, (-0.05, 0.01, 0.05)), at("10:00", 2.40, 2.66, (-0.05, 0.0, 0.05), model="random-walk")]
    assert "## Big moves" in report(pd.DataFrame(rows), 0.05, 0.05)


def on(ticker: str, day: str, cut: str, row: dict) -> dict:
    return row | {"ticker": ticker, "day": day, "cut": cut}


def interval_row(rows: list[dict], model: str = "timestral-28M", **options) -> pd.Series:
    board = intervals(pd.DataFrame(rows), **options)
    return board[board["model"] == model].iloc[0]


def test_each_resample_draws_as_many_ticker_days_as_there_are():
    weights = resample_weights(units=7, resamples=50, seed=0)
    assert weights.shape == (50, 7) and (weights.sum(axis=1) == 7).all()


def test_the_resampling_draws_whole_ticker_days_never_single_cuts():
    # FFIE stays inside its band all day, MULN outside all day: drawn by cut, 156 cuts would pin the
    # coverage near 50 %; drawn by ticker-day, a resample can hold FFIE twice or MULN twice.
    cuts = [f"{9 + (30 + 5 * i) // 60:02d}:{(30 + 5 * i) % 60:02d}" for i in range(78)]
    band = (-0.05, 0.0, 0.05)
    rows = [on("FFIE", "2026-10-26", c, forecast("timestral-28M", 2.41, band)) for c in cuts]
    rows += [on("MULN", "2026-10-26", c, forecast("timestral-28M", 3.10, band)) for c in cuts]
    row = interval_row(rows, resamples=200)
    assert (row["coverage_80_low"], row["coverage_80_high"]) == (0.0, 1.0)


def test_a_model_equal_to_the_random_walk_has_a_skill_interval_of_exactly_zero():
    rows = [
        on(ticker, "2026-10-26", "10:00", forecast(model, actual, (-0.05, 0.0, 0.05)))
        for ticker, actual in [("FFIE", 2.30), ("MULN", 2.52), ("GCTK", 2.41)]
        for model in ("random-walk", "timestral-28M")
    ]
    row = interval_row(rows, resamples=100)
    assert (row["skill_low"], row["skill_high"]) == (pytest.approx(0, abs=1e-12), pytest.approx(0, abs=1e-12))


def test_the_same_seed_gives_the_same_intervals():
    rows = [
        on(ticker, day, "10:00", forecast(model, actual, band))
        for ticker, day, actual in [("FFIE", "2026-10-26", 2.30), ("MULN", "2026-10-27", 2.52), ("GCTK", "2026-10-28", 2.41)]
        for model, band in [("random-walk", (-0.05, 0.0, 0.05)), ("timestral-28M", (-0.08, -0.01, 0.04))]
    ]
    first, again = intervals(pd.DataFrame(rows), seed=7), intervals(pd.DataFrame(rows), seed=7)
    pd.testing.assert_frame_equal(first, again)


def test_down_calls_from_one_ticker_day_count_as_one_day():
    rows = [on("FFIE", "2026-10-26", c, forecast("timestral-28M", 2.10, (-0.30, -0.10, 0.0))) for c in ("10:00", "10:05", "10:10")]
    rows.append(on("MULN", "2026-10-26", "10:00", forecast("timestral-28M", 2.10, (-0.30, -0.10, 0.0))))
    assert interval_row(rows, resamples=10)["down_days"] == 2


def test_the_report_gives_the_skill_interval_next_to_the_skill():
    rows = [
        on(ticker, "2026-10-26", "10:00", forecast(model, actual, (-0.05, 0.0, 0.05)))
        for ticker, actual in [("FFIE", 2.30), ("MULN", 2.52)]
        for model in ("random-walk", "timestral-28M")
    ]
    assert "skill_95" in report(pd.DataFrame(rows), 0.05, resamples=20)


def test_the_cli_scores_only_the_days_between_start_and_end(tmp_path, capsys):
    folder = tmp_path / "forecasts" / "random-walk"
    folder.mkdir(parents=True)
    for day in ("2026-10-27", "2026-10-28"):
        row = on("FFIE", day, "10:00", forecast("random-walk", 2.30, (-0.05, 0.0, 0.05)))
        pd.DataFrame([row]).to_parquet(folder / f"{day}.parquet", index=False)
    main(["--data", str(tmp_path), "--start", "2026-10-28", "--resamples", "10"])
    assert "1 days, 1 ticker-days, 1 forecasts" in capsys.readouterr().out


def test_the_skill_compares_a_model_with_the_random_walk_on_its_own_ticker_days_only():
    # The random walk also ran on 2026-10-28, a wild day the model never saw: it must not flatter the model.
    band = (-0.05, 0.0, 0.05)
    rows = [on("FFIE", "2026-10-27", "10:00", forecast(m, 2.30, band)) for m in ("random-walk", "timestral-28M")]
    rows.append(on("MULN", "2026-10-28", "10:00", forecast("random-walk", 4.80, band)))
    assert board_row(rows)["skill"] == pytest.approx(0, abs=1e-12)
    assert interval_row(rows, resamples=50)["skill_high"] == pytest.approx(0, abs=1e-12)

import numpy as np
import pandas as pd
import pytest

from ts_analysis.replay import QUANTILES
from ts_analysis.score import COLUMNS, big_moves, main, pinball, quantile_coverage, report, scoreboard, with_outcome


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

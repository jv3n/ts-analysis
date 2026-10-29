import numpy as np
import pandas as pd
import pytest

from ts_analysis.replay import QUANTILES
from ts_analysis.score import COLUMNS, main, pinball, quantile_coverage, report, scoreboard, with_outcome


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

from dataclasses import dataclass

import numpy as np
import pytest

from ts_analysis.replay import PRICE_FLOOR, QUANTILES, Timestral
from ts_analysis.series import CLOSE, OPEN


@dataclass
class Out:
    quantiles: np.ndarray
    quantile_levels: tuple[float, ...] = QUANTILES


class FakeTimestral:
    """Records what reaches the model and returns `price` on every quantile, in the input's scale."""

    def __init__(self, price: float = 2.40):
        self.price = price
        self.calls: list[dict] = []

    def forecast(
        self, context, prediction_length, past_covariates, known_covariates, clip_nonnegative, symmetric_averaging
    ):
        self.calls.append(
            {
                "context": context.copy(),
                "known": known_covariates,
                "clip_nonnegative": clip_nonnegative,
                "symmetric_averaging": symmetric_averaging,
            }
        )
        value = self.price if np.nanmax(context) > 1.5 else np.log(self.price)
        return Out(np.full((len(context), 1, prediction_length, len(QUANTILES)), value))


# BEAT at $2.40, then $2.52 with a minute without trade carried at $2.52, then $2.46.
LOG_CLOSE = np.log(np.array([[np.nan, 2.40, 2.52, 2.52, 2.46]]))
VOLUME = np.array([[0.0, 18_000, 9_500, 0, 4_200]])


def wrapped(wrapping: str, price: float = 2.40) -> tuple[FakeTimestral, np.ndarray]:
    fake = FakeTimestral(price)
    return fake, Timestral("28M", wrapping, model=fake).forecast(LOG_CLOSE, VOLUME, length=5)


def test_the_log_wrapping_feeds_log_prices_with_the_carried_minutes_as_observed():
    fake, _ = wrapped("log")
    np.testing.assert_array_equal(fake.calls[0]["context"][:, 0], LOG_CLOSE)


def test_the_raw_wrapping_feeds_prices_and_returns_the_log_of_the_forecast_prices():
    fake, out = wrapped("raw")
    np.testing.assert_allclose(fake.calls[0]["context"][0, 0], [np.nan, 2.40, 2.52, 2.52, 2.46])
    np.testing.assert_allclose(out, np.log(2.40))


def test_the_raw_nan_wrapping_leaves_a_minute_without_trade_missing_instead_of_carried():
    fake, _ = wrapped("raw-nan")
    np.testing.assert_allclose(fake.calls[0]["context"][0, 0], [np.nan, 2.40, 2.52, np.nan, 2.46])


def test_a_raw_quantile_clipped_to_zero_reads_as_the_price_floor_not_minus_infinity():
    _, out = wrapped("raw", price=0.0)
    assert np.isfinite(out).all() and out.max() == pytest.approx(np.log(PRICE_FLOOR))


def test_the_non_negative_clip_stays_on_for_prices_and_is_off_for_capped_log_prices():
    assert wrapped("raw")[0].calls[0]["clip_nonnegative"] is True
    assert wrapped("log-capped")[0].calls[0]["clip_nonnegative"] is False


def test_each_wrapping_is_stored_under_its_own_model_name_and_the_log_one_keeps_the_original_name():
    names = [Timestral("28M", w, model=FakeTimestral()).name for w in ("log", "raw", "raw-nan", "log-capped")]
    assert names == ["timestral-28M", "timestral-28M-raw", "timestral-28M-raw-nan", "timestral-28M-log-capped"]


def test_the_wrapping_never_rewrites_the_grid_it_was_given():
    before = LOG_CLOSE.copy()
    wrapped("raw-nan")
    np.testing.assert_array_equal(LOG_CLOSE, before)


def test_the_clock_covers_the_context_and_the_horizon_from_0_at_the_open_to_the_last_minute():
    fake = FakeTimestral()
    log_close = np.log(np.full((2, OPEN), 2.40))  # BEAT and APUS cut at 9:30
    Timestral("28M", "clock", model=fake).forecast(log_close, np.full((2, OPEN), 1000.0), length=CLOSE - OPEN)
    known = fake.calls[0]["known"]
    assert known.shape == (2, 1, CLOSE)
    assert (known[:, 0, OPEN], known[:, 0, -1]) == (pytest.approx(0.0), pytest.approx(389 / 390))
    assert known[0, 0, 0] < 0  # 4:00, before the open


def test_the_clock_holds_only_the_time_never_a_price():
    fake = FakeTimestral()
    log_close = np.log(np.array([[2.40, 2.52, 2.46], [0.61, 0.58, 0.64]]))
    Timestral("28M", "clock", model=fake).forecast(log_close, np.ones((2, 3)), length=5)
    known = fake.calls[0]["known"]
    np.testing.assert_array_equal(known[0], known[1])


def test_without_the_clock_no_known_covariate_reaches_the_model():
    assert wrapped("log-capped")[0].calls[0]["known"] is None


def test_the_asym_wrappings_turn_the_symmetric_averaging_off_and_the_others_keep_it():
    assert [wrapped(w)[0].calls[0]["symmetric_averaging"] for w in ("log-capped", "clock", "asym", "clock-asym")] == [
        True,
        True,
        False,
        False,
    ]

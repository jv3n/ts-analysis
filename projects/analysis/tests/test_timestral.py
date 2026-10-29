from dataclasses import dataclass

import numpy as np
import pytest

from ts_analysis.replay import PRICE_FLOOR, QUANTILES, Timestral


@dataclass
class Out:
    quantiles: np.ndarray
    quantile_levels: tuple[float, ...] = QUANTILES


class FakeTimestral:
    """Records what reaches the model and returns `price` on every quantile, in the input's scale."""

    def __init__(self, price: float = 2.40):
        self.price = price
        self.calls: list[dict] = []

    def forecast(self, context, prediction_length, past_covariates, clip_nonnegative):
        self.calls.append({"context": context.copy(), "clip_nonnegative": clip_nonnegative})
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

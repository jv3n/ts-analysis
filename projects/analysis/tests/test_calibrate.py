from datetime import date

import numpy as np
import pandas as pd
import pytest

from ts_analysis.calibrate import calibrate, main
from ts_analysis.replay import QUANTILES
from ts_analysis.score import COLUMNS
from ts_analysis.store import Store

DAYS = ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"]
DAYS += ["2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]
# Real moves spread evenly around 0, twice the half-width of the bands below at the edges.
SPREAD = (-0.10, -0.06, -0.02, 0.02, 0.06, 0.10)


def forecast(day: str, real: float, band: tuple[float, float, float], cut: str = "10:00", model: str = "timestral-28M"):
    """FFIE at $2.40 and a band of log moves (q10, q50, q90), spread linearly over the 21 levels."""
    low, mid, high = band
    levels = np.array(QUANTILES)
    moves = np.where(levels <= 0.5, low + (mid - low) * (levels - 0.1) / 0.4, mid + (high - mid) * (levels - 0.5) / 0.4)
    return {
        "model": model,
        "ticker": "FFIE",
        "day": day,
        "cut": cut,
        "horizon": 15,
        "to_close": False,
        "last": 2.40,
        "actual": 2.40 * np.exp(real),
        **dict(zip(COLUMNS, moves)),
    }


def session(day: str, offset: float = 0.0, band=(-0.05, 0.0, 0.05)) -> list[dict]:
    return [forecast(day, real + offset, band, cut=f"10:{5 * i:02d}") for i, real in enumerate(SPREAD)]


def on(result: pd.DataFrame, day: str) -> pd.DataFrame:
    return result[result["day"] == day]


def test_the_correction_applied_to_a_day_reads_no_forecast_of_that_day_or_after():
    rows = [r for day in DAYS[:7] for r in session(day)]
    first = calibrate(pd.DataFrame(rows))
    # The same days, but the 6th and 7th end 40 % higher than forecast: day 6 must not move.
    wild = [r | {"actual": r["actual"] * 1.4} if r["day"] >= DAYS[5] else r for r in rows]
    again = calibrate(pd.DataFrame(wild))
    pd.testing.assert_frame_equal(
        on(first, DAYS[5])[COLUMNS].reset_index(drop=True), on(again, DAYS[5])[COLUMNS].reset_index(drop=True)
    )


def test_a_median_too_low_over_the_past_days_is_shifted_up_by_the_typical_miss():
    rows = [r for day in DAYS[:5] for r in session(day, offset=0.03)] + [forecast(DAYS[5], 0.0, (-0.05, 0.0, 0.05))]
    assert on(calibrate(pd.DataFrame(rows)), DAYS[5])["q50"].iloc[0] == pytest.approx(0.03)


def test_a_band_too_narrow_over_the_past_days_is_widened_until_80_percent_fit():
    # Past reach over the half-width: 2, 1.2, 0.4, 0.4, 1.2, 2; its 80th percentile is 2.
    rows = [r for day in DAYS[:5] for r in session(day)] + [forecast(DAYS[5], 0.0, (-0.05, 0.0, 0.05))]
    row = on(calibrate(pd.DataFrame(rows)), DAYS[5]).iloc[0]
    assert (row["q10"], row["q50"], row["q90"]) == (pytest.approx(-0.10), pytest.approx(0.0), pytest.approx(0.10))


def test_each_side_of_the_band_is_scaled_by_its_own_width():
    rows = [r for day in DAYS[:5] for r in session(day)] + [forecast(DAYS[5], 0.0, (-0.08, 0.0, 0.02))]
    row = on(calibrate(pd.DataFrame(rows)), DAYS[5]).iloc[0]
    assert row["q90"] - row["q50"] < row["q50"] - row["q10"]


def test_the_quantiles_stay_in_order_after_the_correction():
    rows = [r for day in DAYS[:5] for r in session(day, offset=-0.02)] + session(DAYS[5])
    values = on(calibrate(pd.DataFrame(rows)), DAYS[5])[COLUMNS].to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()


def test_a_day_with_fewer_past_days_than_needed_is_left_out():
    rows = [r for day in DAYS[:6] for r in session(day)]
    assert sorted(calibrate(pd.DataFrame(rows), min_days=5)["day"].unique()) == [DAYS[5]]


def test_the_window_forgets_the_days_older_than_it():
    rows = [r for day in DAYS[:5] for r in session(day, offset=0.10)]
    rows += [r for day in DAYS[5:10] for r in session(day)]
    rows.append(forecast(DAYS[10], 0.0, (-0.05, 0.0, 0.05)))
    windowed = on(calibrate(pd.DataFrame(rows), window=5), DAYS[10])["q50"].iloc[0]
    every = on(calibrate(pd.DataFrame(rows)), DAYS[10])["q50"].iloc[0]
    assert windowed == pytest.approx(0.0) and every > 0


def test_the_corrected_model_is_stored_under_its_own_name():
    rows = [forecast(day, 0.01, (-0.05, 0.0, 0.05), model="timestral-28M-log-capped-past5") for day in DAYS[:6]]
    assert list(calibrate(pd.DataFrame(rows))["model"].unique()) == ["timestral-28M-log-capped-past5-cal"]


def test_the_cli_writes_the_corrected_days_next_to_the_raw_model(tmp_path):
    folder = tmp_path / "forecasts" / "timestral-28M"
    folder.mkdir(parents=True)
    for day in DAYS[:6]:
        pd.DataFrame(session(day)).to_parquet(folder / f"{day}.parquet", index=False)
    main(["--data", str(tmp_path), "--all-cuts"])
    assert [p.name for p in (tmp_path / "forecasts" / "timestral-28M-cal").glob("*.parquet")] == [f"{DAYS[5]}.parquet"]


def test_the_cli_never_corrects_the_flat_baseline_or_a_corrected_model(tmp_path):
    for model in ("flat", "timestral-28M-cal"):
        folder = tmp_path / "forecasts" / model
        folder.mkdir(parents=True)
        for day in DAYS[:6]:
            pd.DataFrame([r | {"model": model} for r in session(day)]).to_parquet(folder / f"{day}.parquet", index=False)
    main(["--data", str(tmp_path), "--all-cuts"])
    assert sorted(p.name for p in (tmp_path / "forecasts").iterdir()) == ["flat", "timestral-28M-cal"]


def entered(rows: list[dict], ticker: str, setup: str) -> list[dict]:
    return [r | {"ticker": ticker, "setup": setup} for r in rows]


def gus_and_push(gus_offset: float, push_offset: float, days: list[str] = DAYS[:5]) -> list[dict]:
    """BNAI gapped up in the premarket (gus), AIOK pushed during the session (push), each with its own bias."""
    rows = [r for day in days for r in entered(session(day, offset=gus_offset), "BNAI", "gus")]
    return rows + [r for day in days for r in entered(session(day, offset=push_offset), "AIOK", "push")]


def today(setup: str, ticker: str, real: float = 0.0) -> dict:
    return forecast(DAYS[5], real, (-0.05, 0.0, 0.05)) | {"ticker": ticker, "setup": setup}


def test_a_gus_is_corrected_from_the_past_gus_only_when_learned_per_setup():
    rows = gus_and_push(0.03, -0.03) + [today("gus", "BNAI"), today("push", "AIOK")]
    split = on(calibrate(pd.DataFrame(rows), by_setup=True), DAYS[5]).set_index("ticker")["q50"]
    pooled = on(calibrate(pd.DataFrame(rows)), DAYS[5]).set_index("ticker")["q50"]
    assert (split["BNAI"], split["AIOK"]) == (pytest.approx(0.03), pytest.approx(-0.03))
    assert pooled["BNAI"] == pytest.approx(pooled["AIOK"])


def test_the_per_setup_correction_of_a_day_reads_no_forecast_of_that_day_or_after():
    rows = gus_and_push(0.0, 0.0, DAYS[:7])
    first = calibrate(pd.DataFrame(rows), by_setup=True)
    wild = [r | {"actual": r["actual"] * 1.4} if r["day"] >= DAYS[5] else r for r in rows]
    again = calibrate(pd.DataFrame(wild), by_setup=True)
    pd.testing.assert_frame_equal(
        on(first, DAYS[5])[COLUMNS].reset_index(drop=True), on(again, DAYS[5])[COLUMNS].reset_index(drop=True)
    )


def test_a_setup_seen_on_fewer_past_days_than_needed_is_left_out_while_the_other_is_corrected():
    rows = gus_and_push(0.0, 0.0)
    rows = [r for r in rows if not (r["setup"] == "gus" and r["day"] == DAYS[0])]
    rows += [today("gus", "BNAI"), today("push", "AIOK")]
    assert list(on(calibrate(pd.DataFrame(rows), by_setup=True), DAYS[5])["ticker"].unique()) == ["AIOK"]


def test_the_per_setup_correction_is_stored_under_its_own_name():
    rows = gus_and_push(0.0, 0.0) + [today("gus", "BNAI")]
    assert list(calibrate(pd.DataFrame(rows), by_setup=True)["model"].unique()) == ["timestral-28M-cal-setup"]


def test_the_cli_writes_the_per_setup_correction_next_to_the_pooled_one(tmp_path):
    # FFIE closed at $1.70: +40 % is $2.38, reached on its 9:30 bar, before every cut.
    store = Store(tmp_path)
    folder = tmp_path / "forecasts" / "timestral-28M"
    folder.mkdir(parents=True)
    for day in DAYS[:6]:
        store.save_picks(
            date.fromisoformat(day),
            {"day": day, "rule": {"min_gap": 0.4}, "candidates": [{"ticker": "FFIE", "prev_close": 1.70}], "picked": ["FFIE"]},
        )
        store.save_bars(date.fromisoformat(day), "FFIE", [
            {"t": int(pd.Timestamp(f"{day} 09:30", tz="America/New_York").timestamp() * 1000),
             "o": 2.30, "h": 2.40, "l": 2.25, "c": 2.40, "v": 50000}
        ])
        pd.DataFrame(session(day)).to_parquet(folder / f"{day}.parquet", index=False)
    main(["--data", str(tmp_path)])
    main(["--data", str(tmp_path), "--by-setup"])
    written = tmp_path / "forecasts" / "timestral-28M-cal-setup" / f"{DAYS[5]}.parquet"
    assert (tmp_path / "forecasts" / "timestral-28M-cal" / f"{DAYS[5]}.parquet").exists() and written.exists()
    assert "setup" not in pd.read_parquet(written).columns

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ts_analysis.entry import after_entry, entries, with_setup
from ts_analysis.replay import QUANTILES
from ts_analysis.store import Store

COLUMNS = [f"q{round(q * 100):02d}" for q in QUANTILES]
BUCKETS = ["5", "15", "30", "close"]


def load_forecasts(data: Path) -> pd.DataFrame:
    files = sorted((data / "forecasts").glob("*/*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def with_outcome(forecasts: pd.DataFrame) -> pd.DataFrame:
    """The last cut of the day forecasts 5 minutes that are also the close: it counts for the close."""
    frame = forecasts.copy()
    frame["bucket"] = np.where(frame["to_close"], "close", frame["horizon"].astype(str))
    frame["real"] = np.log(frame["actual"] / frame["last"])
    return frame


def pinball(frame: pd.DataFrame) -> pd.Series:
    """Mean quantile loss over the 21 levels, in log moves: the proper score of a probabilistic forecast,
    lower is better."""
    predicted = frame[COLUMNS].to_numpy()
    error = frame["real"].to_numpy()[:, None] - predicted
    levels = np.array(QUANTILES)[None, :]
    return pd.Series(np.maximum(levels * error, (levels - 1) * error).mean(axis=1), index=frame.index)


def _rate(hits: pd.Series) -> float:
    return float(hits.mean()) if len(hits) else float("nan")


def figures(group: pd.DataFrame, call: float) -> dict:
    real, median = group["real"], group["q50"]
    says = (median != 0) & (real != 0)
    down_call, up_call = median <= np.log1p(-call), median >= np.log1p(call)
    sure_down, sure_up = group["q90"] < 0, group["q10"] > 0
    return {
        "n": len(group),
        "pinball": float(pinball(group).mean()),
        "mae": float(np.expm1((median - real).abs()).mean()),
        "mdae": float(np.expm1((median - real).abs()).median()),
        "coverage_80": _rate((group["q10"] <= real) & (real <= group["q90"])),
        "direction": _rate(np.sign(median[says]) == np.sign(real[says])),
        "real_down": _rate(real < 0),
        "down_calls": int(down_call.sum()),
        "down_hit": _rate(real[down_call] < 0),
        "down_real": float(np.expm1(real[down_call]).mean()) if down_call.any() else float("nan"),
        "up_calls": int(up_call.sum()),
        "up_hit": _rate(real[up_call] > 0),
        "up_real": float(np.expm1(real[up_call]).mean()) if up_call.any() else float("nan"),
        "sure_calls": int((sure_down | sure_up).sum()),
        "sure_hit": _rate(pd.concat([real[sure_down] < 0, real[sure_up] > 0])),
    }


def _unit(frame: pd.DataFrame) -> pd.Series:
    return frame["day"].astype(str) + " " + frame["ticker"]


# Each skill column and the model it is set against; a missing reference leaves its column empty.
REFERENCES = {"skill": "random-walk", "skill_cal": "random-walk-cal"}


def _mean_loss(losses: pd.DataFrame, units: pd.Index) -> float:
    shared = losses.loc[units]
    return shared["sum"].sum() / shared["count"].sum() if shared["count"].sum() else float("nan")


def scoreboard(forecasts: pd.DataFrame, call: float = 0.05) -> pd.DataFrame:
    """Each skill sets a model against its reference on the ticker-days both were forecast on only:
    models replayed over different days, or a corrected model that starts later, would otherwise be
    set against different days."""
    frame = with_outcome(forecasts)
    frame["unit"] = _unit(frame)
    losses = frame.assign(loss=pinball(frame)).groupby(["model", "bucket", "unit"])["loss"].agg(["sum", "count"])
    rows = []
    for (model, bucket), group in frame.groupby(["model", "bucket"]):
        row = {"model": model, "bucket": bucket, **figures(group, call)}
        mine = losses.loc[(model, bucket)]
        for name, reference in REFERENCES.items():
            if (reference, bucket) not in losses.index.droplevel("unit"):
                row[name] = float("nan")
                continue
            theirs = losses.loc[(reference, bucket)]
            shared = mine.index.intersection(theirs.index)
            row[name] = 1 - _mean_loss(mine, shared) / _mean_loss(theirs, shared)
        rows.append(row)
    board = pd.DataFrame(rows)
    board["bucket"] = pd.Categorical(board["bucket"], BUCKETS, ordered=True)
    return board.sort_values(["bucket", "model"]).reset_index(drop=True)


UNIT_SUMS = ["n", "pinball", "inside", "says", "direction_hit", "down", "down_hit", "up", "up_hit"]


def unit_sums(frame: pd.DataFrame, call: float) -> pd.DataFrame:
    """Per model, bucket and ticker-day, the sums every rate is made of, so a resample only reweighs
    ticker-days."""
    real, median = frame["real"], frame["q50"]
    says = (median != 0) & (real != 0)
    down, up = median <= np.log1p(-call), median >= np.log1p(call)
    parts = pd.DataFrame(
        {
            "model": frame["model"],
            "bucket": frame["bucket"],
            "unit": _unit(frame),
            "n": 1,
            "pinball": pinball(frame),
            "inside": (frame["q10"] <= real) & (real <= frame["q90"]),
            "says": says,
            "direction_hit": says & (np.sign(median) == np.sign(real)),
            "down": down,
            "down_hit": down & (real < 0),
            "up": up,
            "up_hit": up & (real > 0),
        }
    )
    return parts.groupby(["model", "bucket", "unit"])[UNIT_SUMS].sum().astype(float)


def resample_weights(units: int, resamples: int, seed: int) -> np.ndarray:
    """How many times each ticker-day is drawn in each resample, (resamples, units): whole ticker-days
    are drawn, since the cuts of one ticker-day share its fate and are far from independent."""
    draws = np.random.default_rng(seed).integers(0, units, size=(resamples, units))
    return np.stack([np.bincount(row, minlength=units) for row in draws]).astype(float)


def _span(values: np.ndarray, level: float) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def intervals(
    forecasts: pd.DataFrame, call: float = 0.05, resamples: int = 1000, seed: int = 0, level: float = 0.95
) -> pd.DataFrame:
    """Percentile bootstrap over the ticker-days. Every model is resampled on the same draws, so a skill
    interval is paired: a model equal to its reference gets exactly 0."""
    sums = unit_sums(with_outcome(forecasts), call)
    units = sorted(sums.index.get_level_values("unit").unique())
    weights = resample_weights(len(units), resamples, seed)

    def drawn(model: str, bucket: str) -> pd.DataFrame:
        return sums.loc[(model, bucket)].reindex(units, fill_value=0.0)

    def rate(group: pd.DataFrame, hits: str, cases: str) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return (weights @ group[hits].to_numpy()) / (weights @ group[cases].to_numpy())

    pairs = sums.index.droplevel("unit").unique()
    rows = []
    for model, bucket in pairs:
        group = drawn(model, bucket)
        row = {"model": model, "bucket": bucket, "down_days": int((group["down"] > 0).sum()),
               "up_days": int((group["up"] > 0).sum())}
        for name, reference in REFERENCES.items():
            if (reference, bucket) not in pairs:
                row[f"{name}_low"] = row[f"{name}_high"] = float("nan")
                continue
            theirs = drawn(reference, bucket)
            shared = (group["n"] > 0) & (theirs["n"] > 0)
            ratio = rate(group.mul(shared, axis=0), "pinball", "n") / rate(theirs.mul(shared, axis=0), "pinball", "n")
            row[f"{name}_low"], row[f"{name}_high"] = _span(1 - ratio, level)
        for name, hits, cases in [
            ("coverage_80", "inside", "n"),
            ("direction", "direction_hit", "says"),
            ("down_hit", "down_hit", "down"),
            ("up_hit", "up_hit", "up"),
        ]:
            row[f"{name}_low"], row[f"{name}_high"] = _span(rate(group, hits, cases), level)
        rows.append(row)
    board = pd.DataFrame(rows)
    board["bucket"] = pd.Categorical(board["bucket"], BUCKETS, ordered=True)
    return board.sort_values(["bucket", "model"]).reset_index(drop=True)


def by_part(
    forecasts: pd.DataFrame, parts: pd.Series, skill: str = "skill", call: float = 0.05, resamples: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """`skill` and its interval per part of the forecasts (a month, a setup), model and horizon: an edge
    that lives in one part only is a regime, not a model."""
    rows = []
    for name in sorted(parts.dropna().unique()):
        part = forecasts[parts == name]
        board = scoreboard(part, call).merge(intervals(part, call, resamples, seed), on=["model", "bucket"])
        ticker_days = part[["day", "ticker"]].drop_duplicates().shape[0]
        for model, group in board[board["model"] != REFERENCES[skill]].groupby("model"):
            row = {"part": name, "ticker_days": ticker_days, "model": model, **dict.fromkeys(BUCKETS, "–")}
            for _, cell in group.iterrows():
                if np.isnan(cell[skill]):
                    continue
                low, high = cell[f"{skill}_low"], cell[f"{skill}_high"]
                span = "" if np.isnan(low) else f" [{low:+.1%}, {high:+.1%}]"
                row[str(cell["bucket"])] = f"{cell[skill]:+.1%}{span}"
            rows.append(row)
    return pd.DataFrame(rows, columns=["part", "ticker_days", "model", *BUCKETS])


def by_month(forecasts: pd.DataFrame, call: float = 0.05, resamples: int = 1000, seed: int = 0) -> pd.DataFrame:
    month = forecasts["day"].astype(str).str[:7]
    return by_part(forecasts, month, "skill", call, resamples, seed).rename(columns={"part": "month"})


def by_setup(
    forecasts: pd.DataFrame, skill: str = "skill", call: float = 0.05, resamples: int = 1000, seed: int = 0
) -> pd.DataFrame:
    return by_part(forecasts, forecasts["setup"], skill, call, resamples, seed).rename(columns={"part": "setup"})


def quantile_coverage(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Share of real moves at or below each quantile: a calibrated model gives back the level itself."""
    frame = with_outcome(forecasts)
    below = pd.DataFrame(frame["real"].to_numpy()[:, None] <= frame[COLUMNS].to_numpy(), columns=COLUMNS)
    below[["model", "bucket"]] = frame[["model", "bucket"]].to_numpy()
    below["bucket"] = pd.Categorical(below["bucket"], BUCKETS, ordered=True)
    return below.groupby(["model", "bucket"], observed=True)[COLUMNS].mean()


def with_context(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Adds what a cut looked like next to its ticker-day: its band width over the usual one (the median
    over the day's cuts at that horizon), and the move since the previous cut, already seen at the cut."""
    frame = with_outcome(forecasts)
    frame["width"] = frame["q90"] - frame["q10"]
    usual = frame.groupby(["model", "ticker", "day", "bucket"])["width"].transform("median")
    frame["width_ratio"] = (frame["width"] / usual).replace([np.inf, -np.inf], np.nan)
    cuts = frame[["model", "ticker", "day", "cut", "last"]].drop_duplicates(["model", "ticker", "day", "cut"])
    cuts = cuts.sort_values("cut")
    cuts["before"] = np.log(cuts["last"] / cuts.groupby(["model", "ticker", "day"])["last"].shift())
    return frame.merge(cuts.drop(columns="last"), on=["model", "ticker", "day", "cut"], how="left")


def big_figures(group: pd.DataFrame, big: float) -> dict:
    is_big = (group["real"] >= np.log1p(big)) | (group["real"] <= np.log1p(-big))
    moves = group[is_big]
    real, median = moves["real"], moves["q50"]
    leans = median != 0
    quiet = leans & (moves["before"].abs() < np.log1p(big / 2))
    return {
        "cuts": len(group),
        "big": int(is_big.sum()),
        "big_share": _rate(is_big),
        "big_down": _rate(real < 0),
        "lean_hit": _rate(np.sign(median[leans]) == np.sign(real[leans])),
        "quiet": int(quiet.sum()),
        "quiet_hit": _rate(np.sign(median[quiet]) == np.sign(real[quiet])),
        "width_big": float(moves["width_ratio"].median()) if len(moves) else float("nan"),
        "width_rest": float(group.loc[~is_big, "width_ratio"].median()) if (~is_big).any() else float("nan"),
        "coverage_big": _rate((moves["q10"] <= real) & (real <= moves["q90"])),
    }


def big_moves(forecasts: pd.DataFrame, big: float = 0.05) -> pd.DataFrame:
    """The cuts followed by a real move of `big` or more, either way, over +5, +15 or +30 min. They are
    picked on the real move only: what the model said has no say in which cuts are looked at."""
    frame = with_context(forecasts)
    frame = frame[frame["bucket"] != "close"]
    rows = [
        {"model": model, "bucket": bucket, **big_figures(group, big)}
        for (model, bucket), group in frame.groupby(["model", "bucket"])
    ]
    if not rows:
        return pd.DataFrame()
    board = pd.DataFrame(rows)
    board["bucket"] = pd.Categorical(board["bucket"], BUCKETS, ordered=True)
    return board.sort_values(["bucket", "model"]).reset_index(drop=True)


def _table(frame: pd.DataFrame, formats: dict[str, str]) -> str:
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = [head, rule]
    for _, row in frame.iterrows():
        cells = [
            "–" if isinstance(v, float) and np.isnan(v) else formats.get(c, "{}").format(v)
            for c, v in row.items()
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _spans(board: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    board = board.copy()
    for name, fmt in names.items():
        low, high = board.pop(f"{name}_low"), board.pop(f"{name}_high")
        board[f"{name}_95"] = [
            "–" if np.isnan(lo) else f"[{fmt.format(lo)}, {fmt.format(hi)}]" for lo, hi in zip(low, high)
        ]
    return board


def report(
    forecasts: pd.DataFrame, call: float, big: float = 0.05, resamples: int = 1000, seed: int = 0, scope: str = "every cut"
) -> str:
    pct = "{:.1%}"
    spans = intervals(forecasts, call, resamples, seed)
    spans = _spans(
        spans,
        {"skill": "{:+.1%}", "skill_cal": "{:+.1%}", "coverage_80": pct, "direction": pct, "down_hit": pct, "up_hit": pct},
    )
    board = scoreboard(forecasts, call).merge(spans, on=["model", "bucket"])
    days = forecasts["day"].nunique()
    tickers = forecasts[["day", "ticker"]].drop_duplicates().shape[0]
    calibrated = board["skill_cal"].notna().any()
    main = board[
        ["bucket", "model", "n", "pinball", "skill", "skill_95", *(["skill_cal", "skill_cal_95"] if calibrated else []),
         "mae", "mdae", "coverage_80", "coverage_80_95", "direction", "direction_95", "real_down"]
    ]
    months = by_month(forecasts, call, resamples, seed) if forecasts["day"].astype(str).str[:7].nunique() > 1 else None
    setup_skill = "skill_cal" if calibrated else "skill"
    setups = (
        by_setup(forecasts, setup_skill, call, resamples, seed)
        if "setup" in forecasts and forecasts["setup"].nunique() > 1
        else None
    )
    calls = board[
        ["bucket", "model", "down_calls", "down_days", "down_hit", "down_hit_95", "down_real", "up_calls", "up_days",
         "up_hit", "up_hit_95", "up_real", "sure_calls", "sure_hit"]
    ]
    coverage = quantile_coverage(forecasts)[["q10", "q25", "q50", "q75", "q90"]].reset_index()
    moves = big_moves(forecasts, big)
    return "\n\n".join(
        [
            "# Score",
            f"{days} days, {tickers} ticker-days, {len(forecasts)} forecasts, on {scope}. "
            "Moves are log returns from the last price at the cut, the scale the model forecasts in; "
            "`mae` and the calls' `real` read as %. "
            f"`*_95`: 95 % interval over {resamples} resamples of the ticker-days (seed {seed}), the unit the "
            "forecasts are independent at; a `skill_95` wholly above 0 beats the random walk.",
            "## Accuracy",
            "`pinball`: mean quantile loss on log moves, lower is better. `skill`: 1 − pinball / random walk's, above 0 "
            "beats it" + (
                "; `skill_cal` the same against the calibrated random walk (`ts-calibrate`), the fair bar for a "
                "calibrated model" if calibrated else ""
            ) + ". Each on the ticker-days the model and its reference share. `mae` / `mdae`: mean / median |median − real|, the median one robust to a few wild names. `coverage_80`: real inside the 10–90 % band, 80 % expected. "
            "`direction`: median and real move of the same sign, flat moves left out. `real_down`: share of "
            "real moves down, the bar a direction call has to clear.",
            _table(
                main,
                {"pinball": "{:.4f}", "skill": "{:+.1%}", "skill_cal": "{:+.1%}", "mae": pct, "mdae": pct, "coverage_80": pct, "direction": pct, "real_down": pct},
            ),
            "## Calls",
            f"A call: the median says {call:.0%} down or up or more (`down_*`, `up_*`), or the band sits "
            "wholly on one side of 0, 90 % sure (`sure_*`). `hit`: the real move went that way. "
            "`real`: mean real move on those calls. `days`: the ticker-days the calls came from.",
            _table(calls, {"down_hit": pct, "down_real": "{:+.1%}", "up_hit": pct, "up_real": "{:+.1%}", "sure_hit": pct}),
            "## Calibration",
            "Share of real moves at or below each quantile: q10 should read 10 %, q90 90 %.",
            _table(coverage, {c: pct for c in ["q10", "q25", "q50", "q75", "q90"]}),
            "## Big moves",
            f"The cuts followed by a real move of {big:.0%} or more, either way, picked on the real move only. "
            "`big_share`: their share of the cuts. `big_down`: share of them down. `lean_hit`: the median leaned "
            "the way the price went, 50 % is a coin flip. `quiet_hit`: the same on the cuts where the price had "
            f"moved less than {big / 2:.1%} since the previous cut, so the move had not started yet. "
            "`width_big` / `width_rest`: the 10–90 % band width over the ticker-day's usual one, before a big "
            "move and before the others; above 1 is wider than usual. `coverage_big`: the real price inside the band.",
            _table(
                moves,
                {"big_share": pct, "big_down": pct, "lean_hit": pct, "quiet_hit": pct, "width_big": "{:.2f}",
                 "width_rest": "{:.2f}", "coverage_big": pct},
            ) if not moves.empty else "No forecast at +5, +15 or +30 min.",
            *(
                [
                    "## By month",
                    "`skill` against the random walk per month, with its 95 % interval over that month's ticker-days.",
                    _table(months, {}),
                ]
                if months is not None
                else []
            ),
            *(
                [
                    "## By setup",
                    f"`{setup_skill}` per setup the ticker entered on, with its 95 % interval over that setup's "
                    "ticker-days: `gus` gapped up in the premarket and enters at the open, `push` enters during the "
                    "session, the minute it reached its push.",
                    _table(setups, {}),
                ]
                if setups is not None
                else []
            ),
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the stored forecasts against what happened.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--call", type=float, default=0.05, help="median move that makes a call")
    parser.add_argument("--big", type=float, default=0.05, help="real move that makes a big move")
    parser.add_argument("--start", type=date.fromisoformat, help="first day scored")
    parser.add_argument("--end", type=date.fromisoformat, help="last day scored")
    parser.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples of the ticker-days")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--all-cuts", action="store_true", help="keep the cuts before the ticker reached its gap, which read the future"
    )
    args = parser.parse_args(argv)

    forecasts = load_forecasts(args.data)
    if forecasts.empty:
        parser.exit(1, "error: no forecasts in data/forecasts, run ts-replay first\n")
    day = forecasts["day"].astype(str)
    start, end = args.start or date.min, args.end or date.max
    forecasts = forecasts[(day >= start.isoformat()) & (day <= end.isoformat())]
    if forecasts.empty:
        parser.exit(1, "error: no forecasts between --start and --end\n")
    scope = "every cut"
    entry = entries(Store(args.data))
    if not args.all_cuts:
        kept = after_entry(forecasts, entry)
        scope = f"the cuts after the ticker reached its gap, {len(kept)} of {len(forecasts)}"
        forecasts = kept
        if forecasts.empty:
            parser.exit(1, "error: no cut after a ticker reached its gap, --all-cuts to keep them all\n")
    forecasts = with_setup(forecasts, entry)
    text = report(forecasts, args.call, args.big, args.resamples, args.seed, scope)
    path = args.data / "reports" / "score.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    print(text)
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()

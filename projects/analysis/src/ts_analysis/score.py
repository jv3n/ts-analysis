import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ts_analysis.replay import QUANTILES

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


def scoreboard(forecasts: pd.DataFrame, call: float = 0.05) -> pd.DataFrame:
    frame = with_outcome(forecasts)
    rows = [
        {"model": model, "bucket": bucket, **figures(group, call)}
        for (model, bucket), group in frame.groupby(["model", "bucket"])
    ]
    board = pd.DataFrame(rows)
    reference = board[board["model"] == "random-walk"].set_index("bucket")["pinball"]
    board["skill"] = 1 - board["pinball"] / board["bucket"].map(reference)
    board["bucket"] = pd.Categorical(board["bucket"], BUCKETS, ordered=True)
    return board.sort_values(["bucket", "model"]).reset_index(drop=True)


def quantile_coverage(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Share of real moves at or below each quantile: a calibrated model gives back the level itself."""
    frame = with_outcome(forecasts)
    below = pd.DataFrame(frame["real"].to_numpy()[:, None] <= frame[COLUMNS].to_numpy(), columns=COLUMNS)
    below[["model", "bucket"]] = frame[["model", "bucket"]].to_numpy()
    below["bucket"] = pd.Categorical(below["bucket"], BUCKETS, ordered=True)
    return below.groupby(["model", "bucket"], observed=True)[COLUMNS].mean()


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


def report(forecasts: pd.DataFrame, call: float) -> str:
    board = scoreboard(forecasts, call)
    days = forecasts["day"].nunique()
    tickers = forecasts[["day", "ticker"]].drop_duplicates().shape[0]
    pct = "{:.1%}"
    main = board[["bucket", "model", "n", "pinball", "skill", "mae", "mdae", "coverage_80", "direction", "real_down"]]
    calls = board[
        ["bucket", "model", "down_calls", "down_hit", "down_real", "up_calls", "up_hit", "up_real", "sure_calls", "sure_hit"]
    ]
    coverage = quantile_coverage(forecasts)[["q10", "q25", "q50", "q75", "q90"]].reset_index()
    return "\n\n".join(
        [
            "# Score",
            f"{days} days, {tickers} ticker-days, {len(forecasts)} forecasts. "
            "Moves are log returns from the last price at the cut, the scale the model forecasts in; "
            "`mae` and the calls' `real` read as %.",
            "## Accuracy",
            "`pinball`: mean quantile loss on log moves, lower is better. `skill`: 1 − pinball / random walk's, above 0 "
            "beats it. `mae` / `mdae`: mean / median |median − real|, the median one robust to a few wild names. `coverage_80`: real inside the 10–90 % band, 80 % expected. "
            "`direction`: median and real move of the same sign, flat moves left out. `real_down`: share of "
            "real moves down, the bar a direction call has to clear.",
            _table(
                main,
                {"pinball": "{:.4f}", "skill": "{:+.1%}", "mae": pct, "mdae": pct, "coverage_80": pct, "direction": pct, "real_down": pct},
            ),
            "## Calls",
            f"A call: the median says {call:.0%} down or up or more (`down_*`, `up_*`), or the band sits "
            "wholly on one side of 0, 90 % sure (`sure_*`). `hit`: the real move went that way. "
            "`real`: mean real move on those calls.",
            _table(calls, {"down_hit": pct, "down_real": "{:+.1%}", "up_hit": pct, "up_real": "{:+.1%}", "sure_hit": pct}),
            "## Calibration",
            "Share of real moves at or below each quantile: q10 should read 10 %, q90 90 %.",
            _table(coverage, {c: pct for c in ["q10", "q25", "q50", "q75", "q90"]}),
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the stored forecasts against what happened.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--call", type=float, default=0.05, help="median move that makes a call")
    args = parser.parse_args(argv)

    forecasts = load_forecasts(args.data)
    if forecasts.empty:
        parser.exit(1, "error: no forecasts in data/forecasts, run ts-replay first\n")
    text = report(forecasts, args.call)
    path = args.data / "reports" / "score.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    print(text)
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()

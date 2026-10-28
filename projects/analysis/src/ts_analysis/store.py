import json
from datetime import date
from pathlib import Path

import pandas as pd

BAR_COLUMNS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "trades"}


def bars_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["t", *BAR_COLUMNS]).astype({key: "float64" for key in BAR_COLUMNS})
    frame.insert(0, "time", pd.to_datetime(frame.pop("t").astype("int64"), unit="ms", utc=True))
    return frame.rename(columns=BAR_COLUMNS)


class Store:
    def __init__(self, root: Path):
        self.root = root

    def _day_json(self, kind: str, day: date) -> Path:
        return self.root / kind / f"{day.isoformat()}.json"

    def bars_path(self, day: date, ticker: str) -> Path:
        return self.root / "bars" / day.isoformat() / f"{ticker}.parquet"

    def load_grouped(self, day: date) -> list[dict] | None:
        return _read_json(self._day_json("grouped", day))

    def save_grouped(self, day: date, rows: list[dict]) -> None:
        _write_json(self._day_json("grouped", day), rows)

    def load_splits(self, day: date) -> list[dict] | None:
        return _read_json(self._day_json("splits", day))

    def save_splits(self, day: date, rows: list[dict]) -> None:
        _write_json(self._day_json("splits", day), rows)

    def load_picks(self, day: date) -> dict | None:
        return _read_json(self._day_json("picks", day))

    def save_picks(self, day: date, picks: dict) -> None:
        _write_json(self._day_json("picks", day), picks)

    def has_bars(self, day: date, ticker: str) -> bool:
        return self.bars_path(day, ticker).exists()

    def save_bars(self, day: date, ticker: str, rows: list[dict]) -> None:
        path = self.bars_path(day, ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        bars_frame(rows).to_parquet(tmp, index=False)
        tmp.replace(path)

    def load_bars(self, day: date, ticker: str) -> pd.DataFrame:
        return pd.read_parquet(self.bars_path(day, ticker))

    def forecasts_path(self, model: str, day: date) -> Path:
        return self.root / "forecasts" / model / f"{day.isoformat()}.parquet"

    def has_forecasts(self, model: str, day: date) -> bool:
        return self.forecasts_path(model, day).exists()

    def save_forecasts(self, model: str, day: date, frame: pd.DataFrame) -> None:
        path = self.forecasts_path(model, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)

    def load_forecasts(self, model: str, day: date) -> pd.DataFrame:
        return pd.read_parquet(self.forecasts_path(model, day))


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)

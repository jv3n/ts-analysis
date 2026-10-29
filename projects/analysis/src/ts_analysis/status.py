import json
import os
from datetime import UTC, datetime
from pathlib import Path


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Status:
    """The progress of a running job in `<data>/status/<job>.json`, rewritten whole on every update so
    a reader never sees half of it."""

    def __init__(self, root: Path, job: str, **info):
        self.path = root / "status" / f"{job}.json"
        self.state = {"job": job, "pid": os.getpid(), "state": "running", "started": now(), **info}
        self._write()

    def update(self, **fields) -> None:
        self.state.update(fields)
        self._write()

    def __enter__(self) -> "Status":
        return self

    def __exit__(self, kind, error, _traceback) -> bool:
        if kind is None:
            self.update(state="done", finished=now())
        elif issubclass(kind, KeyboardInterrupt):
            self.update(state="stopped", finished=now())
        else:
            self.update(state="failed", finished=now(), error=f"{kind.__name__}: {error}")
        return False

    def _write(self) -> None:
        self.state["updated"] = now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_status(root: Path, job: str) -> dict | None:
    path = root / "status" / f"{job}.json"
    if not path.exists():
        return None
    status = json.loads(path.read_text())
    # A job killed hard (SIGKILL, closed terminal) never writes its end: its pid tells.
    if status["state"] == "running" and not is_alive(status["pid"]):
        status["state"] = "lost"
    return status

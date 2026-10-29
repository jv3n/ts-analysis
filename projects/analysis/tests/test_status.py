import json
import os

import pytest
from test_fetch import MON, TUE, FakeSource

from ts_analysis.fetch import backfill
from ts_analysis.selection import GapRule
from ts_analysis.status import Status, load_status
from ts_analysis.store import Store


def test_a_running_job_reports_where_it_is(tmp_path):
    with Status(tmp_path, "fetch", days=2) as status:
        status.update(day="2026-10-26", day_number=1)
        running = load_status(tmp_path, "fetch")
    assert (running["state"], running["day"], running["day_number"], running["days"]) == ("running", "2026-10-26", 1, 2)


def test_a_job_that_ends_normally_is_done(tmp_path):
    with Status(tmp_path, "replay"):
        pass
    assert load_status(tmp_path, "replay")["state"] == "done"


def test_a_job_stopped_with_ctrl_c_is_stopped_not_failed(tmp_path):
    with pytest.raises(KeyboardInterrupt), Status(tmp_path, "fetch"):
        raise KeyboardInterrupt
    assert load_status(tmp_path, "fetch")["state"] == "stopped"


def test_a_job_that_crashes_is_failed_with_its_error(tmp_path):
    with pytest.raises(ValueError), Status(tmp_path, "replay"):
        raise ValueError("Unexpected quantile levels")
    status = load_status(tmp_path, "replay")
    assert (status["state"], status["error"]) == ("failed", "ValueError: Unexpected quantile levels")


def test_a_job_killed_without_writing_its_end_is_lost(tmp_path):
    Status(tmp_path, "fetch")
    path = tmp_path / "status" / "fetch.json"
    status = json.loads(path.read_text())
    path.write_text(json.dumps(status | {"pid": 2**22 + os.getpid()}))
    assert load_status(tmp_path, "fetch")["state"] == "lost"


def test_a_job_never_run_has_no_status(tmp_path):
    assert load_status(tmp_path, "fetch") is None


def test_the_backfill_reports_its_last_day_and_ticker(tmp_path):
    backfill(FakeSource(), Store(tmp_path), MON, MON, GapRule(), per_day=10, seed="s")
    status = load_status(tmp_path, "fetch")
    assert (status["state"], status["day"], status["ticker"], status["ticker_number"]) == ("done", "2026-10-26", "FFIE", 1)


def test_a_backfill_stopped_by_an_unpublished_day_says_so(tmp_path):
    backfill(FakeSource(unpublished=frozenset({TUE})), Store(tmp_path), MON, TUE, GapRule(), per_day=10, seed="s")
    status = load_status(tmp_path, "fetch")
    assert (status["state"], status["note"]) == ("done", "2026-10-27 not published yet")

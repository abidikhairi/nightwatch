import os
import signal
import sqlite3
import time
from pathlib import Path

from nightwatch.db import init_db
from nightwatch.process import launch_vllm_serve


def test_launch_vllm_serve_records_process(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    init_db(db_path)

    pid = launch_vllm_serve(
        args=["sleep", "5"],
        repo_id="fake/repo",
        command_display="vllm serve fake/repo --tensor-parallel-size 1",
        db_path=db_path,
    )

    try:
        os.kill(pid, 0)  # raises if the process isn't alive

        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                "SELECT pid, repo_id, command, started_by, status FROM vllm_processes"
            ).fetchone()

        assert row[0] == pid
        assert row[1] == "fake/repo"
        assert row[2] == "vllm serve fake/repo --tensor-parallel-size 1"
        assert row[3]
        assert row[4] == "running"
    finally:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.2)

import sqlite3
import subprocess
import time
from pathlib import Path

from nightwatch.db import init_db, insert_process, prune_dead_processes


def test_init_db_creates_file_and_table(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "db.sqlite3"

    init_db(db_path)

    assert db_path.exists()
    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]: row[2]
            for row in connection.execute("PRAGMA table_info(vllm_processes)")
        }
    assert columns == {
        "id": "INTEGER",
        "pid": "INTEGER",
        "repo_id": "TEXT",
        "command": "TEXT",
        "started_at": "TEXT",
        "started_by": "TEXT",
        "status": "TEXT",
    }


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"

    init_db(db_path)
    init_db(db_path)

    assert db_path.exists()


def test_prune_dead_processes_removes_only_dead_pids(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    init_db(db_path)

    alive_process = subprocess.Popen(["sleep", "30"])
    try:
        insert_process(
            pid=alive_process.pid,
            repo_id="real/alive",
            command="vllm serve real/alive",
            started_by="tester",
            db_path=db_path,
        )

        dead_process = subprocess.Popen(["true"])
        dead_process.wait()
        insert_process(
            pid=dead_process.pid,
            repo_id="fake/dead",
            command="vllm serve fake/dead",
            started_by="tester",
            db_path=db_path,
        )

        removed = prune_dead_processes(db_path)

        assert removed == [dead_process.pid]
        with sqlite3.connect(db_path) as connection:
            remaining = connection.execute("SELECT pid, repo_id FROM vllm_processes").fetchall()
        assert remaining == [(alive_process.pid, "real/alive")]
    finally:
        alive_process.terminate()
        alive_process.wait()


def test_prune_dead_processes_removes_zombie_pid(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite3"
    init_db(db_path)

    # Deliberately never reaped (no .wait()/.poll()), so it lingers as a
    # zombie: os.kill(pid, 0) still succeeds even though the process is gone.
    zombie_process = subprocess.Popen(["true"])
    time.sleep(0.5)

    insert_process(
        pid=zombie_process.pid,
        repo_id="fake/zombie",
        command="vllm serve fake/zombie",
        started_by="tester",
        db_path=db_path,
    )

    removed = prune_dead_processes(db_path)

    assert removed == [zombie_process.pid]
    with sqlite3.connect(db_path) as connection:
        remaining = connection.execute("SELECT pid FROM vllm_processes").fetchall()
    assert remaining == []

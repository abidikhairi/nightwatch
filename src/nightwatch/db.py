import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB_DIR = Path.home() / ".nightwatch"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "db.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vllm_processes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pid INTEGER NOT NULL,
    repo_id TEXT NOT NULL,
    command TEXT NOT NULL,
    started_at TEXT NOT NULL,
    started_by TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
)
"""


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(_SCHEMA)


def insert_process(
    pid: int,
    repo_id: str,
    command: str,
    started_by: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    started_at = datetime.now(UTC).isoformat()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO vllm_processes (pid, repo_id, command, started_at, started_by)
            VALUES (?, ?, ?, ?, ?)
            """,
            (pid, repo_id, command, started_at, started_by),
        )


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user; treat as alive.
        return True

    try:
        with open(f"/proc/{pid}/status") as status_file:
            for line in status_file:
                if line.startswith("State:"):
                    # A zombie still holds a pid (os.kill succeeds) but has
                    # already exited and released its resources.
                    return "zombie" not in line.lower()
    except FileNotFoundError:
        return False
    return True


def prune_dead_processes(db_path: Path = DEFAULT_DB_PATH) -> list[int]:
    removed_pids: list[int] = []
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("SELECT id, pid FROM vllm_processes").fetchall()
        for row_id, pid in rows:
            if not _is_process_alive(pid):
                connection.execute("DELETE FROM vllm_processes WHERE id = ?", (row_id,))
                removed_pids.append(pid)
    return removed_pids


@dataclass
class ProcessRecord:
    id: int
    pid: int
    repo_id: str
    command: str
    started_at: str
    started_by: str
    status: str


def list_processes(db_path: Path = DEFAULT_DB_PATH) -> list[ProcessRecord]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, pid, repo_id, command, started_at, started_by, status
            FROM vllm_processes
            ORDER BY started_at DESC
            """
        ).fetchall()
    return [ProcessRecord(*row) for row in rows]

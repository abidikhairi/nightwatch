import subprocess
from getpass import getuser
from pathlib import Path

from nightwatch.db import DEFAULT_DB_PATH, insert_process


def launch_vllm_serve(
    args: list[str],
    repo_id: str,
    command_display: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    process = subprocess.Popen(
        args,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    insert_process(
        pid=process.pid,
        repo_id=repo_id,
        command=command_display,
        started_by=getuser(),
        db_path=db_path,
    )
    return process.pid


def get_process_memory_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return None

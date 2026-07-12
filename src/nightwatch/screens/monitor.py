import asyncio
from typing import TYPE_CHECKING, cast

from textual import work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header

from nightwatch.db import list_processes, prune_dead_processes

if TYPE_CHECKING:
    from nightwatch.app import NightwatchApp


class MonitorScreen(Screen):
    TITLE = "Monitoring vLLM"

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("r", "refresh_processes", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="processes-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("PID", "Repo ID", "Command", "Started At", "Started By", "Status")
        self.action_refresh_processes()

    def action_refresh_processes(self) -> None:
        self._load_processes()

    @work(exclusive=True, group="load-processes")
    async def _load_processes(self) -> None:
        app = cast("NightwatchApp", self.app)
        db_path = app.db_path

        await asyncio.to_thread(prune_dead_processes, db_path)
        processes = await asyncio.to_thread(list_processes, db_path)

        table = self.query_one(DataTable)
        table.clear()
        for process in processes:
            table.add_row(
                str(process.pid),
                process.repo_id,
                process.command,
                process.started_at,
                process.started_by,
                process.status,
                key=str(process.id),
            )

import asyncio
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, Static

from nightwatch.db import DEFAULT_DB_PATH, init_db, prune_dead_processes
from nightwatch.screens import ExploreScreen, MonitorScreen, ServeModelScreen

_BANNER = """\
███╗   ██╗██╗ ██████╗ ██╗  ██╗████████╗
████╗  ██║██║██╔════╝ ██║  ██║╚══██╔══╝
██╔██╗ ██║██║██║  ███╗███████║   ██║
██║╚██╗██║██║██║   ██║██╔══██║   ██║
██║ ╚████║██║╚██████╔╝██║  ██║   ██║
╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝

██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
██║ █╗ ██║███████║   ██║   ██║     ███████║
██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
 ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝\
"""

_TAGLINE = "Interactive TUI for serving models with vLLM"
_HINTS = "[b]s[/b] Serve   [b]e[/b] Explore   [b]m[/b] Monitor   [b]q[/b] Quit"


class NightwatchApp(App):
    """Interactive TUI for serving models with vLLM."""

    TITLE = "nightwatch"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "serve_model", "Serve Model"),
        ("e", "explore", "Explore Models"),
        ("m", "monitor", "Monitor vLLM")
    ]

    CSS = """
    #welcome {
        align: center middle;
        height: 1fr;
    }

    #banner {
        text-align: center;
        color: $accent;
        width: auto;
    }

    #tagline {
        text-align: center;
        width: auto;
        margin-top: 1;
    }

    #hints {
        text-align: center;
        width: auto;
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        super().__init__()
        self.db_path = db_path

    def on_mount(self) -> None:
        init_db(self.db_path)
        self._prune_dead_processes()

    @work(exclusive=True, group="prune-processes")
    async def _prune_dead_processes(self) -> None:
        await asyncio.to_thread(prune_dead_processes, self.db_path)

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="welcome"):
            yield Static(_BANNER, id="banner")
            yield Static(_TAGLINE, id="tagline")
            yield Static(_HINTS, id="hints")
        yield Footer()

    def action_serve_model(self) -> None:
        self.push_screen(ServeModelScreen())

    def action_explore(self) -> None:
        self.push_screen(ExploreScreen())

    def action_monitor(self) -> None:
        self.push_screen(MonitorScreen())

def run() -> None:
    NightwatchApp().run()

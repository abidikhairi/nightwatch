from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static

from nightwatch.screens import ExploreScreen, MonitorScreen, ServeModelScreen


class NightwatchApp(App):
    """Interactive TUI for serving models with vLLM."""

    TITLE = "nightwatch"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "serve_model", "Serve Model"),
        ("e", "explore", "Explore Models"),
        ("m", "monitor", "Monitor vLLM")
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("nightwatch is warming up...")
        yield Footer()

    def action_serve_model(self) -> None:
        self.push_screen(ServeModelScreen())

    def action_explore(self) -> None:
        self.push_screen(ExploreScreen())

    def action_monitor(self) -> None:
        self.push_screen(MonitorScreen())

def run() -> None:
    NightwatchApp().run()

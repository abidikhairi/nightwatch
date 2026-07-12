from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static


class NightwatchApp(App):
    """Interactive TUI for serving models with vLLM."""

    TITLE = "nightwatch"
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("nightwatch is warming up...")
        yield Footer()


def run() -> None:
    NightwatchApp().run()

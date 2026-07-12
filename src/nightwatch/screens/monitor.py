from textual.app import ComposeResult, Screen
from textual.widgets import Footer, Header, Static


class MonitorScreen(Screen):
    TITLE = "Monitoring vLLM"
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Monitoring models ...")
        yield Footer()
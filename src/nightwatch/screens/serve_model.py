from textual.app import ComposeResult, Screen
from textual.widgets import Footer, Header, Static


class ServeModelScreen(Screen):
    TITLE = "Serve new LLM"
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Serving new model ...")
        yield Footer()
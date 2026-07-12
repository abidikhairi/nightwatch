from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }

    ConfirmScreen #dialog {
        width: auto;
        max-width: 80%;
        height: auto;
        border: thick $primary;
        padding: 1 2;
        background: $surface;
    }

    ConfirmScreen #question {
        margin-bottom: 1;
    }

    ConfirmScreen #button-row {
        height: auto;
        align: center middle;
    }

    ConfirmScreen Button {
        margin: 0 1;
    }
    """

    def __init__(self, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self._question, id="question")
            with Horizontal(id="button-row"):
                yield Button("Yes", id="yes", variant="success")
                yield Button("No", id="no", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

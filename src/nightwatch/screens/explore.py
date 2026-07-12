import asyncio

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Footer, Header, Input, Markdown, OptionList
from textual.widgets.option_list import Option

from nightwatch.hf import (
    ModelSummary,
    format_details_markdown,
    get_model_details,
    recommend_deployment,
    search_models,
)

_SEARCH_DEBOUNCE_SECONDS = 0.35
_SEARCH_RESULT_LIMIT = 20

_INITIAL_MESSAGE = "Type to search Hugging Face models."
_LOADING_MESSAGE = "Loading details..."


class ExploreScreen(Screen):
    TITLE = "Explore Models"

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    DEFAULT_CSS = """
    ExploreScreen #search-pane {
        width: 35%;
    }

    ExploreScreen #results {
        height: 1fr;
    }

    ExploreScreen #details {
        width: 1fr;
        padding: 0 1;
    }
    """

    results: reactive[list[ModelSummary]] = reactive(list)
    selected_repo_id: reactive[str | None] = reactive(None)
    details_markdown: reactive[str] = reactive(_INITIAL_MESSAGE)

    def __init__(self) -> None:
        super().__init__()
        self._debounce_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="search-pane"):
                yield Input(placeholder="Search Hugging Face models...", id="search-box")
                yield OptionList(id="results")
            yield Markdown(_INITIAL_MESSAGE, id="details")
        yield Footer()

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
        query = event.value
        self._debounce_timer = self.set_timer(
            _SEARCH_DEBOUNCE_SECONDS, lambda: self._search(query)
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is not None:
            self.selected_repo_id = event.option_id

    def watch_results(self, results: list[ModelSummary]) -> None:
        option_list = self.query_one("#results", OptionList)
        option_list.clear_options()
        for model in results:
            label = f"{model.repo_id}\n[dim]{model.downloads:,} downloads[/dim]"
            option_list.add_option(Option(label, id=model.repo_id))

    def watch_selected_repo_id(self, repo_id: str | None) -> None:
        if repo_id is not None:
            self.details_markdown = _LOADING_MESSAGE
            self._fetch_details(repo_id)

    def watch_details_markdown(self, markdown: str) -> None:
        self.query_one("#details", Markdown).update(markdown)

    @work(exclusive=True, group="search")
    async def _search(self, query: str) -> None:
        if not query.strip():
            self.results = []
            return
        try:
            self.results = await asyncio.to_thread(search_models, query, _SEARCH_RESULT_LIMIT)
        except Exception as exc:
            self.details_markdown = f"Search failed: {exc}"

    @work(exclusive=True, group="details")
    async def _fetch_details(self, repo_id: str) -> None:
        try:
            details = await asyncio.to_thread(get_model_details, repo_id)
            recommendation = recommend_deployment(details, num_users=10)
            self.details_markdown = format_details_markdown(details, recommendation)
        except Exception as exc:
            self.details_markdown = f"Failed to load details for {repo_id}: {exc}"

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
    ModelDetails,
    ModelSummary,
    Recommendation,
    get_model_details,
    recommend_deployment,
    search_models,
)

_SEARCH_DEBOUNCE_SECONDS = 0.35
_SEARCH_RESULT_LIMIT = 20

_INITIAL_MESSAGE = "Type to search Hugging Face models."
_LOADING_MESSAGE = "Loading details..."


def _format_bytes(num_bytes: int) -> str:
    gb = num_bytes / 1_000_000_000
    return f"{gb:.2f} GB"


def _format_details_markdown(details: ModelDetails, recommendation: Recommendation) -> str:
    params = f"{details.num_params:,}" if details.num_params is not None else "unknown"
    dtype = details.dtype or "unknown"
    context = f"{details.max_context_length:,} tokens" if details.max_context_length else "unknown"
    chat_template = "yes" if details.has_chat_template else "no"
    tool_support = "yes" if details.supports_tools else "no"

    if details.quantization_config:
        method = details.quantization_config.get("quant_method", "unknown")
        bits = details.quantization_config.get("bits", "?")
        quantization = f"{method}, {bits}-bit"
    else:
        quantization = "none (full precision)"

    lines = [
        f"# {details.repo_id}",
        "",
        f"- **Parameters**: {params}",
        f"- **Size on disk**: {_format_bytes(details.size_on_disk_bytes)}",
        f"- **Dtype**: {dtype}",
        f"- **Max context length**: {context}",
        f"- **Chat template**: {chat_template}",
        f"- **Tool support**: {tool_support}",
        f"- **Quantization**: {quantization}",
        "",
        f"## Recommendation ({recommendation.num_users} concurrent users)",
        "",
        f"- **Estimated memory**: {recommendation.estimated_memory_gb} GB",
        f"- **GPU**: {recommendation.gpu_recommendation}",
        f"- **Tensor parallel size**: {recommendation.tensor_parallel_size}",
        f"- **GPU memory utilization**: {recommendation.gpu_memory_utilization}",
        f"- **Max concurrent sequences**: {recommendation.max_num_seqs}",
    ]
    if recommendation.quantization_advice:
        lines.append(f"- **Advice**: {recommendation.quantization_advice}")

    lines += [
        "",
        "```bash",
        recommendation.serve_command,
        "```",
    ]

    return "\n".join(lines)


class ExploreScreen(Screen):
    TITLE = "Explore Models"

    BINDINGS = [("escape", "pop_screen", "Back")]

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
            self.details_markdown = _format_details_markdown(details, recommendation)
        except Exception as exc:
            self.details_markdown = f"Failed to load details for {repo_id}: {exc}"

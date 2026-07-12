import asyncio
from typing import TYPE_CHECKING, cast

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    OptionList,
    Select,
    Static,
)
from textual.widgets.option_list import Option

from nightwatch.hf import (
    ModelSummary,
    Recommendation,
    build_serve_args,
    build_serve_command,
    format_model_facts_markdown,
    get_model_details,
    recommend_deployment,
    search_models,
)
from nightwatch.process import launch_vllm_serve
from nightwatch.screens.confirm import ConfirmScreen

if TYPE_CHECKING:
    from nightwatch.app import NightwatchApp

_SEARCH_DEBOUNCE_SECONDS = 0.35
_SEARCH_RESULT_LIMIT = 20
_DEFAULT_NUM_USERS = 10
_DTYPE_OPTIONS = ("bfloat16", "float16", "float32", "auto")

_INITIAL_MESSAGE = "Search and select a model to build a `vllm serve` command."
_LOADING_MESSAGE = "Fetching model info..."


class ServeModelScreen(Screen):
    TITLE = "Serve new LLM"

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    DEFAULT_CSS = """
    ServeModelScreen #search-pane {
        width: 35%;
    }

    ServeModelScreen #results {
        height: 1fr;
    }

    ServeModelScreen #num-users-box {
        height: auto;
    }

    ServeModelScreen #detail-pane {
        width: 1fr;
        padding: 0 1;
    }

    ServeModelScreen #model-info {
        height: auto;
    }

    ServeModelScreen #params-form {
        height: auto;
    }

    ServeModelScreen #params-form Label {
        margin-top: 1;
        height: 1;
    }

    ServeModelScreen #params-form Input {
        height: 1;
        border: none;
    }

    ServeModelScreen #params-form Select {
        height: 1;
        border: none;
    }

    ServeModelScreen #run-button {
        margin-top: 1;
    }
    """

    results: reactive[list[ModelSummary]] = reactive(list)
    selected_repo_id: reactive[str | None] = reactive(None)
    model_info_markdown: reactive[str] = reactive(_INITIAL_MESSAGE)

    def __init__(self) -> None:
        super().__init__()
        self._debounce_timer: Timer | None = None
        self._quantization_flag: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="search-pane"):
                yield Input(placeholder="Search Hugging Face models...", id="search-box")
                yield OptionList(id="results")
                yield Input(
                    placeholder=f"Concurrent users (default {_DEFAULT_NUM_USERS})",
                    id="num-users-box",
                )
            with VerticalScroll(id="detail-pane"):
                yield Markdown(_INITIAL_MESSAGE, id="model-info")
                yield Vertical(id="params-form")
        yield Footer()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search-box":
            return
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
        query = event.value
        self._debounce_timer = self.set_timer(
            _SEARCH_DEBOUNCE_SECONDS, lambda: self._search(query)
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "num-users-box" and self.selected_repo_id is not None:
            self._rebuild_recommendation()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is not None:
            self.selected_repo_id = event.option_id

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-button":
            self._handle_run_button()

    def watch_results(self, results: list[ModelSummary]) -> None:
        option_list = self.query_one("#results", OptionList)
        option_list.clear_options()
        for model in results:
            label = f"{model.repo_id}\n[dim]{model.downloads:,} downloads[/dim]"
            option_list.add_option(Option(label, id=model.repo_id))

    def watch_selected_repo_id(self, repo_id: str | None) -> None:
        if repo_id is not None:
            self._rebuild_recommendation()

    def watch_model_info_markdown(self, markdown: str) -> None:
        self.query_one("#model-info", Markdown).update(markdown)

    def _read_num_users(self) -> int | None:
        num_users_raw = self.query_one("#num-users-box", Input).value.strip()
        if not num_users_raw:
            return _DEFAULT_NUM_USERS
        if num_users_raw.isdigit() and int(num_users_raw) > 0:
            return int(num_users_raw)
        self.model_info_markdown = (
            f"Invalid concurrent users value: `{num_users_raw}`. Enter a positive integer."
        )
        return None

    def _rebuild_recommendation(self) -> None:
        if self.selected_repo_id is None:
            return
        num_users = self._read_num_users()
        if num_users is None:
            return
        self.model_info_markdown = _LOADING_MESSAGE
        self._fetch_details(self.selected_repo_id, num_users)

    async def _populate_params_form(self, recommendation: Recommendation) -> None:
        form = self.query_one("#params-form", Vertical)
        await form.remove_children()
        await form.mount_all(
            [
                Label("Tensor parallel size"),
                Input(value=str(recommendation.tensor_parallel_size), id="tp-input"),
                Label("GPU memory utilization"),
                Input(value=str(recommendation.gpu_memory_utilization), id="gpu-util-input"),
                Label("Max concurrent sequences"),
                Input(value=str(recommendation.max_num_seqs), id="max-seqs-input"),
                Label("Max model length"),
                Input(value=str(recommendation.max_model_len or ""), id="max-len-input"),
                Label("Dtype"),
                Select(
                    [(dtype, dtype) for dtype in _DTYPE_OPTIONS],
                    value=recommendation.dtype_flag,
                    id="dtype-select",
                ),
                Button("Run this model", id="run-button", variant="primary"),
                Static("", id="run-status"),
            ]
        )

    def _set_run_status(self, message: str) -> None:
        self.query_one("#run-status", Static).update(message)

    @work(exclusive=True, group="search")
    async def _search(self, query: str) -> None:
        if not query.strip():
            self.results = []
            return
        try:
            self.results = await asyncio.to_thread(search_models, query, _SEARCH_RESULT_LIMIT)
        except Exception as exc:
            self.model_info_markdown = f"Search failed: {exc}"

    @work(exclusive=True, group="details")
    async def _fetch_details(self, repo_id: str, num_users: int) -> None:
        try:
            details = await asyncio.to_thread(get_model_details, repo_id)
            recommendation = recommend_deployment(details, num_users=num_users)
            self._quantization_flag = recommendation.quantization_flag
            self.model_info_markdown = format_model_facts_markdown(details)
            await self._populate_params_form(recommendation)
        except Exception as exc:
            self.model_info_markdown = f"Failed to load details for {repo_id}: {exc}"

    def _handle_run_button(self) -> None:
        repo_id = self.selected_repo_id
        if repo_id is None:
            return

        try:
            tensor_parallel_size = int(self.query_one("#tp-input", Input).value)
            if tensor_parallel_size < 1:
                raise ValueError("tensor parallel size must be at least 1")

            gpu_memory_utilization = float(self.query_one("#gpu-util-input", Input).value)
            if not (0 < gpu_memory_utilization <= 1):
                raise ValueError("GPU memory utilization must be between 0 and 1")

            max_num_seqs = int(self.query_one("#max-seqs-input", Input).value)
            if max_num_seqs < 1:
                raise ValueError("max concurrent sequences must be at least 1")

            max_len_raw = self.query_one("#max-len-input", Input).value.strip()
            max_model_len = int(max_len_raw) if max_len_raw else None
            if max_model_len is not None and max_model_len < 1:
                raise ValueError("max model length must be at least 1")

            dtype_value = self.query_one("#dtype-select", Select).value
            dtype_flag = str(dtype_value) if dtype_value else _DTYPE_OPTIONS[-1]
        except ValueError as exc:
            self._set_run_status(f"Invalid parameter: {exc}")
            return

        self._confirm_and_run(
            repo_id=repo_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            dtype_flag=dtype_flag,
        )

    @work(exclusive=True, group="run")
    async def _confirm_and_run(
        self,
        repo_id: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        max_model_len: int | None,
        dtype_flag: str,
    ) -> None:
        args = build_serve_args(
            repo_id=repo_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            dtype_flag=dtype_flag,
            quantization_flag=self._quantization_flag,
        )
        command_display = build_serve_command(
            repo_id=repo_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            dtype_flag=dtype_flag,
            quantization_flag=self._quantization_flag,
        )

        confirmed = await self.app.push_screen_wait(
            ConfirmScreen(f"Run this model?\n\n{command_display}")
        )
        if not confirmed:
            return

        app = cast("NightwatchApp", self.app)
        try:
            pid = launch_vllm_serve(
                args=args,
                repo_id=repo_id,
                command_display=command_display,
                db_path=app.db_path,
            )
        except Exception as exc:
            self._set_run_status(f"Failed to start: {exc}")
            return

        self._set_run_status(f"Started vllm serve (pid {pid}).")

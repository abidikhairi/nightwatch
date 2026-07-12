# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

nightwatch is an interactive CLI/TUI tool that lets a developer discover, configure, and serve
Hugging Face models via vLLM without hand-writing a `vllm serve ...` command every time. Built as
a [Textual](https://textual.textualize.io/) TUI application.

## Commands

This project uses `uv` for dependency management and packaging.

```bash
uv sync                # install/update dependencies
uv run nightwatch       # launch the TUI (entry point: nightwatch.app:run)
uv run pytest           # run the full test suite
uv run pytest tests/test_app.py::test_app_starts   # run a single test
uv run ruff check .     # lint
uv run ruff format .    # format
```

## Architecture

### Screens (`src/nightwatch/screens/`)

`NightwatchApp` (`src/nightwatch/app.py`) is a thin shell (Header/Footer) with key bindings that
push each feature screen: `s` Serve, `e` Explore, `m` Monitor.

- **`ExploreScreen`** — live (debounced) search over Hugging Face models via `nightwatch.hf`,
  selecting a result shows full model metadata plus a read-only deployment recommendation
  (`format_details_markdown`).
- **`ServeModelScreen`** — same live search, but selecting a model populates an *editable* params
  form (tensor-parallel-size, gpu-memory-utilization, max-num-seqs, max-model-len, dtype)
  pre-filled from the recommendation. "Run this model" builds the final `vllm serve` command,
  confirms via `ConfirmScreen`, and on Yes actually launches the process
  (`nightwatch.process.launch_vllm_serve`).
- **`MonitorScreen`** — scans `~/.nightwatch/db.sqlite3`'s `vllm_processes` table (pruning dead
  pids first) and renders the rows in a `DataTable`. Press `r` to rescan.
- **`ConfirmScreen`** — reusable Yes/No modal (`ModalScreen[bool]`), used via
  `self.app.push_screen_wait(ConfirmScreen(question))`.

All screens bind `escape` to `app.pop_screen` (must use the `app.` namespace prefix — that action
only exists on `App`, not `Screen`; a bare `"pop_screen"` binding silently no-ops).

### Hugging Face integration (`src/nightwatch/hf/`)

Kept separate from the UI so it's reusable/testable:

- `client.py` — `search_models()`, `get_model_details()`. Uses `huggingface_hub`'s `HfApi` for
  search/file listing, but pulls `config.json`/`tokenizer_config.json` directly via
  `hf_hub_download` for fields the API's `expand=["config"]` doesn't return (e.g.
  `max_position_embeddings`, `quantization_config`, `hidden_size`).
- `models.py` — `ModelSummary`, `ModelDetails`, `Recommendation` dataclasses.
- `recommend.py` — `recommend_deployment()` estimates GPU memory (weights + KV cache sized for
  N concurrent users) and derives concrete vLLM flags; `build_serve_args()`/`build_serve_command()`
  turn those into an argv list (used to actually exec) and a shell-quoted display string
  (used for UI preview and the db `command` column) respectively — kept separate so the real
  subprocess never round-trips through a re-parsed string.
- `format.py` — `format_model_facts_markdown()` (model facts only) and `format_details_markdown()`
  (facts + recommendation + command block, built on top of it). Serve uses the former since its
  interactive form already surfaces the recommendation; Explore uses the latter as a read-only view.

### Process management (`src/nightwatch/db.py`, `src/nightwatch/process.py`)

- `db.py` owns `~/.nightwatch/db.sqlite3` (path overridable — see below): `init_db()` creates the
  `vllm_processes` table (pid, repo_id, command, started_at, started_by, status);
  `insert_process()`, `list_processes()`, and `prune_dead_processes()` (checks `os.kill(pid, 0)`,
  distinguishing "process gone" from "exists but owned by another user").
- `process.py`'s `launch_vllm_serve()` spawns the real subprocess (`start_new_session=True`, no
  shell) and records it via `insert_process()`.
- `NightwatchApp.on_mount()` calls `init_db()` then runs `prune_dead_processes()` as a background
  worker — so rows for processes that died since the last run are cleaned up automatically at
  startup, and `MonitorScreen` re-runs the same pruning on every scan/refresh.

### Testing conventions

- Tests use `pytest-asyncio` (`asyncio_mode = "auto"`) to drive Textual's `App.run_test()` async
  pilot — see `tests/test_app.py`.
- `NightwatchApp(db_path=...)` takes an explicit db path specifically so tests (and anything else)
  never touch the real `~/.nightwatch/db.sqlite3`; always pass `tmp_path` in tests.
- Textual workers that touch reactive attributes must stay on the main event loop — use
  `async def` + `asyncio.to_thread(...)` for blocking calls (HF API, sqlite), not `@work(thread=True)`.
  Assigning to a reactive from a raw worker thread crashes widgets whose update path needs a
  running event loop (e.g. `Markdown.update()`).
- Never actually invoke a real `vllm serve` command when testing — `vllm` may genuinely be
  installed and reachable on `PATH`, and it will try to load a real model. Mock
  `nightwatch.process.launch_vllm_serve` for UI-flow tests, and use a harmless placeholder command
  (e.g. `["sleep", "1"]`) when testing `launch_vllm_serve` itself directly.

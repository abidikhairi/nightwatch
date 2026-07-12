# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

nightwatch is an interactive CLI/TUI tool that lets a developer serve models via vLLM without
hand-writing a `vllm serve ...` command every time. Built as a [Textual](https://textual.textualize.io/)
TUI application.

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

- `src/nightwatch/app.py` — the `NightwatchApp` Textual application (main UI/entry point).
- `src/nightwatch/__init__.py` — exposes `main` (aliased to `app.run`), which is the
  `nightwatch` console script defined in `pyproject.toml` under `[project.scripts]`.
- Tests live under `tests/` and use `pytest-asyncio` (`asyncio_mode = "auto"`) to drive
  Textual's `App.run_test()` async test harness — see `tests/test_app.py` for the pattern.

The project is a fresh scaffold; the actual vLLM-serving logic (model selection, building/
launching `vllm serve` commands, process management) has not been implemented yet.

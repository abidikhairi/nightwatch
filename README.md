# nightwatch

Interactive TUI for discovering and serving models with [vLLM](https://github.com/vllm-project/vllm) —
no more hand-writing a `vllm serve` command every time you want to spin up a new model.

## Features

- **Explore** (`e`) — live search across Hugging Face models, with key deployment info at a
  glance: parameter count, size on disk, dtype, max context length, chat template/tool support,
  quantization, and a GPU/memory recommendation.
- **Serve** (`s`) — search and select a model, tweak the recommended params (tensor parallel
  size, GPU memory utilization, max concurrent sequences, max model length, dtype), confirm, and
  launch a real `vllm serve` process.
- **Monitor** (`m`) — see all currently-running vLLM processes launched through nightwatch, in a
  table (pid, model, command, start time, user). Press `r` to rescan.

Launched processes are tracked in a local SQLite database at `~/.nightwatch/db.sqlite3`; stale
entries (processes that are no longer running) are cleaned up automatically on startup and on
every Monitor refresh.

## Development

```bash
uv sync              # install dependencies
uv run nightwatch     # launch the TUI
uv run pytest         # run tests
uv run ruff check .   # lint
```

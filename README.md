# nightwatch

Interactive CLI tool for serving models with [vLLM](https://github.com/vllm-project/vllm) —
no more hand-writing a `vllm serve` command every time you want to spin up a new model.

## Development

```bash
uv sync              # install dependencies
uv run nightwatch     # launch the TUI
uv run pytest         # run tests
uv run ruff check .   # lint
```

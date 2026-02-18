# Development

This project is uv-first and uses `uv.lock` to pin dependencies.

## Setup

```bash
uv lock
uv sync --extra dev
```

## Lint and type check

```bash
uv run ruff check demucs_mlx
uv run pyright
```

## Format

```bash
uv run ruff format demucs_mlx
```

## Tests

```bash
make tests
```

## Build

```bash
uv run python -m build
```

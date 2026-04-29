# Krum

Byzantine-resilient aggregation rules for distributed machine learning.

## Supported Python versions

This project supports Python **3.10 through 3.14**.

## Installation

Install in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

## Development

### Linting and formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for unified linting and formatting.

Run the formatter and linter:

```bash
ruff format .
ruff check --fix .
```

### Pre-commit hooks

Install pre-commit hooks to block non-compliant commits:

```bash
pre-commit install
```

## License

MIT License — see [LICENSE](LICENSE).

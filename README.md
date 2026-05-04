# Krum

**Byzantine-resilient aggregation rules for distributed machine learning.**

This project implements various Byzantine-resilient Gradient Aggregation Rules (GARs) for distributed learning. It allows simulating training sessions under different Byzantine attacks to evaluate the robustness of aggregation strategies like Krum and Multi-Krum.

## Documentation

The reference documentation is available at [calicarpa.github.io/krum](https://calicarpa.github.io/krum/).

## Installation

### Supported Python versions

This project supports Python **3.10 through 3.14**.

### From PyPI

```bash
pip install krum
```

With `uv` (Recommended):

```bash
uv pip install krum
# or directly in a uv project
uv add krum
```

### From source

For development or if you want to modify the source, clone the repository and install in editable mode with the development dependencies:

```bash
git clone https://github.com/calicarpa/krum.git
cd krum
pip install -e ".[dev]"
```

With `uv` (Recommended):

```bash
git clone https://github.com/calicarpa/krum.git
cd krum
uv sync --extra dev
```

This installs all linting, type-checking, and documentation tools.

## Contributing

### Linting, formatting, and type-checking

This project uses [Ruff](https://docs.astral.sh/ruff/) for unified linting and formatting, and [ty](https://github.com/astral-sh/ty) for type-checking.

Run the formatter and linter:

```bash
ruff format .
ruff check --fix .
```

Run the type checker:

```bash
ty check
```

### Pre-commit hooks

Install pre-commit hooks to block non-compliant commits:

```bash
pre-commit install
```

### Documentation

Build the documentation locally:

```bash
cd docs
make html  # 
make watch # 
make serve # 
make clean # 
```

## License

MIT License — see [LICENSE](LICENSE).

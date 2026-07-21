# Krum

**Byzantine-resilient aggregation rules for distributed machine learning.**

Krum provides a modular framework for implementing, comparing, and evaluating
Byzantine-resilient Gradient Aggregation Rules (GARs) for distributed learning.
It ships with state-of-the-art aggregation rules and attack strategies.

## Documentation

The reference documentation is available at
[calicarpa.github.io/krum](https://calicarpa.github.io/krum/).

## Quickstart

```python
import torch
from krum.primitives.aggregators import Krum, Average
from krum.primitives.attacks.gaussian import Gaussian

# Simulate gradients from 10 workers (8 honest, 2 Byzantine)
honest = torch.randn(8, 100)
attack = Gaussian(std=10.0)
byzantine = attack.generate(honest, f=2)
gradients = torch.cat([honest, byzantine], dim=0)

# Compare robust vs naive aggregation
robust = Krum.aggregate(gradients, n=10, f=2)
naive = Average.aggregate(gradients)

print(f"Krum result norm:   {robust.norm().item():.4f}")
print(f"Average result norm: {naive.norm().item():.4f}")
```

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

For development or if you want to modify the source, clone the repository and
install in editable mode with the development dependencies:

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

## Features

- **8 aggregation rules**: Average, Median, Trimmed Mean, Krum, MultiKrum,
  Bulyan, Brute, GeoMed
- **5 attack strategies**: SignFlip, ALIE, Gaussian, Omniscient, NoSmallPerturbation
- **Zero-copy model wrapper**: Flat parameter/gradient views via
  `krum.primitives.Model`
- **Stateless design**: Aggregators and attacks are classmethods, no
  instantiation needed

## Contributing

### Linting, formatting, and type-checking

This project uses [Ruff](https://docs.astral.sh/ruff/) for unified linting and
formatting, and [ty](https://github.com/astral-sh/ty) for type-checking.

```bash
ruff format .
ruff check --fix .
ty check
```

### Pre-commit hooks

```bash
pre-commit install
```

### Running tests

Tests use [pytest](https://docs.pytest.org/) and are located under `tests/`.

```bash
uv run pytest tests/ -v
uv run pytest tests/primitives/aggregators/ -v
```

Tests run automatically on every push and pull request via GitHub Actions
(Python 3.10–3.14).

### Documentation

Build the documentation locally:

```bash
cd docs
make html  # Build HTML documentation
make watch # Watch for changes and auto-rebuild
make serve # Build and serve on port 8000
make clean # Remove generated files
```

## License

MIT License — see [LICENSE](LICENSE).

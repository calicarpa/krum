# Krum

**Byzantine-resilient aggregation rules for distributed machine learning.**

This project implements various Byzantine-resilient Gradient Aggregation Rules (GARs) for distributed learning. It allows simulating training sessions under different Byzantine attacks to evaluate the robustness of aggregation strategies like Krum and Multi-Krum.

## Features

- **Byzantine-resilient GARs**: Implementations of Krum, Multi-Krum, Bulyan, Coordinate-wise Median, and standard Averaging.
- **Byzantine Attacks**: Simulation of various attacks (e.g., NaN attack, identical gradients, "A little is enough").
- **Differential Privacy**: Support for Gaussian noise addition for differential privacy.
- **Reproducibility**: Fixed seed support for reproducible experiments.
- **Extensible**: Easy to add new aggregation rules or attacks.
- **High-performance**: Optional native C++ backend for Krum.

## Supported Python versions

This project supports Python **3.10 through 3.14**.

## Installation

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

## Usage

You can run training simulations using `train.py`. The script accepts numerous arguments to configure the experiment.

### Example: Train on MNIST with Multi-Krum

```bash
uv run python train.py \
    --dataset mnist \
    --model simples-conv \
    --gar krum \
    --gar-args m=2 \
    --attack identical \
    --nb-workers 11 \
    --nb-decl-byz 4 \
    --nb-real-byz 4 \
    --nb-steps 500 \
    --result-directory results/multi-krum-test
```

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | -1 | Fixed seed for reproducibility (-1 for random) |
| `--device` | auto | Device to use (e.g., 'cpu', 'cuda', 'auto') |
| `--nb-workers` | 11 | Total number of worker machines |
| `--nb-decl-byz` | 4 | Declared number of Byzantine workers |
| `--nb-real-byz` | 0 | Actual number of Byzantine workers |
| `--gar` | average | Aggregation rule to use (krum, bulyan, median, etc.) |
| `--gar-args` | | Additional GAR arguments (e.g., `m=2`) |
| `--attack` | nan | Attack to simulate (nan, identical, etc.) |
| `--nb-steps` | 300 | Number of training steps |
| `--result-directory` | None | Path to save results and checkpoints |

## Available Algorithms

### Aggregation Rules (GARs)

- *in progress*

### Attacks

- *in progress*

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

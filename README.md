# Krum

**Byzantine-resilient aggregation rules for distributed machine learning.**

This project implements various Byzantine-resilient Gradient Aggregation Rules (GARs) for distributed learning, based on research from the École Polytechnique Fédérale de Lausanne (EPFL). It allows simulating training sessions under different Byzantine attacks to evaluate the robustness of aggregation strategies like Krum and Multi-Krum.

## Features

- **Byzantine-resilient GARs**: Implementations of Krum, Multi-Krum, Bulyan, Coordinate-wise Median, and standard Averaging.
- **Byzantine Attacks**: Simulation of various attacks (e.g., NaN attack, identical gradients, "A little is enough").
- **Differential Privacy**: Support for Gaussian noise addition for differential privacy.
- **Reproducibility**: Fixed seed support for reproducible experiments.
- **Extensible**: Easy to add new aggregation rules or attacks.
- **High-performance**: Optional native C++ backend for Krum.

## Installation

This project uses `uv` for dependency management.

```bash
# Clone the repository
git clone https://github.com/your-username/krum.git
cd krum

# Install dependencies
uv pip install -e .
```

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

- **`average`**: Standard averaging of gradients (no Byzantine resilience).
- **`krum`**: Multi-Krum algorithm. Selects the $m$ gradients with the lowest scores (sum of distances to $n-f-1$ nearest neighbors).
- **`bulyan`**: Bulyan algorithm built on top of Multi-Krum. Requires $n \ge 4f + 3$.
- **`median`**: Coordinate-wise median aggregation.

### Attacks

- **`nan`**: Injects NaN-valued gradients to test fault tolerance.
- **`identical`**: Injects identical malicious gradients (e.g., "A Little is Enough", "Empire").

## Project Structure

```
krum/
├── train.py              # Main training script
├── reproduce.py          # Script to reproduce paper results
├── histogram.py          # Visualization utilities
├── pyproject.toml        # Project metadata and dependencies
├── aggregators/          # GAR implementations
│   ├── __init__.py       # GAR loader and registry
│   ├── krum.py           # Multi-Krum implementation
│   ├── bulyan.py         # Bulyan implementation
│   ├── median.py         # Coordinate-wise median
│   └── average.py        # Standard average
├── attacks/              # Byzantine attack implementations
│   ├── __init__.py       # Attack loader and registry
│   ├── nan.py            # NaN attack
│   └── identical.py      # Identical gradient attacks
├── experiments/          # Experiment configurations and models
├── native/               # C++ native backend for performance
└── tools/                # Utility functions and helpers
```

## References

- Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent*. NeurIPS.
- El Mhamdi, E. M., Guerraoui, R., & Rouault, S. (2018). *The Hidden Vulnerability of Distributed Learning in Byzantium*. ICML.
- Xie, C., Koyejo, O., & Gupta, I. (2019). *Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation*. UAI.
- Baruch, M., Baruch, M., & Goldberg, Y. (2019). *A Little Is Enough: Circumventing Defenses For Distributed Learning*.

## Acknowledgements

We would like to thank the authors of the original papers on Byzantine-resilient distributed learning, including Rachid Guerraoui, Sébastien Rouault, El Mahdi El Mhamdi, and others, for their foundational work in this field.

## License

See [LICENSE](LICENSE) for details.

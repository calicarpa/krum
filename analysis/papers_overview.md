# Bridge Between My Analysis and the Existing Notes

This note links your analysis to the Markdown files already present in `analysis/`. It serves as an index between a high-level synthesis and the detailed notes already written.

## Quick Mapping

| Synthetic analysis | Existing notes |
| --- | --- |
| Unified Breakdown Analysis for Byzantine Robust Gossip | [byzantine_robust_gossip_analysis.md](byzantine_robust_gossip_analysis.md) |
| SignGuard: Collaborative Malicious Gradient Filtering | [signguard_analysis.md](signguard_analysis.md) |
| DECOR: Decentralized SGD with Differential Privacy | [decor_analysis.md](decor_analysis.md) |
| ByzFL: Research Framework for Robust Federated Learning | [byzfl_analysis.md](byzfl_analysis.md) |
| Robust and Efficient Collaborative Learning (RPEL) | [robust_collaborative_learning_analysis.md](robust_collaborative_learning_analysis.md), [rpel_bf2d_analysis.md](rpel_bf2d_analysis.md) |
| Robust Collaborative Learning with Linear Gradient Overhead | [robust_collaborative_learning_analysis.md](robust_collaborative_learning_analysis.md), [rpel_bf2d_analysis.md](rpel_bf2d_analysis.md) |

## Paper Details

### 1. Unified Breakdown Analysis for Byzantine Robust Gossip

- Associated file: [byzantine_robust_gossip_analysis.md](byzantine_robust_gossip_analysis.md)

### What the paper does

The paper studies the breakdown point of decentralized learning architectures under Byzantine attacks, both theoretically and empirically. It introduces a `SpH`-style attack to break existing algorithms by exploiting the network topology.

### Why they use the code

They need a decentralized orchestration engine capable of simulating complex graph topologies, such as Erdős-Rényi graphs or weakly connected bipartite graphs, in order to empirically evaluate the resilience of algorithms like NNA and ClippedGossip.

### What to implement in the library

- A dynamic communication topology manager, without assuming a complete graph.
- Arbitrary connectivity matrices.
- Attacks that target topology, not just gradient magnitude.

### 2. SignGuard: Collaborative Malicious Gradient Filtering

- Associated file: [signguard_analysis.md](signguard_analysis.md)

### What the paper does

The paper proposes a statistical filter before aggregation. Instead of relying only on Euclidean distance, it examines the distribution of signs and the magnitude of gradients to detect anomalies, including subtle attacks.

### Why they use the code

The authors reuse the base federated infrastructure and standard aggregators to add their filter as a preprocessing step. The framework is also used to generate non-IID distributions and measure the effects of estimation bias under attack.

### What to implement in the library

- Pre-aggregation hooks to insert arbitrary statistical filters.
- A robust data partitioning module to simulate different levels of non-IID.

### 3. DECOR: Decentralized SGD with Differential Privacy

- Associated file: [decor_analysis.md](decor_analysis.md)

### What the paper does

The paper addresses differential privacy in decentralized learning. It adds correlated Gaussian noise that cancels out in pairs to protect communications against honest-but-curious nodes.

### Why they use the code

The authors mainly reuse the networking plumbing of decentralized training, then inject their seed-sharing mechanism and stochastic noise. Robust aggregation is not their main concern.

### What to implement in the library

- Decouple the transport/network layer from the robustness layer.
- Allow stochastic transformations on exchanged tensors without breaking gradient descent.

### 4. ByzFL: Research Framework for Robust Federated Learning

- Associated file: [byzfl_analysis.md](byzfl_analysis.md)

### What the paper does

ByzFL is a research library for evaluating Byzantine attacks and defenses in federated learning. It standardizes the components around a high-level Python API.

### Why they use the code

This is not just a raw reuse of code: the historical base was refactored, redesigned, and packaged to produce a cleaner and more accessible open-source standard.

### What to implement in the library

- Study their API and the clear separation between aggregators, attacks, and simulators.
- Provide primitives that are at least as expressive.
- Better support pure decentralized settings if needed.

### 5. Robust and Efficient Collaborative Learning (RPEL)

- Associated files: [robust_collaborative_learning_analysis.md](robust_collaborative_learning_analysis.md), [rpel_bf2d_analysis.md](rpel_bf2d_analysis.md)

### What the paper does

The paper combines robustness, privacy, and efficiency by reducing communication costs. It focuses in particular on gradient compression and quantization.

### Why they use the code

The research base is used to integrate compression operators, such as sparsification or quantization, alongside robustness filters.

### What to implement in the library

- Compression and quantization operators.
- Compatibility between robust aggregation and sparse vectors.
- A more realistic approach than simply exchanging dense float32 vectors.

### 6. Robust Collaborative Learning with Linear Gradient Overhead

- Associated files: [robust_collaborative_learning_analysis.md](robust_collaborative_learning_analysis.md), [rpel_bf2d_analysis.md](rpel_bf2d_analysis.md)

### What the paper does

The paper addresses the high computational cost of some robust aggregators, often quadratic in the number of nodes, by proposing an algorithm with linear overhead.

### Why they use the code

The experimental base is used to profile CPU and GPU time, then compare the scalability of a new linear aggregator with more expensive historical methods.

### What to implement in the library

- Aggressive optimization of tensor operations.
- Avoid full distance-matrix computations in pure Python.
- Vectorize aggregators to scale to large numbers of workers.

# Unified Attacks & Aggregators List

## Attacks

| # | Name | Description | To Validate | Done |
|---|------|-------------|--------------|------|
| 1 | sign_flip / signflipping / SignFlipping | Invert gradient sign | | |
| 2 | label_flip / labelflipping / LabelFlipping | Flip labels on client side | | |
| 3 | mimic / Mimic | Mimic another worker's gradient | | |
| 4 | nan / NaN | Send NaN values | | |
| 5 | scaledSF / scaled_sign_flip | Scaled sign flipping | | |
| 6 | mimic_heuristic | Heuristic mimic | | |
| 7 | empire | Empire attack | | |
| 8 | anticge | Anti-CGE | | |
| 9 | identical_sparse | Identical sparse | | |
| 10 | identical | Identical gradients | | |
| 11 | dissension | Dissension | | |
| 12 | InnerProductManipulation / IPM | Scale mean by -tau | | |
| 13 | Optimal_InnerProductManipulation | Optimized IPM | | |
| 14 | ALittleIsEnough / ALIE / lie | Scale mean + tau * std | | |
| 15 | Optimal_ALittleIsEnough | Optimized ALIE | | |
| 16 | Inf | Send infinity | | |
| 17 | Gaussian | Random gaussian noise | | |
| 18 | zero | Send zeros | | |
| 19 | noise | Add random noise | | |
| 20 | random | Random attack | | |
| 21 | byzMean | Byzantine mean | | |
| 22 | min_max / minmax | Min-max attack | | |
| 23 | min_sum | Min-sum attack | | |
| 24 | adaptive_std | Adaptive std attack | | |
| 25 | adaptive_sign | Adaptive sign attack | | |
| 26 | adaptive_uv | Adaptive UV attack | | |

## Aggregators

| # | Name | Description | To Validate | Done |
|---|------|-------------|--------------|------|
| 1 | average / Mean / Average | Simple arithmetic mean | | |
| 2 | median / Median | Coordinate-wise median | | |
| 3 | trmean / TrMean / trimmed_mean | Trimmed mean | | |
| 4 | krum / Krum | Krum aggregator | | |
| 5 | multikrum / MultiKrum / Multi-Krum | Multi-Krum | | |
| 6 | bulyan / Bulyan | Bulyan | | |
| 7 | geometric_median / GeoMed / GeometricMedian | Geometric median | | |
| 8 | dnc / DnC / divide_conquer | Divide and conquer | | |
| 9 | signguard / SignGuard | Sign guard | | |
| 10 | rfa / RFA | Robust federated aggregation | | |
| 11 | mom / MoM | Momentum-based | | |
| 12 | mva / MVA | Minimum variance averaging | | |
| 13 | mea / MEA | Mean estimation | | |
| 14 | cva / CVA | Coordinate-wise vector agreement | | |
| 15 | iter_cva / Iter_CVA | Iterative CVA | | |
| 16 | cenna / CENNA | Cross-entropy neural network aggregator | | |
| 17 | bucketing | Bucketing | | |
| 18 | bucketing_stress | Bucketing stress | | |
| 19 | brute / Brute | Brute force | | |
| 20 | aksel | Aksel | | |
| 21 | cge / CGE | CGE | | |
| 22 | centeredclip / CenteredClipping | Centered clipping | | |
| 23 | filterL2 | L2 filtering | | |
| 24 | krum_pseudo | Krum pseudo | | |
| 25 | multiKrum_pseudo | Multi-Krum pseudo | | |
| 26 | ios | IOS | | |
| 27 | robust_gossip | Robust gossip | | |
| 28 | MDA | Minimum Diameter Averaging | | |
| 29 | MoNNA | MoNNA | | |
| 30 | Meamed | Mean around median | | |
| 31 | CAF | Covariance-bound Agnostic Filter | | |
| 32 | SMEA | Smallest Maximum Eigenvalue Averaging | | |
| 33 | SignGuard-Sim | SignGuard simplified | | |
| 34 | SignGuard-Dist | SignGuard distant | | |

## Preaggregators

| # | Name | Description | To Validate | Done |
|---|------|-------------|--------------|------|
| 1 | NNM | Nearest Neighbor Mixing | | |
| 2 | Bucketing | Bucketing | | |
| 3 | Clipping | Static clipping (L2 norm) | | |
| 4 | ARC | Adaptive Robust Clipping | | |

## Data Distributions (IID / Non-IID)

| # | Name | Description | To Validate | Done |
|---|------|-------------|--------------|------|
| 1 | iid / IID | Independent and identically distributed (random split) | | |
| 2 | noniid / non-iid / Non-IID | Each client has different labels | | |
| 3 | hetero / heterogeneous | Different data per worker (one iterator per worker) | | |
| 4 | distinct_data | Honest workers have distinct datasets | | |
| 5 | dirichlet_niid | Dirichlet distribution (alpha parameter) | | |
| 6 | extreme_niid | Extremely non-IID (limited labels per client) | | |
| 7 | gamma_similarity_niid | Gamma-similarity non-IID | | |
| 8 | noniid_s | Mixed IID/Non-IID (parameter s) | | |
| 9 | shards_noniid | Shards-based (2 shards per client) | | |
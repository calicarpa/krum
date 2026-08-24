# Comparison table — clarification notes

Source: PLM Latex

## Data partitioning

### FedLab — 9 schemes
Source: `notes/sota-analysis.md:6.1` and `fedlab/utils/dataset`.

The 9 FedLab schemes to describe in prose (table keeps only the count `9`):

1. Balanced IID
2. Unbalanced IID
3. Heterogeneous Dirichlet
4. Shards
5. Balanced Dirichlet
6. Unbalanced Dirichlet
7. Quantity-based label skew
8. Noise-based feature skew
9. FCUBE synthetic

14 supported datasets (CIFAR-10/100, MNIST, FashionMNIST, SVHN, CelebA, FEMNIST, Shakespeare, Sent14, Reddit, Adult, Covtype, RCV1, FCUBE, LEAF-Synthetic).

### shard (Blades) and sort (FL-Byz-Lib)

- **shard** (Blades, `Dir, shard`): sort by label then split into shards distributed to workers — a label-partition non-IID variant. Same family as FedLab's `shards`, implemented via `fedlib` (Blades' external dependency).
- **sort** (FL-Byz-Lib, `IID, Dir, sort`): `sort` = sort by label then sequential split (equivalent to `sort-then-partition`, close to Krum's `PerLabels` with $\lambda=0$). Normalize in prose as `sort` or `label-sorted split`.

### Table counts → prose mapping (moved from footnote)

Table shows counts only; details belong in prose:

- Krum: IID, PerLabels, Dirichlet, Mixing (4)
- ByzFL: IID, Dirichlet, $\gamma$-similarity (3)
- FedLab: 9 schemes (see above) (9)
- Blades: Dirichlet, shard (2)
- FL-Byz-Lib: IID, Dirichlet, sort (3)
- ByzPy: IID, Dirichlet (2)

## Protocol fidelity — definitions

To be defined in 2–3 sentences in `Section Comparison to Related Software`, not in the table.

- **faithful** (Krum `3 faithful`): exact reproduction of a paper's protocol — LR schedule, initialization, weight decay, metrics and `stop_attack_at` identical to the reference publication (NeurIPS 2017, ICML 2018, ICML 2023). A curated reproduction suite, not a generic loop.
- **generic** (ByzFL `2 generic`, Blades `1 generic (FedAvg+DP)`, FL-Byz-Lib `1 generic`, ByzPy `2 generic (PS+P2P)`): generic training loop (FedAvg or DSGD) where the user plugs in an aggregator/attack without fidelity to any specific paper.
- **baselines** (FedLab `10 baselines`): suite of generic FL algorithms provided as references (FedAvg, FedProx, SCAFFOLD, FedDyn, q-FFL, FedNova, IFCA, Ditto, pFedAvg, CFL) — not related to Byzantine robustness.

### PS+P2P and Byzantine reach (moved from footnote)

- **PS** = parameter server (centralized).
- **P2P** = peer-to-peer (decentralized).
- **Krum `PS+P2P`**: both topologies with faithful protocols (PS: NIPS 2017/ICML 2018, P2P: MoNNA ICML 2023 with `all / sampled` for Byzantine reach). Only library with faithful P2P.
- **ByzPy `2 generic (PS+P2P)`**: both topologies supported but via generic helpers (`NeighborSampler`, `ActorPool`) without a faithful paper protocol — do not conflate topology and fidelity.
- Blades, ByzFL, FL-Byz-Lib, FedLab: `PS` only.
- **Byzantine reach** (P2P only): whether the adversary controls which workers receive Byzantine models — `all`: worst case, every Byzantine model reaches every worker; `sampled`: gossip, each worker receives 0 to $f$ Byzantine models. To be explained in prose, not table footnote.

## Parallelism — two rows

The table splits parallelism into **sweep** (independent experiments) and **deployment / communication** (distributed execution). This addresses Peva's remark: do not conflate `actors`, `Ray`, and sweep parallelism.

### Parallel exps (sweep) — sweep-level parallelism

- **Krum `1 proc / run (sim seq.)`**: each `orch.run()` in a separate process (`multiprocessing`), PRNG seed handling preserved; `sim.step()` itself is sequential — no intra-simulation parallelism. Source: `notes/sota-analysis.md:2.2`, `krum/orchestration`.
- **ByzFL `Pool (per config)`**: `benchmark/run_benchmark()` builds the Cartesian product of JSON configs and executes them via `multiprocessing.Pool` — 1 worker per config. Verified in `sota-analysis.md:3.2`.
- **FedLab `---`**: no native sweep scheduler; sweeps are manual. Its 3 modes belong to the next row.
- **Blades `Ray Tune`**: `blades/train.py` + `tuned_examples/*.yaml` use `ray.tune` for parallel sweeps — sweep-level parallelism.
- **FL-Byz-Lib `---`**: `main.py` / `fl-byzantine` CLI runs sequentially; no sweep parallelism.
- **ByzPy `ActorPool (DAG)`**: `Scheduling Layer` (`ComputationGraph` + `NodeScheduler` + `ActorPool`) fans out tasks via DAG — usable for sweeps among other task parallelism (`sota-analysis.md:6.4`).

### Deployment — deployment / communication parallelism

- **Krum `single host`**: simulations run on a single host, sequential; no distributed deployment (both PS and P2P simulated locally).
- **ByzFL `single host`**: runs locally via Pool; no cross-machine deployment in the benchmark module.
- **FedLab `3 modes`**: `standalone` (single process), `cross-process` (multi-GPU via `torch.distributed`), `hierarchical-hybrid` — deployment topologies, not sweeps.
- **Blades `Ray`**: execution distribution via Ray (actors/callbacks), outside `ray.tune`.
- **FL-Byz-Lib `---`**: no distributed deployment (`--cl_part` only controls client participation locally).
- **ByzPy `5 backends`**: 5 `ActorRef` backends — `Thread`, `Process`, `GPU`, `Remote TCP`, `UCX` (RDMA) — for distributed deployment/communication. Source: `sota-analysis.md:6.4`.

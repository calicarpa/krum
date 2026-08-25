# Which lib fits better to Claude? — study pending

**Which API is the most `Claude-friendly` for generating correct code without hallucination.**

> To be decided before integrating a sentence in `§3 Comparison to Related Software` (or leaving it in `notes`).

## Context
- Krum: `stateless @classmethod` (`Krum.aggregate(grads, f=2, n=10)`) — functional, no state, complete `ty` typing
- ByzFL: `instance + JSON` (`aggregator = Krum(f=2); aggregator(vectors)` + `benchmark/config.json` + `Pool`)
- ByzPy: `actors/DAG` (`Operator` + `ComputationGraph` + `ActorPool` 5 backends)
- Initial source: `notes/sota-analysis.md:7` + `notes/comparison-table-notes.md:Parallelism — two rows`

## Criteria to measure (3 identical prompts × 3 libs)
1. Call simplicity (one line vs graph)
2. Composability (stateless vs DAG)
3. Typing/docs and Claude's correct generation rate (compile/run without correction)

## Expected deliverable
One JMLR sentence in `§3`, e.g.:
> *Ergonomics for LLM code generation: Krum's stateless functional API is the most Claude-friendly for correct code generation without hallucination.*

## Why not FedLab and FL-Byzantine-Library

The Claude-friendly study measures **code generation against a callable aggregator API** (one-line call,
composition, typed config). Two related libraries from the comparison table are out of scope for that axis:

- **FedLab** — not a Byzantine-robust *aggregator* library. Its 10 "baselines" are generic FL algorithms
  (FedAvg, FedProx, SCAFFOLD, …) unrelated to Byzantine robustness (`comparison-table-notes.md:46`). Its
  surface is a training-loop framework + deployment topologies (standalone / cross-process / hierarchical),
  not an aggregator you invoke in one line. Different category → not comparable on aggregator API ergonomics.
- **FL-Byz-Lib** — is Byzantine-related (Krum-like aggregators) but exposes a **CLI**, not a Python API
  (`main.py` / `fl-byzantine` CLI, sequential, `--cl_part` local only; `comparison-table-notes.md:67,76`).
  There is no callable surface for Claude to generate code against and no typed config to complete → out of
  scope for *code-generation* ergonomics. (It would be relevant for a CLI-usability study, not this one.)

Kept in the study: **Krum** (stateless `classmethod`), **ByzFL** (instance + JSON + `Pool`),
**ByzPy** (`Operator` + `ComputationGraph` + `ActorPool`) — three genuinely different *programmatic* API shapes.

## Protocol (prompts + scoring — to run in a fresh session)

Each prompt is given **verbatim** to Claude in a new conversation, with only the `<<LIB>>` token substituted
(`Krum` | `ByzFL` | `ByzPy`). No repo context, no file contents. 3 prompts × 3 libs = 9 generations.
Record the **first** output only — no follow-up correction.

### Prompt A — call simplicity (one line vs graph)
```
Using the <<LIB>> library, aggregate a set of 10 gradient vectors (PyTorch tensors) where 2 are
Byzantine (attacked). Use the Krum aggregator with f=2. Return the single aggregated gradient vector.
Write only the code that performs the aggregation.
```

### Prompt B — composability (stateless vs DAG)
```
Using the <<LIB>> library, build a pipeline that first applies the Krum aggregator (f=2) to a set of
client gradients, then applies a TrimmedMean aggregator (beta=2) to the Krum output plus the remaining
honest gradients. Show how the two aggregators compose, and write the code for the composed step.
```

### Prompt C — typing / parallelism (config + types)
```
Using the <<LIB>> library, set up a benchmark sweep over f in {1, 2, 3} for the Krum aggregator, using
the library's native configuration / parallelism mechanism (JSON config, Pool, ActorPool/ComputationGraph,
or equivalent). Show the typed config and the call that launches the sweep.
```

## Scoring grid

For each of the 9 generations, score:

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Compiles / typechecks without edit | fails | needs 1 fix | passes as-is |
| Runs to correct output without correction | fails | runs after 1 fix | runs as-is |
| Call simplicity (A) | graph/boilerplate heavy | moderate | one clear call |
| Composability (B) | no composition shown | manual re-wrap | native composition |
| Typing/config clarity (C) | untyped/loose | partial types | complete, typed config |

**Correct-generation rate** = `runs as-is (2) / 9`, per lib and overall. This is the headline number for §3.

### Notes for the runner
- Work on `test` (already rebased on `main`); the two study files are on `main`.
- Krum API ref: `Krum.aggregate(grads, f=2, n=10)` (stateless `classmethod`).
- ByzFL API ref: `aggregator = Krum(f=2); aggregator(vectors)` + `benchmark/config.json` + `Pool`.
- ByzPy API ref: `Operator` + `ComputationGraph` + `ActorPool` (5 backends).

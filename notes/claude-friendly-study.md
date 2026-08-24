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

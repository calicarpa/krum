# How the orchestrator works

A high-level tour of `krum.orchestration`. For *why* it is built this way, and
for the details each decision turns on, see
[2026-10-02-orchestrator-v2-a-design.md] and [adr-2026-07-31.md].

## The problem it solves

A study is a sweep: one experiment run over many hyper parameter tuples. Doing
that by hand has two costs. Re-running the sweep after a change repeats the
runs that were already fine, which on a 300-round simulation is minutes to
hours of nothing. And a result on disk is of little use later if you cannot say
which code produced it.

The orchestrator addresses both with one idea: **give every run an identity
derived from the question it answers, and keep its output in a folder named by
that identity.** A run whose folder already holds an answer is not run again.

## A job

A *job* is one execution of the user's experiment function on one hyper
parameter tuple. Its identity — the `job_key` — is a hash of:

- the parameters, bound to the function's signature with defaults applied, so
  that parameter *names* count and an unstated default still identifies the job;
- the function's own bytecode;
- everything that function reaches *by name*: the helpers it calls, the classes
  passed to it, the constants it reads, transitively, stopping at the boundary
  of the modules you own (`krum`, your own package) so that a hash never walks
  into pytorch.

Nothing is executed to compute this. Two consequences worth internalising:

- **Changing code that the key covers produces a different key**, hence a
  different folder. The old result stays where it is; the new one is computed
  beside it. A code version is a different question, not a stale answer.
- File and line positions are excluded, so reformatting, moving a function
  within its file, or adding a comment above it changes nothing.

## Writing an experiment

The experiment is an ordinary function. It takes hyper parameters, and records
numbers through `Metric`, which finds the running job by itself:

```python
from krum.orchestration import Metric, Orchestrator
from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.krum import Krum

def my_experiment(n, f, aggregator, rounds, seed=42):
    simulation = ...  # build it here, from the parameters
    loss = Metric("loss", dtype=float)
    for step in range(rounds):
        simulation.step()
        loss.push(step, simulation.evaluate()[0])

with Orchestrator("results/byzantine_study") as orch:
    for aggregator in (Average, Krum):
        for f in (0, 2):
            orch.run(my_experiment, n=10, f=f, aggregator=aggregator, rounds=300)
```

Two conventions matter:

- **Build heavy objects inside the function**, from hashable parameters. Pass
  `dataset="mnist"`, not a loaded `Dataset`. The parameters are what identify
  the job, so they should describe the run rather than be its materials.
- `run` only *enqueues*. The queue is drained when the `with` block exits, or
  on the first `get`. Nothing happens before that.

## Reading results back

`get` returns a tidy `pandas.DataFrame`, one row per recorded point, with the
sweep's parameters alongside:

```python
loss = orch.get("loss")
# columns: step, value, n, f, aggregator, rounds, seed, job_key

loss.groupby(["aggregator", "f"])["value"].mean()
loss[loss["aggregator"] == "Krum"]
```

It reads the jobs enqueued on *that* orchestrator, in that order — not every
folder in the store. This matters because a store accumulates one folder per
code version: reading all of them would mix versions, two of which can carry
the same label and be drawn over each other without saying so. Use
`krum.orchestration.metrics.collect(store, name)` to read a whole store
deliberately.

`step`, `value` and `job_key` are the frame's own columns, so a parameter of
one of those names is refused when the run is enqueued.

## What makes a job run

`plan` answers this without running anything, which is the cheapest way to see
what a sweep is about to do:

```python
for decision in orch.plan():
    print(decision)
```
```
skip 3ca7515b22969224
run 561b37817a14e2ba (no recorded result)
run d55cb07373e9221e (uv_lock changed (651669ac047d… -> 81583a2f1641…))
run 66d74f96aa945bdf (helpers:compute changed)
```

The reasons, in the order they are checked:

| reason | means |
|---|---|
| `no recorded result` | no folder for this key — a new configuration, or code the key covers has changed |
| `previous attempt failed` | the folder is there but marked `FAILED` |
| `forced` | `force=True` was passed |
| `<witness> changed` | same question, but the environment it was answered in differs — `uv_lock`, the interpreter, the `-O` flag |
| `<module:name> changed` | same question, but a function the job *called at runtime* has been edited |

The last two are the interesting half. A job key covers what the experiment
reaches by name; it cannot cover a dependency reached only while running — a
module imported inside a function body, a helper pulled from a dict of
handlers — nor third-party versions, which are folded in as a name only. Those
live in a **fingerprint** written beside the output, and re-checked on each
pass. So:

- code the key covers changes → **new key**, new folder, reason
  `no recorded result`;
- the environment, or code reached only at runtime, changes → **same key**, and
  the folder is re-run with the drifted entry named.

## What a run looks like

```
$ uv run python -m experiments.centralised.krum_nips_2017.experiment
...
4 done
  done     3ca7515b22969224     7.102s  (no recorded result)
  done     ad74a48355e052fc     4.416s  (no recorded result)
  done     c8382b63f1725eb6     6.302s  (no recorded result)
  done     ac3ca6ebd48d19d9     4.286s  (no recorded result)

$ uv run python -m experiments.centralised.krum_nips_2017.experiment
4 skipped

# after editing MultiKrum.aggregate
$ uv run python -m experiments.centralised.krum_nips_2017.experiment
2 done, 2 skipped
```

That last line is the point of the whole thing: the two configurations using
MultiKrum were recomputed, and the two using Mean were left alone.

## On disk

```
results/byzantine_study/
  .staging/<key>.<pid>/   a job being built right now
  <job_key>/
    manifest.json   parameters, which function, git commit and whether the
                    tree was dirty, the environment, timings
    deps.json       the fingerprint: environment witnesses, and the functions
                    the job called with a hash of each
    metrics/*.csv   append-only (step, value), one file per metric
    DONE | FAILED   FAILED carries the traceback
```

A job is built in `.staging` and renamed into place only once its marker is
written, so a folder at its final path is never half-finished. A job killed
mid-flight leaves nothing there and is simply "not done" next time. A job that
*failed* is kept, with its traceback, so you can read it.

The manifest is the traceability half: given a number in a plot, it says which
parameters, which commit, and which resolved environment produced it.

## Where the code lives

| module | role |
|---|---|
| `hashing` | job keys, and the ownership boundary that bounds them |
| `storage` | the folder protocol, the manifest, the fingerprint |
| `metrics` | the `Metric` writer, and the frame read back |
| `tracing` | recording which functions a job actually called |
| `execution` | running a job's body, in process or in a child |
| `__init__` | `Orchestrator`, which ties the above together |

## Things worth knowing

- **Failure stops the sweep.** The first failing job ends the drain and the
  queue is abandoned, so the results that did complete stay readable. The
  failed job's folder is marked `FAILED`; re-enqueueing retries it.
- **A dirty git tree warns.** Results are still recorded, with `dirty: true` in
  the manifest, but the stored commit will not reproduce them.
- **Unhashable dependencies raise.** If something the experiment reaches cannot
  be hashed reproducibly, `HashError` says so rather than quietly hashing a
  placeholder, which would hide a change.
- **`isolate=True`** runs each job in a fresh process, so nothing one job
  leaves behind reaches the next. It needs the experiment and its parameters to
  be picklable, and a sweep script guarded by `if __name__ == "__main__":`.
- **Tracing is on by default** and wants one of Python's monitoring tool ids;
  pass `trace=False` if you need to run under a debugger or profiler.
- **Folders accumulate.** One per code version, kept indefinitely. Nothing
  collects the superseded ones yet.

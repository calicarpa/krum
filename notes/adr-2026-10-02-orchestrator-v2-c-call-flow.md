# Orchestrating a sweep, step by step

What the objects in `krum.orchestration` call, in the order they call it, from
constructing an `Orchestrator` to reading a metric back. This is the
maintainer's view: for the mental model see
[adr-2026-10-02-orchestrator-v2-b-overview.md], and for why it is shaped this
way see [adr-2026-10-02-orchestrator-v2-a-design.md].

## The cast

| object | module | owns |
|---|---|---|
| `Orchestrator` | `__init__` | the queue, the decisions, the sweep's key list |
| `PendingRun` | `__init__` | one enqueued callable, its parameters, its key |
| `JobDecision` | `__init__` | whether a run executes, and why |
| `JobOutcome`, `RunSummary` | `__init__` | what happened, for reporting |
| `Hasher`, `Modules` | `hashing` | deriving a key, and where recursion stops |
| `JobStore` | `storage` | the root directory, staging, promotion |
| `JobFolder` | `storage` | one job's directory, read-only |
| `JobWriter` | `storage` | one job under construction, and its manifest |
| `MetricRecorder` | `storage` | the half of a job a child process can own |
| `Metric`, `Sink` | `metrics` | one metric's open file |
| `MetricTable` | `metrics` | the interface a metric reads back as |
| `InMemoryTable` | `metrics` | the rows read back, held in memory |
| `DependencyTracker` | `tracing` | what the job called |
| `InlineRunner`, `SubprocessRunner` | `execution` | where the body runs |

The modules are layered, and nothing points back up:

```
hashing                 (no intra-package imports)
  ├── storage  ─────────────► metrics
  └── tracing  ─────────────► execution ◄── storage
                                  │
__init__ ◄── execution, hashing, metrics, storage, tracing
```

## 1. Construction

```python
orch = Orchestrator("results/study", owned=None, lock=None, source=None,
                    force=False, isolate=False, trace=True)
```

`Orchestrator.__init__` builds a `JobStore(root)`, whose own `__init__`
creates the root directory. Everything else is recorded and nothing else
happens: no scan of existing folders, no environment probe.

## 2. Enqueueing a run

```python
orch.run(my_experiment, n=10, f=2, aggregator=Krum)
```

1. `hashing.bind_params(callable, params)` binds the parameters to the
   signature and applies defaults.
2. `metrics.reserved(...)` checks the resulting names against
   `RESERVED_COLUMNS`; a clash raises `ValueError` here, at the call site.
3. `Orchestrator._owned_for(callable)` returns the ownership setting as a
   `Modules`, falling back to `owned_for(callable)`, which is
   `Modules({"__main__", "krum", <the callable's top-level package>},
   local=True)`: a local test also owns, at lookup time, every module loaded
   from outside the standard library and site-packages, such as a `utils.py`
   next to the sweep script.
4. `hashing.static_key(callable, params, owned)` builds a `Hasher`, pushes the
   callable, then pushes `bind_params(...)` again, and digests. This is where
   the key's whole dependency walk happens — see
   [adr-2026-10-02-orchestrator-v2-a-design.md]. `HashError` surfaces here.
5. A `PendingRun(callable, params, key)` goes on `_queue`, and the key's hex
   name is appended to `_enqueued` if not already there.

Nothing is executed, and nothing touches the store. `run` returns the key,
which is provisional: `plan` and `drain` both start with
`Orchestrator._refresh()`, which recomputes every pending key through
`PendingRun.refresh` and, through `_enlist`, replaces the provisional name in
`_enqueued` when it moved. A module value set after `run` thus counts.

## 3. Deciding what to execute

`Orchestrator.drain()` and `Orchestrator.plan()` both go through
`Orchestrator.decide(pending, force)`, which is the only place the re-run rule
lives:

1. `JobStore.name_for(pending.key)` → the folder name; `JobStore.folder_for`
   → a `JobFolder` (which need not exist).
2. `JobFolder.status` reads the markers: `absent` → run
   (`"no recorded result"`); `failed` → run (`"previous attempt failed"`).
3. The effective `force` → run (`"forced"`).
4. `storage.drift(folder.witnesses(), Orchestrator.witnesses())`. The
   orchestrator's side is computed once and cached:
   `witnesses_of(environment_fingerprint(lock, source))`, which calls
   `find_lock`, `hash_file` and `installed_packages`. The folder's side is
   read from `deps.json`.
5. `tracing.verify_called(folder.called() or {})`. For each recorded entry it
   does `Location.decode(...)`, `Location.fetch()` (an import plus `getattr`),
   and `hashing.callee_key(...)`, comparing against the stored hash. An entry
   recorded as unhashable, or unhashable now, is a reason on its own. Entries
   are functions entered, members of owned modules that the entered code
   names (`DependencyTracker._members_named`), and extension modules.
6. A `JobDecision` carries `run`/`skip` and the accumulated reasons.

`plan()` stops here, which is why it runs nothing.

## 4. Preparing a job

Only for a decision that runs. `drain` first selects the runner, once for the
whole sweep:

- `Orchestrator._owned_modules()` → `_owned_for(queue[0].callable)`, the
  `Modules` the tracer records against; it is picklable, so a child receives
  it as is.
- `Orchestrator.runner(owned)` → `SubprocessRunner(owned, trace)` when
  `isolate`, else `InlineRunner(owned, trace)`. It is entered as a context
  manager, so a pool is released at the end of the drain.

Then, per job:

1. `storage.build_manifest(key, callable, pending.bound_params, owned, lock,
   start)` assembles the manifest, calling `encode_params` (which walks values
   through `encode_value`), `git_provenance` (which shells out to `git`), and
   `environment_fingerprint`, which calls `find_lock` and then `hash_file`
   only if a lock file was found, and `installed_packages`.
2. `Orchestrator._warn_if_dirty(manifest)` warns once per sweep if
   `git.dirty`.
3. `JobStore.writer(key, manifest)` → `JobWriter.__init__`, which calls
   `MetricRecorder.__init__(store.staging_path(key), create=True)`: that
   removes any leftover staging directory and creates
   `<root>/.staging/<key>.<pid>/metrics/`. The manifest is held in memory, not
   yet written.

## 5. Executing the body

`runner.run(writer, callable, params)`.

**In process** (`InlineRunner.run`): builds a `DependencyTracker(owned)` when
tracing, then an `ExitStack` entering the tracker and
`storage.bind_job(writer)`, and calls `callable(**params)`. The tracker's
`__enter__` claims a `sys.monitoring` tool id, registers `_record` for
`PY_START`, and calls `restart_events()`.

**Out of process** (`SubprocessRunner.run`): submits
`execution.execute_job(str(writer.path), callable, params, owned, trace)` to a
spawned pool retiring workers after one task. In the child, `execute_job`
builds its *own* `MetricRecorder(path)` over the directory the parent staged,
runs the same tracker-plus-`bind_job` stack, and returns
`{"error", "metrics", "called"}`. Back in the parent, `JobWriter.adopt(metrics)`
takes on what the child registered.

Either way the result is an `execution.Executed` carrying the traceback (and,
in process, the exception itself), plus the callees.

## 6. Recording, from inside the body

```python
loss = Metric("loss", dtype=float)
loss.push(step, value)
```

1. `Metric.__init__` calls `storage.current_job()`, which reads the context
   variable `bind_job` set. `None` raises `NoActiveJob`.
2. It looks for an open sink in `recorder.sinks[name]`. On a miss it calls
   `recorder.metric_path(name, dtype_name(dtype))` — which *registers* the
   metric in the recorder's `_metrics` and returns the percent-encoded path —
   and constructs a `Sink`, which opens the file and writes the header.
3. `Metric.push` delegates to `Sink.push`: honour `skip_if_exists` against the
   seen steps, coerce through `coercion(dtype)`, `csv.writer.writerow`, then
   `flush`.

The recorder, not the metric, holds the sinks, so constructing the same
`Metric` twice inside one job appends to one file.

## 7. Finishing and promoting

1. `JobWriter.record_called(executed.called)` stores the callees, or `None`
   when untraced.
2. `JobWriter.finish(status, error)`:
   - `MetricRecorder.close()` closes every sink, takes each one's row count
     into the registry, and returns it;
   - the manifest gains `status`, `metrics` and the `finished`/`seconds`
     timings, and is written to `manifest.json`;
   - `deps.json` is written as `witnesses_of(manifest["environment"])` plus
     the environment itself, plus `called` when there is one — derived from
     the manifest so the two cannot disagree;
   - the `DONE` or `FAILED` marker is written, holding the traceback;
   - `JobStore.promote(writer)` removes any existing folder, then
     `os.replace`s the staging directory onto the final path and returns a
     `JobFolder`.

The marker is written *before* the rename, so the final path never holds a
job that is incomplete yet marked complete.

## 8. When it does not finish

- **The body raised.** `Executed.ok` is false. The job is finished as
  `"failed"` and promoted, a `JobOutcome` records it, the queue is **cleared**,
  and `RunFailed` is raised carrying the `RunSummary` — chained onto
  `Executed.exception` when there is one.
- **Interrupted** (`BaseException` out of `runner.run`, e.g.
  `KeyboardInterrupt`). `JobWriter.discard()` removes the staging directory and
  the exception propagates. Nothing is promoted, so the next pass reads the
  job as absent.
- **The child died.** `SubprocessRunner.run` catches `BrokenProcessPool`,
  closes the pool so the next job gets a fresh one, and reports it as that job
  failing.
- **The job could not be sent.** Any other exception from the submit means the
  callable or a parameter is not picklable; the runner raises `RuntimeError`
  naming the requirement, which stops the sweep.

## 9. Reading back

```python
frame = orch.get("loss")
```

1. `Orchestrator.get` calls `drain()` first, so a sweep need not be run
   explicitly.
2. `metrics.collect(store, name, self._enqueued or None)`.
3. `collect` maps the keys through `JobStore.folder_for` and keeps the folders
   where `JobFolder.done`; with no keys it falls back to `JobStore.done()`.
4. Per folder, `metrics.read_metric`: `JobFolder.metric_path(name)`, and
   `None` if absent; the CSV is parsed with `csv` and `parse_value`, which
   reads each field back as the dtype the manifest recorded rather than
   guessing; `storage.read_manifest_params`, which
   renders each encoded parameter through `display_value`; a second `reserved`
   check guards against a folder whose parameters would shadow the frame's own
   columns; then the parameter columns and `job_key` are attached.
5. `metrics.concat` stacks the tables, taking the union of their columns so
   that a parameter only some jobs carry reads as None for the rest, and
   keeping the `[step, value, *params, job_key]` order. No tables at all
   raises `KeyError` listing `JobStore.metric_names()`.

## One job, end to end

```
Orchestrator.run ──► bind_params ─► reserved ─► static_key ─► PendingRun
                                                  │
                                              (Hasher, Modules)

Orchestrator.get ──► drain ──► _refresh ─► runner(owned) ─┐
                                                  ▼
                        decide ──► JobFolder.status
                                   drift(witnesses)
                                   verify_called ─► Location.fetch, callee_key
                                                  │
                                      skip ◄──────┴──────► run
                                                             │
                        build_manifest ◄─────────────────────┘
                          (encode_params, git_provenance,
                           environment_fingerprint ─► find_lock, hash_file,
                                                      installed_packages)
                                                             │
                        JobStore.writer ─► JobWriter ─► MetricRecorder(create)
                                                             │
                        runner.run ─► bind_job + DependencyTracker
                                          │
                                          ▼  the user's function
                                     Metric ─► current_job ─► recorder.sinks
                                          └─► Sink.push ─► csv, flush
                                                             │
                        record_called ─► finish ─► close, manifest.json,
                                                   deps.json, marker
                                                             │
                                          JobStore.promote ─► os.replace
                                                             │
                        collect ─► read_metric ─► read_csv,
                                   read_manifest_params ─► concat
```

## The same thing, observed

Subscribing `sys.monitoring` to `PY_START` over a one-job sweep, with tracing
off and no lock file present, and keeping the first entry into each function:

```
 1 Orchestrator.__init__         36 build_manifest
 2 JobStore.__init__             37 encode_params
 3 Orchestrator.run              38 git_provenance
 4 bind_params                   39 environment_fingerprint
 5 reserved                      40 find_lock
 6 Orchestrator._owned_for       41 Orchestrator._warn_if_dirty
 7 owned_for                     42 JobStore.writer
 8 static_key                    43 JobWriter.__init__
 9 Hasher.__init__               44 MetricRecorder.__init__
10 Modules.__init__              45 InlineRunner.run
11 Hasher.push                   46 bind_job
12 Location.__init__             47 Metric.__init__
13 Hasher.push                   48 current_job
14 Location.__init__             49 MetricRecorder.metric_path
15 Hasher.push                   50 Sink.__init__
16 bind_params                   51 Metric.push
17 Hasher.push                   52 Sink.push
18 PendingRun.__init__           53 Executed.__init__
19 Orchestrator.get              54 JobWriter.record_called
20 Orchestrator.drain            55 JobWriter.finish
21 Orchestrator._owned_modules  56 MetricRecorder.close
22 Orchestrator._owned_for       57 Sink.close
23 Orchestrator._owned_for       58 witnesses_of
24 owned_for                     59 JobStore.promote
25 Orchestrator.runner           60 JobFolder.__init__
26 InlineRunner.__init__         61 JobOutcome.__init__
27 Orchestrator._drain           62 RunSummary.__init__
28 Orchestrator.decide           63 InlineRunner.close
29 JobFolder.__init__            64 collect
30 JobFolder.status              65 JobFolder.__init__
31 JobDecision.__init__          66 JobFolder.status
32 bind_params                   67 read_metric
33 Orchestrator._owned_for       68 JobFolder.metric_path
34 Orchestrator._owned_for       69 read_manifest_params
35 owned_for                     70 reserved
```

Two things this shows that reading the code does not. `bind_params` runs three
times for one job — once for the reserved-name check, once inside
`static_key`, once for `PendingRun.bound_params` — as does `_owned_for`, which
`run`, `_refresh`, `_owned_modules` and `build_manifest` each reach separately.
Both are cheap, and neither is memoised. `static_key` itself runs twice, once
at `run` and once at `_refresh`. And `decide` returned after
`JobFolder.status` here, the folder being absent, which is why
`Orchestrator.witnesses`, `drift` and `verify_called` do not appear: they are
reached only for a job that *is* recorded.

## Invariants to keep

- `decide` is the only place the re-run rule lives; `plan` and `drain` share
  it so that what is reported is what happens.
- The key is computed in `run`, not in `drain`, so a bad parameter or an
  unhashable dependency is reported before a long sweep starts.
- A marker is written before the rename, never after.
- `deps.json` is derived from the manifest's own `environment`, so the
  fingerprint cannot drift from what was recorded beside it.
- Sinks live on the recorder, so the same metric constructed twice appends.
- `MetricRecorder` knows nothing about promotion, which is what lets a child
  process own one.

# The problem

Currently, the orchestrator
- for each hyperparameter tuple value, run the user-defined (python) function
- metrics are collected in memory and not persisted
- execution is synchronous, mono-process and fail-fast (everything stops whenever 

Simulation job
- refers to one possible execution of the user-defined function on a given
  hyper parameter tuple value
- depends on: the hyper parameter value, and the user code being executed
- state: pending, in progress, done, or failed
- outputs: metric data

For the next iteration, we want (informally)
- the orchestrator to be smart about which simulation jobs need to be
  re-executed, i.e. avoid re-executing jobs that have succeeded
- guaranteeing reproducibility (running a job twice should yield the same
  output) and traceability (each output can be traced back to the code, e.g.
  git commit, that generated it, and the hyperparameter value)
- the whole orchestration still runs on a single machine
- the outputs (metric data) are stored on local disk
    - each job is assigned a unique folder that will contain its output

Particular care must be taken to identify a job. Job identity should be tied to
- (easy) the named tuple of hyper parameter values, user-defined function name
- (difficult) the actual user-defined code

E.g. the following actions should yield a distinct job identity
- changing a hyper parameter value
- changing a hyper parameter name
- changing the name of the user-defined function
- changing the body of the user-defined function
- changing an object that the user-defined function depends on
- changing the version of an underlying dependency, e.g. pytorch

# Solution

Identity is split in two, as decided in [adr-2026-07-31.md]: a static key that
names the job, and a fingerprint that says whether the stored output is still
valid. Capturing the exact runtime dependencies of a run is undecidable in
general (c.f. `exec(random())`), so we do not try. The first run records what
it depended on; later runs only re-validate that record.

## Two-level identity

- `job_key` — computed without running anything, from the parameters and the
  code. Names the job folder. This is the job's *question*.
- `fingerprint` — the `location -> content hash` manifest observed during the
  last successful run, stored inside the folder. This is the warrant that the
  stored *answer* is still valid.

The re-run decision is then:

```
key = static_key(callable, params)
dir = root / key
if not (dir / "DONE").exists():      run
elif fingerprint(dir) has drifted:   run, and record which entry drifted
else:                                skip
```

No static whole-program analysis is needed, and the key is computable without
executing the run, as required by [adr-2026-07-31.md].

## The static key

Hashing the parameters is already handled by `hashing.Hasher.push`. The
signature is first bound and normalized (`inspect.signature(...).bind`, then
sorted items) so that parameter *names* participate and defaults are explicit:
renaming `f` to `n_byz` yields a new identity.

Hashing the callable is the open case (`Hasher.push` currently raises on
anything with `__code__`). The recipe:

- `__module__` and `__qualname__` — catches renaming the function.
- `__code__`: `co_code`, `co_consts` (recursing into nested code objects),
  `co_names`, `co_varnames`, argument count and flags. Deliberately **not**
  `co_filename` nor `co_firstlineno`, so that adding a comment above the
  function, or moving it within its file, does not invalidate the job.
- `__defaults__`, `__kwdefaults__`, and the closure cells (`__closure__` ->
  `cell_contents`).
- the globals it actually references: `co_names` intersected with
  `__globals__`, pushed recursively with an `id()`-keyed `seen` set.

That last step is what makes transitive dependencies tractable: everything the
function reaches is reachable statically through its name table, so the
referent-graph walk currently sketched in `Dependencies.derive` is not needed.
Recursion stops at the third-party boundary — `Modules.__contains__` decides
*owned* (recurse into bytecode) versus *external* (hash as `Location` plus
distribution version only). Classes recurse over their methods' code, their
class attributes, and the `Location` of their bases.

## Environment

The environment hash lives in the fingerprint, not in the folder name. This is
a deliberate departure from the problem statement above, which lists a
dependency version bump as something that should change job *identity*:

- the folder name is the question (`Krum, n=10, f=2, seed=42`), which a pytorch
  bump does not change;
- the fingerprint is whether the stored answer is still trustworthy, which a
  pytorch bump does change.

Mechanically this still forces the re-run, but results are not orphaned into a
parallel directory tree on every patch bump, and `Orchestrator.get` can still
read across them. A `strict_env` knob can promote the environment hash into the
key if that turns out to be wrong.

The environment hash is the content hash of `uv.lock` — exact and cheap, and
better than sniffing `importlib.metadata` package by package — together with
the interpreter version and the `__debug__` flag that `Dependencies.__preinit__`
already folds in. A dirty git tree is recorded in the manifest and warned
about, not rejected.

## Capture during execution

Two hooks feed the content side of the fingerprint:

- `InterceptFinder` on `sys.meta_path`, for just-in-time imports that the
  static pass cannot see (an `import` inside a function body).
- `sys.monitoring` (PEP 669) subscribed to call events only. Line tracing a
  100-round simulation would be punishing; call-only is tolerable, and
  monitoring is markedly cheaper than `sys.settrace`. It needs Python 3.12,
  which is the project minimum, so there is no need to fall back on
  `settrace`.

  A callback returning `DISABLE` stops that function reporting again, so the
  cost is one callback per distinct function rather than one per call. That is
  cheap enough that tracing is **on by default** rather than kept behind an
  opt-in flag, as first planned: the alternative to paying for it is keeping a
  stale result, which is the outcome this design rules out. The flag remains,
  to be turned off for a sweep that must share monitoring with a debugger or a
  profiler.

Residual false negatives, both accepted per the false-positive/false-negative
trade-off in [adr-2026-07-31.md]:

- A dependency reached only through a branch not taken, changed, while the
  function body is unchanged.
- A module imported at runtime and read only for data, never called into. No
  function of it is entered, so call tracing does not see it. The meta path
  hook above is what would close this.

## Storage

```
<root>/<job_key>/
  manifest.json   params (readable), callable location, git commit + dirty,
                  uv.lock hash, timings
  deps.json       the fingerprint
  metrics/*.csv   append-only (step, value)
  DONE | FAILED   marker; FAILED carries the traceback
```

Job state is derived from marker presence rather than a mutable `status`
field, and the job is written into a temporary directory that is `os.replace`d
into place on success. A job killed mid-flight leaves no `DONE` and is simply
"not done" on the next pass: no stale-lock reasoning, and no folder that is
corrupt but marked complete.

## Metrics

`Metric` is a thin append-only writer bound to the running job's folder,
flushing incrementally so that a crashed run's partial data stays inspectable
without being promoted. `Orchestrator.get(name)` scans `DONE` folders, reads
each metric file, and joins each row with that job's parameters from
`manifest.json`, yielding one tidy frame with columns `[step, value, *params]` —
the shape [orchestration_example.py] already assumes. Parameters therefore need
a stable *readable* encoding (class -> qualified name) alongside the hash
encoding.

## Execution

Fail-fast is kept, and `run` stays enqueue-only with the queue drained on the
first `get` (as `Orchestrator._queue` and the "may be blocking" note on `get`
already intend), plus a `with Orchestrator(...)` form that drains on exit.

Each job runs in a **subprocess**. This is a reproducibility win independent of
parallelism — a fresh interpreter, no global or CUDA state leaking between
aggregators — and it is the seam a process pool slots into later without a
redesign.

## Sequencing

1. `Hasher.push` for functions and classes, with tests pinning the invariants:
   comment above the function -> same key; body change -> different key;
   parameter rename -> different key; helper function change -> different key.
2. Folder protocol, `Metric` writer, `get` read path. Persistence and
   traceability are done at this point.
3. Staleness check against `deps.json`, with a `force` escape hatch and a
   reason reported in the run summary.
4. Subprocess isolation, then tracing capture, then parallelism.

Dropped: the `Context` and `gc.get_referents` exploration. It is an open-ended
graph walk, and resolving `co_names` against `__globals__` reaches the same
objects statically, which is also what [adr-2026-07-31.md] requires.

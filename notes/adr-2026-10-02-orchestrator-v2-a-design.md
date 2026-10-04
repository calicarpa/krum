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

Built across `krum/orchestration`: `hashing` (keys), `storage` (folders and
fingerprints), `metrics` (recording and reading back), `tracing` (what a job
called), `execution` (where a job's body runs), and `__init__`
(`Orchestrator`).

## Two-level identity

- `job_key` — computed without running anything, from the parameters and the
  code. Names the job folder, as the 32 hex characters of a 16-byte digest.
  This is the job's *question*.
- `fingerprint` — written beside the output once a job completes, in
  `deps.json`. Two parts: `witnesses`, comparable facts about the environment,
  and `called`, the `location -> code hash` map of the owned functions the job
  entered. This is the warrant that the stored *answer* is still valid.

`Orchestrator.decide` returns both the decision and the reasons behind it, and
`Orchestrator.plan` reports them for a whole sweep without running any of it:

```
key = static_key(callable, params)
folder = root / key
no marker                  -> run ("no recorded result")
FAILED marker              -> run ("previous attempt failed")
forced                     -> run ("forced")
a witness differs          -> run ("uv_lock changed (651669ac047d… -> …)")
a recorded callee differs  -> run ("pkg.mod:Class.method changed")
otherwise                  -> skip
```

No static whole-program analysis is needed, and the key is computable without
executing the run, as required by [adr-2026-07-31.md].

## The static key

`hashing.static_key(callable, params, owned)`. The parameters are bound to the
signature and its defaults applied (`hashing.bind_params`), so that parameter
*names* participate and a default left unstated still identifies the job:
renaming `f` to `n_byz` yields a new identity, and the manifest records the
same bound parameters the key was computed from.

`Hasher.push` folds a callable in as:

- `__module__` and `__qualname__` — catches renaming the function.
- `__code__`: `co_code`, `co_consts` (recursing into nested code objects),
  `co_names`, `co_varnames`, `co_freevars`, `co_cellvars`, the argument counts
  and the flags. Deliberately **not** `co_filename`, `co_firstlineno` nor the
  line table, so that adding a comment above the function, or moving it within
  its file, does not invalidate the job.
- `__defaults__`, `__kwdefaults__`, and the closure cells (`__closure__` ->
  `cell_contents`).
- the globals it actually references: `co_names`, gathered through nested code
  objects as well, intersected with `__globals__`, pushed recursively with an
  `id()`-keyed `seen` set.

That last step is what makes transitive dependencies tractable: everything the
function reaches by name is reachable statically through its name table, so no
referent-graph walk is needed. Nested code has to be walked because a lambda or
an inner function resolves its globals against the *enclosing* function's
`__globals__`.

Docstrings live in `co_consts` and are kept rather than stripped: the
conservative direction is a needless re-run, never a stale result.

An owned module's own members are folded in too, which covers the
`import mymod; mymod.helper()` shape, where the dependency is reached by
attribute rather than by a name in the caller's globals.

### Where recursion stops

`Modules` decides ownership by dotted prefix. Owned code is folded in by
content; everything else stops at its `Location`. Dependency *versions* are not
hashed here at all — they belong to the fingerprint, keyed on `uv.lock` — which
is what keeps a hash of a user experiment from walking into pytorch.

Exclusions override the prefixes, which is what lets a broad prefix like `krum`
be owned while a subpackage within it is not. `krum.orchestration` is always
excluded: it *runs* a job rather than defining what the job computes, so
folding it in would change every key whenever the harness is edited, and would
drag in runtime state — the context variable naming the current job — that has
no reproducible hash.

Three further things are folded in by name rather than by content, being
machinery rather than content: `_abc_impl`, an identity-based `ABCMeta` cache;
`__firstlineno__`, which Python 3.13 adds to every class `__dict__` and which
is exactly the line provenance excluded everywhere else; and `ContextVar`,
which names a slot for runtime state.

Anything that cannot be hashed reproducibly raises `HashError` rather than
folding in a placeholder. Bytecode is not stable across interpreter versions,
so keys change on a Python upgrade; the interpreter is a witness too, so that
is visible rather than silent.

`hashing.shallow_key` hashes one callable's own code and nothing it refers to.
A full key moves when a helper does; a shallow key stays put, which is what
lets a re-run name the function that changed rather than only the job.

## Environment

The environment lives in the fingerprint, not in the folder name. This is a
deliberate departure from the problem statement above, which lists a dependency
version bump as something that should change job *identity*:

- the folder name is the question (`Krum, n=10, f=2, seed=42`), which a pytorch
  bump does not change;
- the fingerprint is whether the stored answer is still trustworthy, which a
  pytorch bump does change.

Mechanically this still forces the re-run, but results are not orphaned into a
parallel directory tree on every patch bump.

`storage.witnesses_of` reduces a recorded environment to what gets compared:

| witness          | compared as                    | why                                       |
|------------------|--------------------------------|-------------------------------------------|
| `uv_lock`        | content hash of the lock file  | the one witness a key cannot cover        |
| `python`         | major and minor only           | bytecode is stable across patch releases  |
| `implementation` | exactly                        |                                           |
| `debug`          | exactly                        | `-O` strips assertions                    |

`platform` is recorded for traceability but not compared: a different machine
is no reason to discard a result. The lock file hash is cached per path, size
and modification time, so a sweep reads the lock once rather than once per job.

Only `uv_lock` does work a key does not. Because a key hashes bytecode, a Python
minor upgrade or an `-O` change mostly shifts keys on its own; the other three
are kept because they cost nothing, and stay correct should key derivation ever
stop depending on bytecode. Third-party code, by contrast, is folded into a key
as a `Location` only, never by content, so a dependency bump is invisible to
the key by design — hence the lock file.

A job recorded without a fingerprint cannot be checked, which counts as stale.

A dirty git tree is recorded in the manifest, and warned about once per sweep,
rather than rejected.

## Capture during execution

`tracing.DependencyTracker` subscribes to `sys.monitoring` (PEP 669) call
events — `PY_START` — and records the owned functions a job entered. Line
tracing a 100-round simulation would be punishing; call-only is tolerable.

A callback returning `DISABLE` stops that function reporting again, so the cost
is one callback per distinct function rather than one per call. That is cheap
enough that tracing is **on by default** rather than kept behind an opt-in
flag, as first planned: the alternative to paying for it is keeping a stale
result, which is the outcome this design rules out. The flag remains, to be
turned off for a sweep that has to share monitoring with a debugger or a
profiler, those holding the other tool ids.

`DISABLE` is recorded against the code object and outlives the tracker that
asked for it, so events are restarted when a tracker starts. Without that, only
the first traced job of a process records anything and every later one silently
records nothing.

A callee is hashed through the object fetched back by name, not through the
code object seen running: the two differ when a decorator stands between them,
and it is the fetched object a later pass will re-hash, so both sides have to
hash the same thing. Locations are encoded `module:Qual.name`, a dotted form
not saying whether `a.b.C.d` lives in `a` or in `a.b`. Qualified names that
cannot be fetched back — nested functions, lambdas, comprehensions,
module-level code — are left out, there being nothing to re-hash later. An
untraced job records no `called` key at all, which is told apart from a traced
job that called nothing.

Residual false negatives, both accepted per the false-positive/false-negative
trade-off in [adr-2026-07-31.md]:

- A dependency reached only through a branch not taken, changed, while the
  function body is unchanged.
- A module imported at runtime and read only for data, never called into. No
  function of it is entered, so call tracing does not see it. A `sys.meta_path`
  hook intercepting imports is what would close this, and is **not built**.

## Storage

```
<root>/
  .staging/<job_key>.<pid>/   a job under construction
  <job_key>/
    manifest.json   job_key, callable location, params (readable), owned
                    prefixes, git commit + dirty + branch, environment,
                    timings, status, metrics (each with its dtype, file
                    and row count)
    deps.json       witnesses, environment, called
    metrics/*.csv   append-only (step, value), one per metric, its name
                    percent-encoded so that any name round-trips
    DONE | FAILED   marker; FAILED carries the traceback
```

Job state is read from the markers rather than from a mutable field, and a job
is built in a staging directory renamed into place only once its marker is
written. A job killed mid-flight therefore leaves no marker at the final path
and is simply "not done" on the next pass: no stale-lock reasoning, and no
folder that is incomplete yet marked complete.

A failed job is promoted too, carrying its traceback, so that it can be
inspected and so that a later pass reads it as failed rather than absent.
Promotion replaces any earlier attempt rather than merging into it; removing
the earlier one first is safe, since we only get there having decided to
re-run it.

## Metrics

`Metric` is a thin append-only writer that finds the running job through a
context variable, so that a user experiment never threads a handle through its
own call stack. Rows are flushed as they are pushed, so a crashed job's partial
output stays readable where it was staged, without ever being promoted to the
final path where the read path would pick it up. Values are coerced to the
declared dtype, which may be a Python type or a torch dtype.

`Orchestrator.get(name)` reads the jobs enqueued on that orchestrator, in that
order (falling back on the whole store when nothing was enqueued), joining each
row with the parameters that job's key was computed from. The result is one
tidy `MetricTable` with columns `[step, value, *params, job_key]` — the shape
[orchestration_example.py] already assumes. Parameters therefore need a stable
*readable* encoding (a class renders as its short name, its full location kept
alongside for tracing) next to the hash encoding.

Reading one sweep rather than every folder in the store matters: a store
accumulates a folder per version of an experiment, so reading all of them mixes
code versions, two of which can carry the same label and be drawn over each
other without saying so. `metrics.collect` takes a store directly for the rare
case where everything is wanted. The enqueued order matters too, plots grouping
with `sort=False` and assigning colours in sequence.

`step`, `value` and `job_key` are the table's own columns, so a parameter of
one of those names is refused when the run is enqueued: it could not be told
apart from the metric it was recorded against.

`MetricTable` is an interface, not a class to hold rows: `columns`, `__len__`
and `rows` are abstract, and everything else — the column accessor, the row
iteration, `to_dict`, `to_pandas`, `to_csv` — is derived from those three.
`InMemoryTable` is the in-memory implementation, and the one `collect` builds.
The split costs nothing today and is what keeps a later out-of-core
implementation from being a rewrite: holding the rows differently is a matter
of three members, and reading a metric back is already reading per-job CSVs
off disk. The row count each metric wrote is recorded in the manifest for the
same reason, so `len` need never read a file to answer.

## Execution

Fail-fast is kept, and `run` stays enqueue-only with the queue drained on the
first `get`, plus a `with Orchestrator(...)` form that drains on exit.

A run that fails stops the drain, and the queue is abandoned so that the
results which did complete stay readable — which they would not be if a later
`get` re-ran the failing job and re-raised. Re-enqueueing retries it, its
folder being marked failed rather than done. An interrupt is told apart from a
failure: it discards the staged directory rather than promoting a marker, so
the job runs again next time.

A job's body runs either in the orchestrator's own process (`InlineRunner`) or
in a freshly spawned child retired after that one job (`SubprocessRunner`).
Isolation is a reproducibility measure before it is a performance one: nothing
a job leaves behind can reach the next, so a sweep's results stop depending on
the order its jobs happened to run in.

It is opted into with `isolate=True` rather than assumed, because it costs two
requirements the in-process path does not have: the experiment and its
parameters must be picklable, hence reachable at module level, and a sweep
script's top level must be guarded by `if __name__ == "__main__":`, a spawned
child re-importing the module it came from.

A child that fails sends its traceback back, formatted where its frames still
exist. A child that dies outright fails that job rather than the sweep, and the
pool is discarded so that the next job starts from a fresh one. An experiment a
child cannot import stops the sweep instead, naming the requirement: every job
would hit it, so marking them all failed would be noise.

Raising the runner's worker count above one is all that now stands between this
and running a sweep in parallel.

## Python floor

`requires-python = ">=3.12"`, chosen rather than inherited.

One thing in the library genuinely needs it: `sys.monitoring`, and so the
dependency tracing built on it. Everything else below 3.12 is mechanical — the
`type` alias statements, the runtime `collections.abc.Buffer` isinstance,
`typing.Self`, `co_qualname`, and `max_tasks_per_child` on the process pool.
Going lower therefore means either giving tracing up below 3.12 or writing a
`sys.settrace` fallback, which has no per-code `DISABLE` and so fires on every
call rather than once per function — a cliff that stays silent until a long
sweep.

The `experiments` extra is separately held at 3.11 by numpy and pandas, which
both require it. Since metrics read back as a `MetricTable`, pandas is no
longer a dependency of the library at all, so that bound applies to the
analysis rather than to recording or reading.

What the floor costs in audience, from torch's own download share in October
2026: 3.12 and above is about 64%, 3.11 about 21%, 3.10 about 14%. So the
floor forgoes roughly a third of installs. Accepted, because 3.10 reaches
end of life on 31 October 2026 and its share had already fallen from 27% to
18% of the 3.10–3.12 band over the preceding six months; because download
counts are inflated by CI and container rebuilds rather than counting people;
and because anyone installing torch is managing an environment already, where
`uv python install 3.12` costs a line.

If this is ever reopened, 3.11 is the interesting one rather than 3.10: it is
the larger share, and only the `type` statements and `Buffer` stand in its
way. Its price is making tracing conditional on 3.12, which is a correctness
feature to give up for reach.

## Status

All four steps are built, with tests under `tests/orchestration`:

1. Keys for callables and classes — `hashing`.
2. Folder protocol, `Metric` writer, read path — `storage`, `metrics`.
3. Staleness against the fingerprint, with `force` and `plan` — `storage` and
   `Orchestrator`.
4. Subprocess isolation and call tracing — `execution`, `tracing`.

The three experiments under `experiments/` are driven by it.

Remaining, in rough order of value:

- The `sys.meta_path` hook, for the one runtime dependency call tracing cannot
  see.
- Parallelism: raise `SubprocessRunner`'s worker count and relax fail-fast.
- Pruning. A store keeps a folder per code version indefinitely and nothing
  collects those whose version is gone; only `prune_staging` exists.

Dropped: the `Context`, `Dependencies` and `InterceptFinder` sketches, deleted
once superseded. Resolving `co_names` against `__globals__` reaches the same
objects as the `gc.get_referents` walk they explored, statically, which is what
[adr-2026-07-31.md] requires.

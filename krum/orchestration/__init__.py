"""Run experiments over parameter ranges and collect their metrics.

The user writes an experiment as a single run, sweeps it with ordinary Python
loops, and reads the results back per metric.

Metric values are collected in memory and not persisted. Execution is
synchronous and fail-fast in this version; the near-term plan is multi-process,
one process per run.

Example::

    from krum.orchestration import Metric, Orchestrator

    def my_experiment(n, f, aggregator, attack, n_steps):
        simulation = KrumSimulation(n=n, f=f, aggregator=aggregator, attack=attack)
        loss = Metric("loss", dtype=float)
        for step in range(n_steps):
            simulation.step()
            loss.push(step, simulation.loss())

    orch = Orchestrator("byzantine_study")
    for n in [10, 20]:
        for f in [2, 3]:
            for aggregator in [Average, Krum, Bulyan]:
                for attack in [ALIEAttack, SignFlipAttack, None]:
                    orch.run(
                        my_experiment,
                        n=n, f=f, aggregator=aggregator, attack=attack,
                        n_steps=100,
                    )

    loss = orch.get("loss")               # MetricDataFrame
    krum_alie = loss.filter(aggregator=Krum, attack=ALIEAttack)  # narrowed MetricDataFrame
    frame = krum_alie.to_pandas()         # pandas.DataFrame for plotting/analysis
"""

import gc
import sys

from collections import deque as Deque
from hashlib import blake2b as Blake2b
from importlib.machinery import ModuleSpec
from pathlib import Path

def bytes_to_int(data: bytes, size: int = 16) -> int:
    value = 0
    for byte in data[:size]:
        value = value * 2**8 + byte
    return value

def playground_hash(root: object) -> int:
    # Hash about current interpreter
    b2b = Blake2b()
    b2b.update(sys.version.encode())
    b2b.update(b"\xfe" if __debug__ else b"\xff")
    hash = bytes_to_int(b2b.digest())
    # Process whole referent tree
    todo = Deque()
    todo.append(root)
    seen = set()
    origins = set()
    while True:
        # Pull next object to process
        try:
            obj = todo.popleft()
        except IndexError:
            break
        # Ensure objects are seen at most once
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        # Push sub-referents
        todo.extend(gc.get_referents(obj))
        # Skip if not a module specification
        if not isinstance(obj, ModuleSpec):
            continue
        # Ignore namespace/unknown specification
        origin = obj.origin
        if origin is None or not obj.has_location:
            continue
        # Recover actual location and open file
        origin = Path(origin)
        while True:
            try:
                fd = origin.open("rb")
                break
            except NotADirectoryError:
                # Handle ZIP container (e.g. /path/to/container.zip/package/submodule.py)
                origin = origin.parent
        # Hash module name and "content" together
        with fd:
            b2b = Blake2b()
            b2b.update(obj.name.encode())
            b2b.update(b"\x00")
            buf = memoryview(bytearray(65536))
            while True:
                read = fd.readinto(buf)
                if read == 0:
                    break
                b2b.update(buf[:read])
        # Update order-invariant hash
        hash ^= bytes_to_int(b2b.digest())
    # Forward resulting hash
    return hash

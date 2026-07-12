"""Run experiments over parameter ranges and collect their metrics.

The user writes an experiment as a single run, sweeps it with ordinary Python
loops, and reads the results back per metric.

Metric values are collected in memory and not persisted. Execution is
synchronous and fail-fast in this version; the near-term plan is multi-process,
one process per run.

Example::

    from krum.orchestration import Metric, Orchestrator
    from krum.primitives.aggregators.average import Average
    from krum.primitives.aggregators.krum import Krum
    from krum.primitives.aggregators.bulyan import Bulyan
    from krum.primitives.attacks.alie import ALIEAttack
    from krum.primitives.attacks.sign_flip import SignFlipAttack
    from krum.primitives.data_partitioners.iid import IidPartitioner
    from krum.simulations.centralised.krum_nips_2017 import KrumSimulation

    def my_experiment(n, f, aggregator, attack, seed):
        train_set, test_set = ...  # e.g. torchvision datasets
        worker_datasets = IidPartitioner.partition(train_set, n=n, seed=seed)
        simulation = KrumSimulation(
            model_cls=..., train_datasets=worker_datasets, test_set=test_set,
            aggregator=aggregator, attack=attack,
            n=n, f=f, rounds=100, batch_size=32, lr=0.1, seed=seed,
        )
        simulation.setup()
        loss = Metric("loss", dtype=float)
        for step in range(100):
            simulation.step()
            if step % 10 == 0:
                test_loss, _test_accuracy = simulation.evaluate()
                loss.push(step, test_loss)

    orch = Orchestrator("byzantine_study")
    for n, f in [(10, 2), (20, 3)]:
        for aggregator in [Average, Krum, Bulyan]:
            for attack in [ALIEAttack, SignFlipAttack]:
                orch.run(
                    my_experiment,
                    n=n, f=f, aggregator=aggregator, attack=attack, seed=42,
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

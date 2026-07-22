Troubleshooting
===============

Common issues when running Krum simulations, with likely causes and fixes.

``ValueError``: aggregator bound not satisfied
----------------------------------------------

Each aggregator has a requirement on ``n`` and ``f``:

.. code-block:: text

   ValueError: For aggregator MultiKrum: need n >= 2f + 3, got n=5, f=2

**Fix:** Increase ``n`` or decrease ``f``. The strictest bound among
built-in aggregators is Bulyan's (``n >= 4f + 3``); the loosest is
Median's (``f < n / 2``). See the docstring of each aggregator for its
exact bound.

Loss diverges (accuracy drops to chance)
----------------------------------------

The aggregator is not resilient enough for the attack.

**Verify:**
* Is the attack expected to be tolerated by your aggregator at your
  ``(n, f)``? For example, ``Average`` has no resilience -- any single
  Byzantine worker can break it.
* Are ``lr`` and ``batch_size`` appropriate for the model and dataset?
  Try the values from the reference paper.

.. code-block:: python

   # Compare with no-attack baseline
   sim = KrumSimulation(
       ...,
       aggregator=MultiKrum,
       attack=None,  # no attack
       n=10, f=0,    # no Byzantine workers
   )

Loss is NaN
-----------

The aggregated gradient contains extremely large values.

**Likely causes:**
* ``FullGradientNegationAttack`` with high ``kappa`` can produce NaN with
  some learning-rate schedules.
* ``GaussianAttack`` with large ``std`` (default 200.0) can push the
  model into unstable regions.
* Learning rate too high for the model/dataset.

**Fix:** Lower the learning rate, reduce attack magnitude (e.g.
``GaussianAttack(std=10.0)``), or cap gradient norms.

``StopIteration`` in decentralised simulation
---------------------------------------------

One of the per-worker data streams ran out of batches.

**Fix:** Use ``itertools.cycle`` or a ``DataLoader`` with an infinite
iterator:

.. code-block:: python

   from itertools import cycle

   loader = DataLoader(dataset, batch_size=64, shuffle=True)
   workers_data = [cycle(loader) for _ in range(n - f)]

Simulation is very slow
-----------------------

**Likely causes:**
* ``Brute`` aggregator enumerates all :math:`\binom{n}{n-f}` subsets --
  only feasible for ``n <= 8``.
* ``Bulyan`` has :math:`O(n^3 d)` complexity. For large ``n``, prefer
  ``Aksel`` which is :math:`O(nd)`.
* CPU training with large models. Move to GPU with ``device="cuda"``.
* ``n`` is very large and the aggregator computes pairwise distances.

Model accuracy is poor even without attack
------------------------------------------

**Check:**
* Dataset shapes: does the model's input dimension match the data?
* Loss function: does ``y`` have the right type (``int64`` for
  ``cross_entropy``)?
* Learning rate: too high causes divergence, too low stalls.
* Epochs: increase ``rounds``.
* Batch size: very small batches increase gradient variance.

Results differ between runs with the same seed
----------------------------------------------

If you set ``seed`` but still observe variance, ensure deterministic
behaviour:

.. code-block:: python

   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

``Orchestrator.get()`` returns an empty frame
---------------------------------------------

**Likely causes:**
* The metric name doesn't match any channel recorded during the run.
* The experiment function raised an exception before pushing any values.

Check available metrics:

.. code-block:: python

   print(orchestrator.metrics())
   frame = orch.get("test_accuracy").to_pandas()
   print(frame.shape)

No matching ``.grad`` tensors after ``zero_grad()``
---------------------------------------------------

PyTorch 2.11+ defaults to ``set_to_none=True``, which breaks the cached
flat view in the ``Model`` wrapper.

**Fix:** Use ``relink_gradients()`` after calling ``zero_grad()``:

.. code-block:: python

   optimizer.zero_grad()
   grads = model.relink_gradients()
   grads[:] = 0  # equivalent to zero_grad

Next steps
----------

* If you encounter an issue not listed here, check the
  :doc:`/reference/index` or open an issue on GitHub.
* :doc:`end_to_end` — a complete working example from data to export.

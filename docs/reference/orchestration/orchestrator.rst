Orchestrator
============

.. currentmodule:: krum.orchestration

The orchestrator drives a sweep: :meth:`Orchestrator.run` enqueues one run,
:meth:`Orchestrator.plan` reports what a drain would do without running it, and
:meth:`Orchestrator.get` reads a metric back across the sweep.

.. autoclass:: Orchestrator
   :members:
   :undoc-members:
   :show-inheritance:

Runs and decisions
------------------

.. autoclass:: PendingRun
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: JobDecision
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: owned_for

Outcomes
--------

.. autoclass:: JobOutcome
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RunSummary
   :members:
   :undoc-members:
   :show-inheritance:

.. autoexception:: RunFailed
   :members:
   :show-inheritance:

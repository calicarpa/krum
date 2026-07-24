Tutorials
=========

Step-by-step guides to help you get the most out of Krum.

Start with :doc:`using_aggregators_attacks` and
:doc:`working_with_models` to learn the basics. Next,
:doc:`implement_aggregator` and :doc:`implement_attack` show you how to
extend Krum. Then run full simulations with
:doc:`centralised_simulation_walkthrough` or
:doc:`decentralised_simulation_walkthrough`.
Finally, :doc:`structured_experiments` covers data collection and
analysis at scale.

Available Tutorials
-------------------

.. toctree::
   :maxdepth: 1

   using_aggregators_attacks
   working_with_models
   implement_aggregator
   implement_attack
   centralised_simulation_walkthrough
   decentralised_simulation_walkthrough
   structured_experiments

Detailed Tutorials
-----------------

.. list-table::
   :widths: 25 75

   * - :doc:`using_aggregators_attacks`
     - All built-in aggregation rules and attack strategies with a
       resilience table and a combined example.

       See :doc:`/reference/primitives/aggregators/index` and
       :doc:`/reference/primitives/attacks/index`.
   * - :doc:`working_with_models`
     - The ``Model`` wrapper for zero-copy flat tensor views plus the
       standard models from the literature (Krum NIPS 2017, MONNA ICML 2023).

       See :doc:`/reference/primitives/models/index`.
   * - :doc:`implement_aggregator`
     - Subclass ``Aggregator`` and implement ``aggregate``, with tests.
       Build from gradients-only to fully parameterised.

       See :doc:`/reference/primitives/aggregators/index`.
   * - :doc:`implement_attack`
     - Subclass ``Attack`` and implement ``generate``, with tests.

       See :doc:`/reference/primitives/attacks/index`.
   * - :doc:`centralised_simulation_walkthrough`
     - Parameter-server simulation with ``KrumSimulation``: MultiKrum +
       SignFlip on MNIST, dataset setup, training loop, baseline comparison.

       See :doc:`/reference/simulations/centralised/index` and
       :doc:`/reference/simulations/decentralised/index`.
   * - :doc:`decentralised_simulation_walkthrough`
     - Peer-to-peer simulations with ``DecentralisedSimulation`` and
       ``MonnaSimulation``: per-worker models, model mixing, Byzantine
       reach modes, custom data streams.

       See :doc:`/reference/simulations/decentralised/index`.
   * - :doc:`structured_experiments`
     - Collect metrics with ``Metric`` and ``Orchestrator``, analyse
       with filtering and plotting, run N×M benchmarks.

       See :doc:`/reference/orchestration/index` and
       :doc:`/reference/orchestration/metricdataframe`.


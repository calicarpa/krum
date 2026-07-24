Tutorials
=========

Step-by-step guides to help you get the most out of Krum.

If you are new to Krum, start with :doc:`using_aggregators_attacks` and
:doc:`working_with_models`. Next, learn how to extend Krum with
:doc:`implement_aggregator` and :doc:`implement_attack`, then move on to
:doc:`centralised_simulation_walkthrough` and
:doc:`decentralised_simulation_walkthrough`.
For structured data collection and analysis, see
:doc:`structured_experiments`.

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
     - All built-in aggregation rules and attack strategies, with a resilience
       table and a combined example.

       See also :doc:`/reference/primitives/aggregators/index` and
       :doc:`/reference/primitives/attacks/index`.
   * - :doc:`working_with_models`
     - The ``Model`` wrapper for zero-copy flat tensor views and the standard
       models from the literature (Krum NIPS 2017, MONNA ICML 2023).

       See also :doc:`/reference/primitives/models/index`.
   * - :doc:`implement_aggregator`
     - Write a custom aggregation rule by subclassing ``Aggregator`` and
       implementing ``aggregate``, with tests. Build up from gradients-only
       to fully parameterised rules.

       See also :doc:`/reference/primitives/aggregators/index`.
   * - :doc:`implement_attack`
     - Write a custom Byzantine attack by subclassing ``Attack`` and
       implementing ``generate``, with tests.

       See also :doc:`/reference/primitives/attacks/index`.
   * - :doc:`centralised_simulation_walkthrough`
     - A complete simulation with the ``KrumSimulation``: MultiKrum + SignFlip on
       MNIST, dataset setup, training loop, and baseline comparison.

       See also :doc:`/reference/simulations/centralised/index` and
       :doc:`/reference/simulations/decentralised/index`.
   * - :doc:`decentralised_simulation_walkthrough`
     - Peer-to-peer simulations with ``DecentralisedSimulation`` and
       ``MonnaSimulation``: per-worker models, model mixing, Byzantine
       reach modes, and custom data streams.

       See also :doc:`/reference/simulations/decentralised/index`.
   * - :doc:`structured_experiments`
     - Collecting metrics with ``Metric`` and ``Orchestrator``, analysing
       results with filtering and plotting, and running N×M systematic
       benchmarks with comparison tables.

       See also :doc:`/reference/orchestration/index` and
       :doc:`/reference/orchestration/metricdataframe`.


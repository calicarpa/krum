Tutorials
=========

Step-by-step guides to help you get the most out of Krum.

If you are new to Krum, start with :doc:`using_aggregators_attacks` and
:doc:`working_with_models`, then move on to
:doc:`centralised_simulation_walkthrough` and
:doc:`decentralised_simulation_walkthrough`. The last three tutorials show
you how to extend Krum with custom logic.

Available Tutorials
-------------------

.. toctree::
   :maxdepth: 1

   using_aggregators_attacks
   working_with_models
   centralised_simulation_walkthrough
   decentralised_simulation_walkthrough
   custom_dataset
   working_with_orchestrator
   results_analysis
   systematic_benchmark
   implement_simulation
   implement_aggregator
   implement_attack

Detailed Tutorials
------------------

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
   * - :doc:`decentralised_simulation_walkthrough`
     - Peer-to-peer simulations with ``DecentralisedSimulation`` and
       ``MonnaSimulation``: per-worker models, model mixing, Byzantine
       reach modes, and custom data streams.

       See also :doc:`/reference/simulations/decentralised/index`.
   * - :doc:`custom_dataset`
     - Using any PyTorch ``Dataset`` with Krum simulations:
       ``TensorDataset``, ``torchvision`` datasets, loading from
       disk, and matching the model's input and output dimensions.

       See also :doc:`/reference/simulations/centralised/index`.
   * - :doc:`centralised_simulation_walkthrough`
     - A complete simulation with the ``KrumSimulation``: MultiKrum + SignFlip on
       MNIST, dataset setup, training loop, and baseline comparison.

       See also :doc:`/reference/simulations/centralised/index` and
       :doc:`/reference/simulations/decentralised/index`.
   * - :doc:`working_with_orchestrator`
     - Collecting structured results with ``Metric`` and ``Orchestrator``:
       running multiple configurations, filtering, and exporting to DataFrames.

       See also :doc:`/reference/orchestration/index`.
   * - :doc:`results_analysis`
     - Filtering, merging, pivoting, exporting, and plotting
       ``MetricDataFrame`` results. Aggregating across seeds and
       comparing multiple metrics on shared axes.

       See also :doc:`/reference/orchestration/metricdataframe`.
   * - :doc:`systematic_benchmark`
     - Running N aggregators × M attacks on a shared dataset with
       ``Orchestrator``, building a comparison table, interpreting
       results, and exporting for papers.

        See also :doc:`/reference/orchestration/index`.
   * - :doc:`implement_simulation`
     - Creating a custom simulation by subclassing
       ``CentralisedSimulation`` or ``DecentralisedSimulation``: custom
       evaluation, local update, and communication topology.

       See also :doc:`/reference/simulations/centralised/index` and
       :doc:`/reference/simulations/decentralised/index`.
   * - :doc:`implement_aggregator`
     - Write a custom aggregation rule by subclassing ``Aggregator`` and
       implementing ``aggregate``, with tests.

       See also :doc:`/reference/primitives/aggregators/index`.
   * - :doc:`implement_attack`
     - Write a custom Byzantine attack by subclassing ``Attack`` and
       implementing ``generate``, with tests.

       See also :doc:`/reference/primitives/attacks/index`.

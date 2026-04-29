How-to guides
=============

How-to guides are goal-oriented recipes. Each one answers a specific question
and provides a sequence of steps to achieve a practical result.

Unlike tutorials, how-to guides assume you already know the basics and want to
get something done quickly.

Extending the framework
-----------------------

.. toctree::
   :hidden:
   :maxdepth: 1

   add-aggregator
   add-attack
   add-model
   add-dataset
   add-custom-model
   add-custom-dataset

:doc:`add-aggregator`
   Create and register a new Byzantine-resilient gradient aggregation rule.

:doc:`add-attack`
   Create and register a new Byzantine attack strategy.

:doc:`add-model`
   Add a new model architecture to the experiments module.

:doc:`add-dataset`
   Add a new dataset loader to the experiments module.

:doc:`add-custom-model`
   Register an external model via symlinks and the auto-discovery system.

:doc:`add-custom-dataset`
   Register an external dataset via symlinks and the auto-discovery system.

Running experiments
-------------------

.. toctree::
   :hidden:
   :maxdepth: 1

   enable-native
   run-experiment-campaign

:doc:`enable-native`
   Switch from Python to the C++/CUDA accelerated variants of aggregators
   and attacks.

:doc:`run-experiment-campaign`
   Launch a batch of experiments with different hyper-parameters and collect
   results automatically.

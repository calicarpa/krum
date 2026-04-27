Attacks
=======

Krum provides several Byzantine attack strategies. Each attack follows a
keyword-only contract and includes parameter validation.

Overview
--------

.. list-table:: Attacks
   :header-rows: 1
   :widths: 20 42 20 18

   * - Attack
     - Description
     - Complexity
     - Target
   * - NaN
     - Generates gradients with NaN values
     - :math:`\mathcal{O}(d)`
     - Simple aggregators
   * - Identical (Bulyan)
     - Uses ones vector as attack direction
     - :math:`\mathcal{O}(nd)`
     - Multi-Krum based
   * - Identical (Empire)
     - Uses negative average as direction
     - :math:`\mathcal{O}(nd)`
     - Inner product based
   * - Identical (Little)
     - Uses std dev as attack direction
     - :math:`\mathcal{O}(nd)`
     - Distance-based

where:

- :math:`n` = total number of workers
- :math:`f` = number of Byzantine workers
- :math:`d` = gradient dimension

Quick Start
-----------

All attacks can be called with the same keyword-only interface:

.. code-block:: python

    import attacks

    # Using NaN attack
    byzantine = attacks.nan(
        grad_honests=honest_grads,
        f_decl=1,
        f_real=1,
        model=model
    )

    # Using identical (Little) attack
    byzantine = attacks.little(
        grad_honests=honest_grads,
        f_decl=1,
        f_real=1,
        defense=aggregator,
        model=model,
        factor=1.5
    )

    # Checking validity before calling
    if attacks.nan.check(grad_honests=honest_grads, f_real=1) is None:
        byzantine = attacks.nan(grad_honests=honest_grads, f_real=1)

Available Attacks
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Byzantine Attack Strategies:

   classes/identical
   classes/nan

API Reference
-------------

.. automodule:: attacks
   :members:
   :show-inheritance:

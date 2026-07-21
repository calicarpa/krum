Understanding Byzantine attacks
===============================

Distributed learning accelerates training by spreading the workload across
multiple workers. But what if some workers are compromised, faulty, or
adversarial? This is the **Byzantine fault model**, and it is the standard
threat model for robust distributed SGD.

This tutorial covers the threat model, the Byzantine budget, and the
attacks implemented in Krum — their strategies, assumptions, and effects.

The threat model
----------------

In Krum, a **Byzantine worker** is one that sends arbitrary or adversarial
gradients instead of the true gradient computed on its local data. The
standard setup is:

* **n** workers, of which up to **f** may be Byzantine.
* Honest workers compute unbiased stochastic gradients on their own shard.
* Byzantine workers collude and see all honest gradients for the current
  round before crafting their attack.
* The simulation runs synchronously: every round, ``n`` gradients arrive
  and the aggregator must produce a single update.

The threat model is **Byzantine** (from Lamport, Shostak, and Pease, 1982)
— not merely crash-fault or omission-fault. Byzantine workers can send
anything: zeros, random noise, the opposite gradient, or carefully crafted
vectors designed to evade detection.

The Byzantine budget
--------------------

The number of Byzantine workers ``f`` determines which aggregation rules
can tolerate them. Three regimes appear across Krum's built-in
aggregators:

.. list-table::
   :widths: 15 35 50

   * - Bound
     - Implication
     - Aggregators
   * - ``f < n / 2``
     - The honest workers hold the majority. The most permissive bound.
     - ``Average``, ``Median``, ``GeoMed``, ``TrimmedMean``, ``Aksel``,
       ``Brute``
   * - ``2f + 2 < n``
     - At least two honest gradients must agree with each other more than
       with any Byzantine one.
     - ``Krum``, ``MultiKrum``
   * - ``4f + 2 < n``
     - Byzantine workers must be outnumbered by honest workers at least
       4-to-1. Required by the strongest defensive stacking.
     - ``Bulyan``

**Intuition.** Every Byzantine worker can send at most one gradient vector.
If ``f >= n/2``, the Byzantine participants can outvote the honest ones
in any distance-based or majority-based rule. The stricter bounds for
Krum and Bulyan reflect their geometric strategy: they compare every
gradient against every other and discard the outliers. More comparison
pairs means a smaller tolerable Byzantine fraction.

Attacks by strategy
-------------------

Krum implements five attacks. They differ in knowledge assumptions,
adaptivity, and the kind of damage they inflict.

.. list-table::
   :widths: 15 25 25 35

   * - Attack
     - Strategy
     - Knowledge
     - Effect
   * - ``GaussianAttack``
     - Isotropic noise,
       independent of honest gradients.
     - None (ignores
       honest data).
     - Adds extreme outliers in every coordinate. Weak against
       distance-based rules; a pure-noise baseline.
   * - ``SignFlipAttack``
     - Sign-reversed
       honest mean,
       scaled by a factor.
     - Honest mean.
     - Points the update in the opposite direction. Collapses
       ``Average``. Simple but detected by robust rules because all
       Byz workers send the same outlier.
   * - ``ALIEAttack``
     - Honest mean
       shifted by
       ``z`` standard
       deviations.
     - Honest mean
       and variance.
     - **Stealth attack.** Stays within the natural variance of the
       honest gradients. ``z = "max"`` auto-tunes the shift to
       the worker configuration. Defeats ``Median``.
   * - ``FullGradient-
       NegationAttack``
     - Negated
       full-dataset
       gradient,
       scaled by ``kappa``.
     - Full dataset
       gradient
       (omniscient).
     - The strongest theoretical attack. Requires the aggregator to
       withstand an adversary that knows the true update direction
       and amplifies its opposite arbitrarily.
   * - ``SmallPerturb-
       ationAttack``
     - Boundary search
       on one coordinate
       to maximally
       perturb while
       staying "selected"
       by the target
       aggregator.
     - Honest gradients
       and target
       aggregator
       (adaptive).
     - **Targeted attack.** Exploits the curse of dimensionality:
       honest gradients disagree by :math:`\Theta(\sqrt{d})` in
       :math:`\ell_2` norm when :math:`d \gg 1`, so a single
       coordinate can be poisoned without raising suspicion.
       Defeats ``Krum`` and ``MultiKrum`` by staying inside their
       selection window.

Each attack class takes ``honest_gradients`` and ``f`` and returns
``f`` Byzantine gradient vectors. The simulation concatenates them with
the honest ones before passing to the aggregator.

How attacks affect learning
---------------------------

When an attack succeeds, the aggregated gradient is biased away from the
true descent direction. The model drifts, converges to a poor solution,
or diverges entirely.

.. list-table::
   :widths: 15 30 55

   * - Attack
     - Bias mechanism
     - Observable symptom
   * - ``GaussianAttack``
     - Large random perturbation.
     - Loss stagnates or climbs.
       Accuracy stays at chance level.
   * - ``SignFlipAttack``
     - Update points opposite
       to the true gradient.
     - Loss diverges; accuracy drops
       below chance.
   * - ``ALIEAttack``
     - Small per-coordinate
       drift, hard to detect.
     - Slow degradation over many
       rounds not sharp divergence.
   * - ``FullGradient-
       NegationAttack``
     - Strong pull away from
       the optimum.
     - Immediate divergence,
       often to NaN.
   * - ``SmallPerturb-
       ationAttack``
     - One coordinate poisoned
       while the aggregator
       selects the malicious worker.
     - Converges to a poor
       local minimum; accuracy
       is noticeably below the
       no-attack baseline.

The limits of robustness
------------------------

No aggregation rule is universally robust. Each rule assumes a specific
bound on ``f``, and most assume the honest gradients are identically
distributed — an assumption that can be violated by data heterogeneity
even without attacks.

The **curse of dimensionality** (el Mhamdi, Guerraoui, Rouault, ICML 2018)
is a fundamental limitation: two honest gradients in :math:`d` dimensions
disagree by :math:`\Theta(\sqrt{d})` in :math:`\ell_2` norm. An adaptive
attacker can hide a poisoned gradient inside this natural disagreement
window, as ``SmallPerturbationAttack`` demonstrates.

Because of this, robustness guarantees are always paired with assumptions:
the number of Byzantine workers, the dimension, the data distribution, and
whether the attacker adapts to the aggregation rule. Changing any of
these can break a guarantee.

Next steps
----------

* :doc:`systematic_benchmark` — run multiple aggregators against
  multiple attacks and compare results empirically.
* :doc:`using_aggregators_attacks` — the full list of built-in
  aggregators and attacks with their API.
* :doc:`centralised_simulation_walkthrough` — a complete training
  loop with ``KrumSimulation`` under attack.

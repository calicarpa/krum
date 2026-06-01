"""KrumSimulation — NIPS 2017 protocol (Blanchard et al.).

Reproduces the parameter-server distributed SGD experiments from:

    Blanchard, Peva, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer.
    "Machine learning with adversaries: Byzantine tolerant gradient descent."
    In Advances in Neural Information Processing Systems 30 (NIPS 2017).

One ``KrumSimulation`` instance = one (aggregator, attack, dataset, model)
configuration run over multiple synchronous rounds with no learning rate decay.
"""

from typing import Any

from krum.simulations.centralised import CentralisedSimulation


class KrumSimulation(CentralisedSimulation):
    """Distributed SGD simulation with Byzantine workers — NIPS 2017 protocol.

    Compared to the ICML 2018
    :class:`~krum.simulations.hidden-vulnerability-icml-2018.simulation.Simulation`,
    this variant:

    - Uses a **fixed learning rate** (no scheduler; ``lr_decay=None``, the
      default inherited from :class:`~krum.simulations.centralised.CentralisedSimulation`).
    - Reports a single **misclassification error rate** on the test set
      instead of the full ``(train_loss, test_accuracy, test_loss)`` triple.

    References:
        Blanchard, Peva, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien
        Stainer. "Machine learning with adversaries: Byzantine tolerant
        gradient descent." NIPS 2017.

    See Also:
        :class:`~krum.simulations.centralised.CentralisedSimulation`
            for the full constructor parameter list.
    """

    def evaluate(self) -> float:
        """Compute misclassification error rate on the test set.

        Returns:
            Error rate in :math:`[0, 1]`.
        """
        return self.evaluate_test_error()

    def _log_round(self, t: int, result: float) -> None:
        if self.label:
            print(f"[{self.label}] round {t:3d}  error={result:.4f}")

    def _save_traces(self, traces: list[tuple[int, Any]]) -> None:
        self._save_pt({"errors": traces, "label": self.label, "seed": self.seed})

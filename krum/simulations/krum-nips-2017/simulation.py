"""KrumSimulation — NIPS 2017 protocol (Blanchard et al.).

Reproduces the parameter-server distributed SGD experiments from:

    Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer.
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

    See Also:
        :class:`~krum.simulations.centralised.CentralisedSimulation`
            for the full constructor parameter list.
    """

    def evaluate(self) -> tuple[float, float]:
        """Compute misclassification error and cross-entropy loss on the test set.

        Returns:
            Tuple of ``(error, loss)``. Error is the misclassification rate
            in :math:`[0, 1]`; loss is the cross-entropy on the test set.
        """
        return self.evaluate_test_error_and_loss()

    def _log_round(self, t: int, result: tuple[float, float]) -> None:
        error, loss = result
        if self.label:
            print(f"[{self.label}] round {t:3d}  error={error:.4f}  loss={loss:.4f}")

    def _save_traces(self, traces: list[tuple[int, Any]]) -> None:
        # traces are (round, error, loss)
        self._save_pt({"traces": traces, "label": self.label, "seed": self.seed})

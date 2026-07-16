"""KrumSimulation.

Reference:
    Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer.
    "Machine learning with adversaries: Byzantine tolerant gradient descent."
    In Advances in Neural Information Processing Systems 30 (NIPS 2017).

One ``KrumSimulation`` instance = one (aggregator, attack, dataset, model)
configuration run over multiple synchronous rounds with no learning rate decay.
"""

from typing import Any

import torch

from ...simulations.centralised import CentralisedSimulation


class KrumSimulation(CentralisedSimulation):
    """Distributed SGD simulation.

    Compared to the ICML 2018
    :class:`~krum.simulations.centralised.HiddenVulnerabilitySimulation`,
    this variant:

    - Uses a **fixed learning rate** (no scheduler; ``lr_decay=None``, the
      default inherited from :class:`~krum.simulations.centralised.CentralisedSimulation`).
    - Reports misclassification error and cross-entropy loss on the test set.
    """

    def __init__(self, **kwargs: Any) -> None:
        """See :class:`~krum.simulations.centralised.CentralisedSimulation`."""
        kwargs.setdefault("lr_schedule", "none")
        super().__init__(**kwargs)

    def evaluate(self) -> tuple[float, float]:
        """Evaluate the model on the test set.

        Returns:
            Tuple of ``(test_loss, test_accuracy)``.
        """
        assert self._model is not None and self._test_loader is not None
        self._model.module.eval()
        with torch.no_grad():
            x, y = next(iter(self._test_loader))
            x, y = x.to(self.device), y.to(self.device)
            logits = self._model.module(x)
            loss = self.loss_fn(logits, y).item()
            preds = logits.argmax(dim=1)
            accuracy = (preds == y).float().mean().item()
        return loss, accuracy

    def evaluate_train(self) -> float:
        """Evaluate the model on the training set.

        Returns:
            Training loss.
        """
        assert self._model is not None and self._full_loader is not None
        self._model.module.eval()
        with torch.no_grad():
            x, y = next(iter(self._full_loader))
            x, y = x.to(self.device), y.to(self.device)
            logits = self._model.module(x)
            return self.loss_fn(logits, y).item()

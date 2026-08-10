"""How the layers get visited --- the layer loop as a swappable component.

:class:`SequentialStrategy` is the default: layers ``1..N``, once, forward
only. It exists as its own class rather than as a ``for`` loop inside the
Learner because the interesting research variants are *orderings*, not new
pipelines --- revisiting earlier layers once later ones exist, fitting a
subset, or stopping on a criterion. Those are new strategies against the same
:meth:`~spifil.learner.Learner.fit_layer`, not Learner rewrites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import cycle: the Learner owns a strategy
    from spifil.learner import Learner

__all__ = ["FitStrategy", "SequentialStrategy"]


@runtime_checkable
class FitStrategy(Protocol):
    """Drives the layer loop of a :class:`~spifil.learner.Learner`."""

    def run(self, learn: Learner) -> None:
        """Fit the layers, in whatever order this strategy prescribes."""
        ...


class SequentialStrategy:
    """Layers ``1..N`` once, in order.

    Layer ``L`` consumes the ranks the scoring pass after layer ``L-1``
    produced, so forward order is the only order in which every layer's
    input exists.
    """

    def run(self, learn: Learner) -> None:
        """Fit every layer once, low to high."""
        for layer in range(1, learn.n_layers + 1):
            learn.fit_layer(layer)

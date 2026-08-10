"""Filter-count allocation.

A :class:`FilterAllocator` turns a layer's target ``out_channels`` into a
per-class filter budget. :class:`UniformAllocator`, the default, splits the
budget equally by integer division, which means a layer produces fewer filters
than requested when ``out_channels`` is not a multiple of the class count.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from spifil.config import LayerSpec
from spifil.types import PatchSet

__all__ = ["FilterAllocator", "UniformAllocator"]


@runtime_checkable
class FilterAllocator(Protocol):
    """Turns a layer's target filter count into a per-class budget."""

    def __call__(
        self, layer: int, spec: LayerSpec, patches: PatchSet, scores: Tensor
    ) -> dict[int, int]:
        """Class -> filter count for this layer."""
        ...


class UniformAllocator:
    """Equal filter budget per class: ``out_channels // n_classes``.

    When ``out_channels`` does not divide evenly by the class count, the layer
    produces ``per_class * n_classes < out_channels`` filters and the
    remainder is dropped --- so ``7`` filters over ``2`` classes gives ``6``.
    The built layer records what it actually holds, so the stack stays
    consistent; pick ``out_channels`` as a multiple of the class count to get
    the exact number you asked for.

    ``patches`` and ``scores`` go unused here; they exist for protocol
    symmetry with allocators that budget by class difficulty instead.
    """

    def __call__(
        self, layer: int, spec: LayerSpec, patches: PatchSet, scores: Tensor
    ) -> dict[int, int]:
        classes = torch.unique(patches.labels).tolist()
        if not classes:
            return {}
        per_class = spec.out_channels // len(classes)
        return {cls: per_class for cls in sorted(classes)}

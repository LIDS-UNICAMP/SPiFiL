"""Protocols for the two swappable halves of layer 0.

Superpixel segmentation and seed extraction are separate concerns: any
algorithm that partitions the image into labeled regions can feed any rule for
picking a region's representative pixel.

Both protocols speak numpy, not torch --- the segmentation libraries do, and
layer 0 is the boundary where that conversion belongs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = ["SeedExtractor", "SuperpixelAlgorithm"]


@runtime_checkable
class SuperpixelAlgorithm(Protocol):
    """Partitions a multiband feature map into labeled regions."""

    def __call__(
        self,
        features: npt.NDArray[np.float32],
        mask: npt.NDArray[np.int32] | None,
        n_superpixels: int,
    ) -> npt.NDArray[np.int32]:
        """Segment ``features`` into at most ``n_superpixels`` regions.

        Parameters
        ----------
        features
            ``(C, H, W)`` float32 multiband image — the layer-0 color
            features, *not* the raw RGB.
        mask
            ``(H, W)`` region of interest; nonzero means inside. ``None``
            segments the whole image.
        n_superpixels
            Target number of regions.

        Returns
        -------
        ``(H, W)`` int32 labels. Region ids run from 1; ``0`` marks pixels
        outside the mask and belongs to no region.
        """
        ...


@runtime_checkable
class SeedExtractor(Protocol):
    """Picks one representative pixel per superpixel."""

    def __call__(
        self,
        labels: npt.NDArray[np.int32],
        features: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.int64]:
        """Reduce each region of ``labels`` to a single coordinate.

        Parameters
        ----------
        labels
            ``(H, W)`` int32 region ids as returned by a
            :class:`SuperpixelAlgorithm`; ``0`` is ignored.
        features
            ``(C, H, W)`` multiband features, for extractors that choose by
            appearance rather than geometry. Geometric extractors ignore it.

        Returns
        -------
        ``(n, D)`` int64 coordinates, slowest spatial axis first — ``(y, x)``
        for 2D. Order follows ascending region id.
        """
        ...

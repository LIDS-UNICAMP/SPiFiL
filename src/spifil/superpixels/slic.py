"""SLIC superpixels — the dependency-free default.

SLIC needs nothing beyond scikit-image, so ``pip install spifil`` is enough to
run the whole pipeline. It is *not* the algorithm the published SPiFiL results
use: those use :class:`~spifil.superpixels.disf.DISF`, which partitions
differently and gives different filters. Install the ``disf`` extra to
reproduce them.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from skimage.segmentation import slic as skimage_slic

__all__ = ["SLIC"]


class SLIC:
    """Simple Linear Iterative Clustering via scikit-image.

    Parameters
    ----------
    compactness
        Trades color proximity against spatial proximity. Higher values give
        squarer, more regular superpixels.
    sigma
        Width of the Gaussian smoothing applied before segmentation.

    Notes
    -----
    No seed parameter: skimage's SLIC initializes its cluster centers on a
    regular grid rather than randomly, so it is already deterministic.
    """

    def __init__(self, compactness: float = 10.0, sigma: float = 0.0) -> None:
        self.compactness = compactness
        self.sigma = sigma

    def __call__(
        self,
        features: npt.NDArray[np.float32],
        mask: npt.NDArray[np.int32] | None,
        n_superpixels: int,
    ) -> npt.NDArray[np.int32]:
        if features.ndim != 3:
            raise ValueError(f"features must be (C, H, W); got {features.shape}")

        labels = skimage_slic(
            np.ascontiguousarray(np.transpose(features, (1, 2, 0))),
            n_segments=n_superpixels,
            compactness=self.compactness,
            sigma=self.sigma,
            mask=None if mask is None else mask != 0,
            channel_axis=-1,
            start_label=1,
            # The input is already a color-transformed feature map; skimage
            # would otherwise treat 3-channel input as sRGB and convert it to
            # Lab a second time.
            convert2lab=False,
        )
        # skimage already emits 0 outside a mask; make that explicit either way.
        labels = np.asarray(labels, dtype=np.int32)
        if mask is not None:
            labels[mask == 0] = 0
        return np.ascontiguousarray(labels)

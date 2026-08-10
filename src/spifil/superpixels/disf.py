"""DISF superpixels — the algorithm the published SPiFiL results use.

Dynamic and Iterative Spanning Forest. There is no independent Python
implementation, so this wraps the authors' one through PyIFT rather than
risking a reimplementation bug in the component every later stage depends on.

PyIFT is an **optional** dependency, shipped as a prebuilt wheel::

    uv sync --group disf                              # with uv
    pip install vendor/pyift-*-linux_x86_64.whl       # with pip

Everything except DISF and :class:`~spifil.superpixels.centers.GeodesicCenters`
works without it, and :class:`~spifil.superpixels.slic.SLIC` is the
dependency-free alternative. The import happens lazily, here.

The ``pyift`` distribution on PyPI is an unrelated project of the same name
and does not provide DISF; use the vendored wheel.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["DISF", "require_pyift"]

_PYIFT_HINT = (
    "DISF requires PyIFT, which ships as a prebuilt wheel in the "
    "repository's vendor/ directory.\n"
    "  Install it:      uv sync --group disf\n"
    "                   (or: pip install vendor/pyift-*-linux_x86_64.whl)\n"
    "  It also needs:   sudo apt-get install -y liblapack3 libblas3\n"
    "  The wheel is built for CPython 3.12 on linux-x86_64. Elsewhere, use\n"
    "  spifil.superpixels.SLIC: no extra dependencies, but it does not\n"
    "  reproduce the published results."
)


def require_pyift() -> Any:
    """Import and return ``pyift.pyift``, with an actionable error if absent."""
    try:
        import pyift.pyift as ift
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_PYIFT_HINT) from exc
    return ift


class DISF:
    """Dynamic and Iterative Spanning Forest superpixels via PyIFT.

    Parameters
    ----------
    n_init_seeds
        Initial seed count DISF starts from and prunes down. ``None`` (the
        default) uses ``10 x n_superpixels``.
    adjacency_radius
        Radius of the circular adjacency relation. Keep the default of
        ``1.0`` (4-connectivity) unless you have a reason not to: at
        ``sqrt(2)`` (8-connectivity) DISF leaks superpixels across object
        boundaries.
    """

    def __init__(
        self, n_init_seeds: int | None = None, adjacency_radius: float = 1.0
    ) -> None:
        self.n_init_seeds = n_init_seeds
        self.adjacency_radius = adjacency_radius

    def __call__(
        self,
        features: npt.NDArray[np.float32],
        mask: npt.NDArray[np.int32] | None,
        n_superpixels: int,
    ) -> npt.NDArray[np.int32]:
        ift = require_pyift()

        if features.ndim != 3:
            raise ValueError(f"features must be (C, H, W); got {features.shape}")
        _, height, width = features.shape

        # PyIFT's MImage constructor takes channels-last.
        mimg = ift.CreateMImageFromNumPy(
            np.ascontiguousarray(np.transpose(features, (1, 2, 0)), dtype=np.float32)
        )

        if mask is None:
            ift_mask = None
        else:
            if mask.shape != (height, width):
                raise ValueError(
                    f"mask shape {mask.shape} does not match features {(height, width)}"
                )
            ift_mask = ift.CreateImageFromNumPy(
                np.ascontiguousarray(mask, dtype=np.int32), False
            )

        n_init = (
            self.n_init_seeds if self.n_init_seeds is not None else 10 * n_superpixels
        )
        adjacency = ift.Circular(self.adjacency_radius)
        labels = ift.DISF(mimg, adjacency, n_init, n_superpixels, ift_mask)
        return np.ascontiguousarray(labels.AsNumPy().astype(np.int32))

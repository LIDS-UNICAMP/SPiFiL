"""Seed extractors: one representative pixel per superpixel.

The three definitions differ in what "representative" means --- geometry
(:class:`Centroids`, :class:`GeodesicCenters`) or appearance
(:class:`Medoids`) --- and typically disagree by only a few pixels, because
superpixels are small relative to the patches learned around them. They are
not interchangeable, though: a different seed pixel means a different patch,
and therefore different filters.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spifil.superpixels.disf import require_pyift

__all__ = ["Centroids", "GeodesicCenters", "Medoids"]


class GeodesicCenters:
    """Spatial geodesic center of each region.

    The pixel most central to a region's *shape*. Unlike :class:`Centroids`
    the result is always inside its region, including for non-convex shapes.

    Requires the optional ``disf`` extra (see :mod:`spifil.superpixels.disf`).
    """

    def __call__(
        self,
        labels: npt.NDArray[np.int32],
        features: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.int64]:
        ift = require_pyift()
        if labels.ndim != 2:
            raise ValueError(f"labels must be (H, W); got {labels.shape}")
        width = labels.shape[1]

        image = ift.CreateImageFromNumPy(
            np.ascontiguousarray(labels, dtype=np.int32), False
        )
        # Maps a flat pixel index to its region id.
        centers = ift.GeodesicCenters(image).AsDict()
        elems = np.array(
            [elem for elem, _ in sorted(centers.items(), key=lambda kv: kv[1])],
            dtype=np.int64,
        )
        return np.stack([elems // width, elems % width], axis=1)


class Centroids:
    """Mean of each region's pixel coordinates, truncated to integers.

    The cheapest seed rule, and the only one with no dependencies. Note that a
    centroid can fall *outside* a non-convex superpixel and land on
    background, which the other two extractors cannot.
    """

    def __call__(
        self,
        labels: npt.NDArray[np.int32],
        features: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.int64]:
        if labels.ndim != 2:
            raise ValueError(f"labels must be (H, W); got {labels.shape}")

        n_regions = int(labels.max())
        if n_regions < 1:
            return np.zeros((0, 2), dtype=np.int64)

        flat = labels.reshape(-1).astype(np.int64)
        ys, xs = np.divmod(np.arange(flat.size, dtype=np.int64), labels.shape[1])

        bins = n_regions + 1
        count = np.bincount(flat, minlength=bins)
        sum_y = np.bincount(flat, weights=ys, minlength=bins).astype(np.int64)
        sum_x = np.bincount(flat, weights=xs, minlength=bins).astype(np.int64)

        present = np.nonzero(count[1:])[0] + 1
        return np.stack(
            [sum_y[present] // count[present], sum_x[present] // count[present]],
            axis=1,
        )


class Medoids:
    """Feature-space medoid of each region: ``argmin_p ||I(p) - mu_tau||_2``.

    The seed rule the published SPiFiL results use, and the default. Like
    :class:`GeodesicCenters` it always lands inside its region, but it picks
    the pixel most representative of the region's *appearance* rather than of
    its shape.

    Unlike the geometric extractors this one needs the feature map, and raises
    if it is not given.

    Notes
    -----
    Per-region sums and the squared distance accumulate in **float64** even
    though the features are float32; the distance is left squared (monotone,
    so the argmin is unchanged); and ties go to the lowest flat pixel index.
    """

    def __call__(
        self,
        labels: npt.NDArray[np.int32],
        features: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.int64]:
        if labels.ndim != 2:
            raise ValueError(f"labels must be (H, W); got {labels.shape}")
        if features is None:
            raise ValueError(
                "Medoids needs the feature map to measure appearance; pass "
                "features=(C, H, W), or use GeodesicCenters/Centroids for a "
                "purely geometric seed."
            )
        if features.ndim != 3 or features.shape[1:] != labels.shape:
            raise ValueError(
                f"features must be (C, {labels.shape[0]}, {labels.shape[1]}) to "
                f"match labels; got {features.shape}"
            )

        n_regions = int(labels.max())
        if n_regions < 1:
            return np.zeros((0, 2), dtype=np.int64)

        flat = labels.reshape(-1).astype(np.int64)
        # (n_pixels, n_bands), accumulated in float64.
        values = features.reshape(features.shape[0], -1).T.astype(np.float64)

        index = np.nonzero(flat > 0)[0]
        region = flat[index]
        inside = values[index]

        bins = n_regions + 1
        count = np.bincount(region, minlength=bins)
        sums = np.stack(
            [
                np.bincount(region, weights=inside[:, band], minlength=bins)
                for band in range(inside.shape[1])
            ],
            axis=1,
        )
        means = np.zeros_like(sums)
        nonempty = count > 0
        means[nonempty] = sums[nonempty] / count[nonempty, None]

        deviation = inside - means[region]
        distance = np.einsum("nb,nb->n", deviation, deviation)

        # Primary key last: region, then distance, then pixel index — so the
        # first row of each region block is its medoid, ties broken by the
        # lowest index.
        order = np.lexsort((index, distance, region))
        ranked = region[order]
        is_first = np.empty(ranked.shape, dtype=bool)
        is_first[0] = True
        np.not_equal(ranked[1:], ranked[:-1], out=is_first[1:])
        chosen = index[order][is_first]

        width = labels.shape[1]
        return np.stack([chosen // width, chosen % width], axis=1)

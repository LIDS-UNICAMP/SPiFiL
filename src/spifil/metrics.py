"""Distance metrics for scoring seed patches.

Every scorer needs pairwise distances over a patch dataset; the metric owns
however much state that requires (nothing for :class:`Euclidean`, a fitted
covariance for :class:`Mahalanobis`).

**Mahalanobis via whitening, not a quadratic form.** Given ``Sigma = L L^T``
(Cholesky), ``Sigma^-1 = W^T W`` for ``W = L^-1``, so
``sqrt((x-y)^T Sigma^-1 (x-y)) = ||W(x-y)||_2``. Transforming the patch set by
``W`` once turns "N^2 quadratic forms" into "one torch.cdist on transformed
points".

**Covariance estimation runs in float64**, because the accumulation is where
precision is actually lost. The result is cast back to the input dtype before
any O(N^2) distance work, which stays in that dtype end to end.

.. note::
   The covariance is ``D x D`` for ``D = kernel_size^2 x in_channels``, and
   ``N``, the number of seed patches, is easily smaller than ``D`` at deeper
   layers. Shrinkage keeps the estimate invertible there, but not informative:
   for a statistically meaningful metric aim for ``N > D``, which is best
   reached by adding **images**, since seeds merge under pooling and their
   count saturates with ``n_superpixels``.
"""

from __future__ import annotations

import warnings
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

__all__ = ["DistanceMetric", "Euclidean", "Mahalanobis"]


@runtime_checkable
class DistanceMetric(Protocol):
    """Fits to a reference feature set, then answers pairwise distance queries."""

    def fit(self, feats: Tensor) -> DistanceMetric:
        """Estimate whatever statistics the metric needs (e.g. covariance)."""
        ...

    def pairwise(self, a: Tensor, b: Tensor) -> Tensor:
        """``(na, D)`` and ``(nb, D)`` -> ``(na, nb)`` distances."""
        ...


class Euclidean:
    """Plain Euclidean distance. Stateless; ``fit`` is a no-op."""

    def __init__(self, *, chunk_size: int = 4096) -> None:
        self.chunk_size = chunk_size

    def fit(self, feats: Tensor) -> Euclidean:
        """Nothing to estimate; returns ``self`` so the protocol still holds."""
        return self

    def pairwise(self, a: Tensor, b: Tensor) -> Tensor:
        """``(na, D)`` and ``(nb, D)`` -> ``(na, nb)`` distances, in row blocks."""
        return _chunked_cdist(a, b, self.chunk_size)


class Mahalanobis:
    """Mahalanobis distance from a covariance fitted on a reference set.

    Parameters
    ----------
    estimator
        ``"shrinkage"`` (default): Ledoit-Wolf shrinkage toward a scaled
        identity. Positive-definite for any ``N``, so ``fit`` always succeeds
        and ``pairwise`` always uses the Cholesky-whitening path.
        ``"empirical"``: the plain (biased) sample covariance, which is
        rank-deficient whenever ``N <= D``; ``fit`` then warns and falls back
        to Euclidean distance rather than inverting a singular matrix.
    chunk_size
        Row-block size for the O(N^2) pairwise computation, so the full
        distance matrix is never required to fit in memory at once.

    Notes
    -----
    Device-agnostic: runs on whatever device ``feats`` lives on, CPU or GPU,
    with no explicit ``.cuda()`` calls.
    """

    def __init__(
        self,
        estimator: Literal["shrinkage", "empirical"] = "shrinkage",
        *,
        chunk_size: int = 4096,
    ) -> None:
        if estimator not in ("shrinkage", "empirical"):
            raise ValueError(
                f"unknown estimator {estimator!r}; use 'shrinkage' or 'empirical'"
            )
        self.estimator = estimator
        self.chunk_size = chunk_size
        self._whitener: Tensor | None = None
        self._use_euclidean = False

    def fit(self, feats: Tensor) -> Mahalanobis:
        """Estimate the covariance over ``(N, D)`` reference rows and invert it.

        The covariance is accumulated in float64 and divided by ``N`` rather
        than ``N - 1``. Under the default shrinkage estimator the result is
        positive-definite for any ``N``, so the Cholesky whitener always
        exists; the empirical estimator falls back to Euclidean distance when
        the covariance turns out to be singular.

        Returns ``self``, so ``metric.fit(x).pairwise(a, b)`` reads as one step.
        """
        if feats.ndim != 2:
            raise ValueError(f"feats must be (N, D); got {tuple(feats.shape)}")
        n, d = feats.shape

        x64 = feats.to(torch.float64)
        centered = x64 - x64.mean(dim=0)
        cov = (centered.T @ centered) / n  # biased (divide by N)

        if n <= d:
            warnings.warn(
                f"Mahalanobis: N={n} <= D={d}: the empirical covariance is "
                f"rank-deficient (rank <= {n - 1}). Shrinkage keeps it "
                "invertible, but the metric is only informative for N > D "
                "— add more images per class",
                stacklevel=2,
            )

        if self.estimator == "shrinkage":
            lam = _ledoit_wolf_lambda(centered, cov)
            cov = _shrink_toward_identity(cov, lam)

        self._whitener = None
        self._use_euclidean = False

        try:
            chol = torch.linalg.cholesky(cov)
        except torch.linalg.LinAlgError:  # type: ignore[attr-defined]
            chol = None

        if chol is not None:
            eye = torch.eye(d, dtype=torch.float64, device=cov.device)
            upper = chol.transpose(-2, -1).contiguous()
            whitener = torch.linalg.solve_triangular(upper, eye, upper=True)
            self._whitener = whitener.to(feats.dtype)
            return self

        warnings.warn(
            "Mahalanobis: singular covariance, falling back to Euclidean distance",
            stacklevel=2,
        )
        self._use_euclidean = True
        return self

    def pairwise(self, a: Tensor, b: Tensor) -> Tensor:
        """``(na, D)`` and ``(nb, D)`` -> ``(na, nb)`` distances, in row blocks.

        Whitens both sides and takes a plain Euclidean norm rather than
        evaluating the quadratic form, so the result can never be the square
        root of a negative number. Raises if called before :meth:`fit`.
        """
        if self._use_euclidean:
            return _chunked_cdist(a, b, self.chunk_size)
        if self._whitener is not None:
            wa, wb = a @ self._whitener, b @ self._whitener
            return _chunked_cdist(wa, wb, self.chunk_size)
        raise RuntimeError("Mahalanobis.pairwise() called before fit()")


def _ledoit_wolf_lambda(centered: Tensor, cov: Tensor) -> float:
    """Shrinkage intensity ``lambda = min(b2, d2) / d2``."""
    n, d = centered.shape
    mu = (torch.trace(cov) / d).item()
    norm_s2 = torch.sum(cov * cov).item()
    d2 = (norm_s2 - d * mu * mu) / d
    if d2 <= 1e-12:
        return 1.0  # already (numerically) a scaled identity

    xx = (centered * centered).sum(dim=1)
    xsx = torch.einsum("ni,ij,nj->n", centered, cov, centered)
    sample_terms = xx * xx - 2.0 * xsx + norm_s2
    b2 = (sample_terms.sum() / (n * n * d)).item()
    b2 = min(max(b2, 0.0), d2)  # b2 <= d2 by construction; clip for safety
    return b2 / d2


def _shrink_toward_identity(cov: Tensor, lam: float) -> Tensor:
    d = cov.shape[0]
    mu = torch.trace(cov) / d
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    return lam * mu * eye + (1.0 - lam) * cov


def _chunked_cdist(a: Tensor, b: Tensor, chunk_size: int) -> Tensor:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")
    rows = [
        torch.cdist(a[start : start + chunk_size], b)
        for start in range(0, a.shape[0], chunk_size)
    ]
    return torch.cat(rows, dim=0)

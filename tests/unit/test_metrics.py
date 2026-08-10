"""Distance metrics: Euclidean, and Mahalanobis (Ledoit-Wolf shrinkage / empirical)."""

from __future__ import annotations

import warnings

import pytest
import torch

from spifil.metrics import DistanceMetric, Euclidean, Mahalanobis


def naive_ledoit_wolf(x: torch.Tensor) -> torch.Tensor:
    """The Ledoit-Wolf shrinkage estimate, written out in double precision.

    Unvectorized — the shipped implementation batches these same reductions
    with tensor ops, which is easy to get subtly wrong (b2's
    normalisation especially); this is what it must agree with.
    """
    x = x.double()
    n, d = x.shape
    mean = [sum(x[k, j].item() for k in range(n)) / n for j in range(d)]
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            s = 0.0
            for k in range(n):
                s += (x[k, i].item() - mean[i]) * (x[k, j].item() - mean[j])
            cov[i][j] = s / n

    mu = sum(cov[i][i] for i in range(d)) / d
    norm_s2 = sum(cov[i][j] ** 2 for i in range(d) for j in range(d))
    d2 = (norm_s2 - d * mu * mu) / d

    if d2 <= 1e-12:
        lam = 1.0
    else:
        b2_sum = 0.0
        for k in range(n):
            xk = [x[k, j].item() - mean[j] for j in range(d)]
            xx = sum(v * v for v in xk)
            xsx = sum(
                xk[i] * sum(cov[i][j] * xk[j] for j in range(d)) for i in range(d)
            )
            b2_sum += xx * xx - 2 * xsx + norm_s2
        b2 = b2_sum / (n * n * d)
        b2 = min(max(b2, 0.0), d2)
        lam = b2 / d2

    shrunk = [
        [(lam * mu if i == j else 0.0) + (1 - lam) * cov[i][j] for j in range(d)]
        for i in range(d)
    ]
    return torch.tensor(shrunk, dtype=torch.float64)


def naive_mahalanobis(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Distances from the naive shrunk covariance, via direct matrix inversion."""
    cov_inv = torch.linalg.inv(naive_ledoit_wolf(x))
    a64, b64 = a.double(), b.double()
    out = torch.zeros(a.shape[0], b.shape[0], dtype=torch.float64)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            diff = a64[i] - b64[j]
            out[i, j] = torch.sqrt((diff @ cov_inv @ diff).abs())
    return out


class TestEuclidean:
    def test_matches_cdist(self) -> None:
        torch.manual_seed(0)
        a, b = torch.randn(5, 4), torch.randn(3, 4)
        assert torch.allclose(Euclidean().pairwise(a, b), torch.cdist(a, b))

    def test_chunking_does_not_change_the_result(self) -> None:
        torch.manual_seed(1)
        a, b = torch.randn(23, 6), torch.randn(11, 6)
        full = Euclidean(chunk_size=1000).pairwise(a, b)
        chunked = Euclidean(chunk_size=3).pairwise(a, b)
        assert torch.allclose(full, chunked, atol=1e-5)

    def test_fit_is_a_no_op(self) -> None:
        metric = Euclidean()
        assert metric.fit(torch.randn(4, 2)) is metric

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(Euclidean(), DistanceMetric)


class TestMahalanobisShrinkage:
    def test_matches_the_naive_implementation(self) -> None:
        torch.manual_seed(2)
        x = torch.randn(20, 5)
        metric = Mahalanobis().fit(x)

        got = metric.pairwise(x[:4], x[:4])
        expected = naive_mahalanobis(x, x[:4], x[:4])

        assert torch.allclose(got.double(), expected, atol=1e-3)

    def test_distance_to_self_is_zero(self) -> None:
        torch.manual_seed(3)
        x = torch.randn(30, 4)
        metric = Mahalanobis().fit(x)

        d = metric.pairwise(x[:5], x[:5])

        assert torch.allclose(torch.diagonal(d), torch.zeros(5), atol=1e-4)

    def test_rank_deficient_covariance_still_fits(self) -> None:
        """N <= D: shrinkage keeps Sigma positive-definite (M0 finding)."""
        torch.manual_seed(4)
        x = torch.randn(3, 10)  # N=3 < D=10

        with pytest.warns(UserWarning, match="rank-deficient"):
            metric = Mahalanobis().fit(x)

        assert torch.isfinite(metric.pairwise(x, x)).all()

    def test_chunking_does_not_change_the_result(self) -> None:
        torch.manual_seed(5)
        x = torch.randn(40, 6)
        full = Mahalanobis(chunk_size=1000).fit(x).pairwise(x, x)
        chunked = Mahalanobis(chunk_size=4).fit(x).pairwise(x, x)

        assert torch.allclose(full, chunked, atol=1e-4)

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(Mahalanobis(), DistanceMetric)

    def test_pairwise_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            Mahalanobis().pairwise(torch.randn(2, 3), torch.randn(2, 3))

    def test_rejects_unknown_estimator(self) -> None:
        with pytest.raises(ValueError, match="estimator"):
            Mahalanobis(estimator="bogus")  # type: ignore[arg-type]


class TestMahalanobisEmpirical:
    def test_rank_deficient_covariance_warns_and_stays_finite(self) -> None:
        torch.manual_seed(6)
        x = torch.randn(3, 10)  # N < D: empirical covariance is exactly singular

        with pytest.warns(UserWarning, match="rank-deficient"):
            metric = Mahalanobis(estimator="empirical").fit(x)

        assert torch.isfinite(metric.pairwise(x, x)).all()

    def test_singular_covariance_falls_back_to_euclidean(self) -> None:
        """Identical rows -> covariance is exactly zero -> Cholesky fails."""
        x = torch.ones(5, 4)

        with pytest.warns(UserWarning, match="falling back to Euclidean"):
            metric = Mahalanobis(estimator="empirical").fit(x)

        assert torch.allclose(metric.pairwise(x, x), torch.zeros(5, 5))

    def test_well_conditioned_data_still_uses_cholesky(self) -> None:
        """Empirical only diverges from shrinkage once Cholesky actually fails.

        A well-conditioned covariance never reaches the Euclidean fallback,
        so no warning should fire and the (Cholesky-whitened) distances should
        come out finite and symmetric — the same path ``estimator="shrinkage"``
        takes here, since shrinkage barely perturbs an already-PD covariance.
        """
        torch.manual_seed(7)
        x = torch.randn(200, 5)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metric = Mahalanobis(estimator="empirical").fit(x)
        assert not caught, [str(w.message) for w in caught]

        d = metric.pairwise(x[:5], x[:5])
        assert torch.isfinite(d).all()
        assert torch.allclose(d, d.T, atol=1e-4)

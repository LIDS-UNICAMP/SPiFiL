"""Superpixel algorithms and seed extractors on synthetic input."""

from __future__ import annotations

import numpy as np
import pytest

from spifil.superpixels import (
    DISF,
    SLIC,
    Centroids,
    GeodesicCenters,
    Medoids,
    SeedExtractor,
    SuperpixelAlgorithm,
)

ALL_EXTRACTORS = [GeodesicCenters(), Centroids(), Medoids()]
EXTRACTOR_IDS = ["geodesic", "centroid", "medoid"]


def naive_medoids(labels: np.ndarray, features: np.ndarray) -> np.ndarray:
    """The medoid rule, stated directly as a loop.

    Unvectorized. The shipped implementation is a lexsort trick that is easy
    to get subtly wrong (especially the tie-break); this is the
    thing it must agree with.
    """
    width = labels.shape[1]
    flat = labels.reshape(-1)
    values = features.reshape(features.shape[0], -1).T.astype(np.float64)

    picks = []
    for region in range(1, int(labels.max()) + 1):
        members = np.nonzero(flat == region)[0]
        if members.size == 0:
            continue
        mean = values[members].mean(axis=0)
        distance = ((values[members] - mean) ** 2).sum(axis=1)
        # argmin returns the FIRST minimum -> lowest flat index, as the
        # ascending scan with a strict '<' does.
        best = members[int(np.argmin(distance))]
        picks.append([best // width, best % width])
    return np.array(picks, dtype=np.int64).reshape(-1, 2)


pyift = pytest.importorskip("pyift.pyift", reason="DISF needs PyIFT")


@pytest.fixture
def stripes() -> np.ndarray:
    """A (3, 24, 24) image split into four horizontal bands."""
    features = np.zeros((3, 24, 24), dtype=np.float32)
    for i in range(4):
        features[:, i * 6 : (i + 1) * 6, :] = i / 4.0
    return features


def test_algorithms_satisfy_the_protocol() -> None:
    assert isinstance(DISF(), SuperpixelAlgorithm)
    assert isinstance(SLIC(), SuperpixelAlgorithm)


@pytest.mark.parametrize("extractor", ALL_EXTRACTORS, ids=EXTRACTOR_IDS)
def test_extractors_satisfy_the_protocol(extractor: SeedExtractor) -> None:
    assert isinstance(extractor, SeedExtractor)


@pytest.mark.parametrize("algorithm", [DISF(), SLIC()], ids=["disf", "slic"])
def test_segmentation_shape_and_label_range(
    algorithm: SuperpixelAlgorithm, stripes: np.ndarray
) -> None:
    labels = algorithm(stripes, None, 8)

    assert labels.shape == stripes.shape[1:]
    assert labels.dtype == np.int32
    assert labels.min() >= 0
    # A loose upper bound: SLIC treats n_segments as a target and may overshoot
    # slightly, whereas DISF honours it exactly (asserted separately below).
    assert 1 <= labels.max() <= 12


def test_disf_honours_the_requested_count_exactly(stripes: np.ndarray) -> None:
    """DISF prunes down to exactly n_superpixels, unlike SLIC."""
    assert DISF()(stripes, None, 8).max() == 8


@pytest.mark.parametrize("algorithm", [DISF(), SLIC()], ids=["disf", "slic"])
def test_mask_confines_superpixels(
    algorithm: SuperpixelAlgorithm, stripes: np.ndarray
) -> None:
    mask = np.zeros((24, 24), dtype=np.int32)
    mask[4:20, 4:20] = 255

    labels = algorithm(stripes, mask, 6)

    assert np.all(labels[mask == 0] == 0)
    assert np.any(labels[mask != 0] > 0)


@pytest.mark.parametrize("algorithm", [DISF(), SLIC()], ids=["disf", "slic"])
def test_segmentation_is_deterministic(
    algorithm: SuperpixelAlgorithm, stripes: np.ndarray
) -> None:
    assert np.array_equal(algorithm(stripes, None, 8), algorithm(stripes, None, 8))


def test_disf_rejects_mismatched_mask(stripes: np.ndarray) -> None:
    with pytest.raises(ValueError, match="does not match"):
        DISF()(stripes, np.ones((10, 10), dtype=np.int32), 4)


@pytest.mark.parametrize("algorithm", [DISF(), SLIC()], ids=["disf", "slic"])
def test_segmentation_rejects_2d_input(algorithm: SuperpixelAlgorithm) -> None:
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        algorithm(np.zeros((8, 8), dtype=np.float32), None, 4)


@pytest.mark.parametrize("extractor", ALL_EXTRACTORS, ids=EXTRACTOR_IDS)
def test_one_seed_per_region_inside_the_grid(
    extractor: SeedExtractor, stripes: np.ndarray
) -> None:
    labels = DISF()(stripes, None, 8)

    coords = extractor(labels, stripes)

    assert coords.shape == (len(np.unique(labels[labels > 0])), 2)
    assert coords.dtype == np.int64
    assert np.all(coords >= 0)
    assert np.all(coords < np.array(labels.shape))


def test_geodesic_centers_lie_inside_their_region(stripes: np.ndarray) -> None:
    """The property that motivates geodesic centers over centroids."""
    labels = DISF()(stripes, None, 8)

    coords = GeodesicCenters()(labels)

    assert np.all(labels[coords[:, 0], coords[:, 1]] > 0)


def test_centroid_is_the_truncated_coordinate_mean() -> None:
    """Hand-checkable: one L-shaped region, integer division as in C."""
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0, 0] = labels[1, 0] = labels[2, 0] = labels[2, 1] = 1
    # ys = 0,1,2,2 -> 5//4 = 1 ; xs = 0,0,0,1 -> 1//4 = 0

    assert np.array_equal(Centroids()(labels), np.array([[1, 0]]))


def test_centroids_skip_absent_region_ids() -> None:
    """Region ids need not be contiguous; only present ones yield a seed."""
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0, :2] = 1
    labels[3, :2] = 3

    assert Centroids()(labels).shape == (2, 2)


def test_centroids_on_empty_labels() -> None:
    assert Centroids()(np.zeros((4, 4), dtype=np.int32)).shape == (0, 2)


def test_medoid_is_the_closest_pixel_to_the_region_mean() -> None:
    """Hand-checkable: one 1-band region of three pixels."""
    labels = np.zeros((1, 4), dtype=np.int32)
    labels[0, :3] = 1
    features = np.array([[[0.0, 10.0, 11.0, 99.0]]], dtype=np.float32)
    # mean of {0, 10, 11} = 7 ; squared distances {49, 9, 16} -> pixel 1 wins

    assert np.array_equal(Medoids()(labels, features), np.array([[0, 1]]))


def test_medoid_lands_inside_its_region() -> None:
    """The property medoid shares with geodesic and centroid lacks."""
    rng = np.random.default_rng(0)
    features = rng.random((3, 24, 24), dtype=np.float32)
    labels = DISF()(features, None, 8)

    coords = Medoids()(labels, features)

    assert np.all(labels[coords[:, 0], coords[:, 1]] > 0)


def test_medoid_ties_go_to_the_lowest_pixel_index() -> None:
    """Constant features make every distance 0, exposing the tie rule.

    Pixels are scanned in row-major order with a strict '<', so the first
    (lowest-index) pixel of each region wins. Any other rule — last, or an
    unstable sort — would silently change which patches become filters.
    """
    labels = np.zeros((3, 4), dtype=np.int32)
    labels[1, 0:3] = 1  # linear indices 4, 5, 6
    labels[2, 1:3] = 2  # linear indices 9, 10
    features = np.ones((2, 3, 4), dtype=np.float32)

    coords = Medoids()(labels, features)

    assert np.array_equal(coords, np.array([[1, 0], [2, 1]]))


def test_medoid_matches_the_naive_implementation() -> None:
    """The vectorized lexsort must agree with the literal loop above."""
    rng = np.random.default_rng(7)
    for trial in range(5):
        features = rng.random((3, 20, 20), dtype=np.float32)
        labels = DISF()(features, None, 6 + trial)

        assert np.array_equal(
            Medoids()(labels, features), naive_medoids(labels, features)
        )


def test_medoid_matches_naive_under_heavy_ties() -> None:
    """Quantized features create many exact ties — the tie-break's stress test."""
    rng = np.random.default_rng(11)
    features = rng.integers(0, 3, (2, 16, 16)).astype(np.float32)
    labels = rng.integers(0, 5, (16, 16)).astype(np.int32)

    assert np.array_equal(Medoids()(labels, features), naive_medoids(labels, features))


def test_medoid_skips_absent_region_ids() -> None:
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0, :2] = 1
    labels[3, :2] = 4
    features = np.ones((1, 4, 4), dtype=np.float32)

    assert Medoids()(labels, features).shape == (2, 2)


def test_medoid_on_empty_labels() -> None:
    labels = np.zeros((4, 4), dtype=np.int32)

    assert Medoids()(labels, np.ones((1, 4, 4), dtype=np.float32)).shape == (0, 2)


def test_medoid_requires_features() -> None:
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0, 0] = 1

    with pytest.raises(ValueError, match="needs the feature map"):
        Medoids()(labels)


def test_medoid_rejects_mismatched_features() -> None:
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0, 0] = 1

    with pytest.raises(ValueError, match="to match labels"):
        Medoids()(labels, np.ones((3, 8, 8), dtype=np.float32))


def test_medoid_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    features = rng.random((3, 16, 16), dtype=np.float32)
    labels = DISF()(features, None, 5)

    assert np.array_equal(Medoids()(labels, features), Medoids()(labels, features))


@pytest.mark.parametrize("extractor", ALL_EXTRACTORS, ids=EXTRACTOR_IDS)
def test_extractors_reject_3d_labels(extractor: SeedExtractor) -> None:
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        extractor(np.zeros((2, 4, 4), dtype=np.int32))

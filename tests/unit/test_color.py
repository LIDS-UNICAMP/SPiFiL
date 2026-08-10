"""LabNorm behavior on synthetic input.

This file
covers the shape/range contracts and the properties that are easy to "fix"
by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from spifil.color import ColorTransform, LabNorm


@pytest.fixture
def lab() -> LabNorm:
    return LabNorm()


def test_satisfies_the_protocol(lab: LabNorm) -> None:
    assert isinstance(lab, ColorTransform)


def test_output_is_channels_first_float32(lab: LabNorm) -> None:
    image = np.random.default_rng(0).integers(0, 256, (7, 11, 3), dtype=np.uint8)

    out = lab(image)

    assert out.shape == (3, 7, 11)
    assert out.dtype == np.float32


def test_channels_land_in_roughly_unit_range(lab: LabNorm) -> None:
    """LABNorm2 divides by hard-coded ranges, so [0, 1] is nominal, not exact."""
    image = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)

    out = lab(image)

    assert out.min() >= -0.1
    assert out.max() <= 1.1


def test_is_deterministic(lab: LabNorm) -> None:
    image = np.random.default_rng(2).integers(0, 256, (16, 16, 3), dtype=np.uint8)

    assert np.array_equal(lab(image), lab(image))


def test_alpha_channel_is_dropped(lab: LabNorm) -> None:
    rgb = np.random.default_rng(3).integers(0, 256, (5, 5, 3), dtype=np.uint8)
    rgba = np.concatenate([rgb, np.full((5, 5, 1), 255, np.uint8)], axis=2)

    assert np.array_equal(lab(rgba), lab(rgb))


def test_grayscale_input_raises_rather_than_guessing(lab: LabNorm) -> None:
    """Grayscale input is refused rather than silently colourized."""
    with pytest.raises(ValueError, match="RGB"):
        lab(np.zeros((8, 8), dtype=np.uint8))


def test_no_srgb_gamma_decode(lab: LabNorm) -> None:
    """The XYZ matrix is applied to raw R/G/B.

    Mid-gray is the cheapest place to see it: with sRGB linearization L* of
    128-gray is ~53.6, without it ~76. Guards against "correcting" the
    conversion to a textbook CIELAB, which would silently change every
    feature the pipeline learns from.
    """
    gray = np.full((1, 1, 3), 128, dtype=np.uint8)

    lightness = float(lab(gray)[0, 0, 0]) * 99.998337

    assert lightness == pytest.approx(76.0, abs=1.5)


def test_ycbcr_round_trip_is_lossy(lab: LabNorm) -> None:
    """Reading quantizes through BT.2020 YCbCr, so not every RGB survives.

    If the round trip were ever made symmetric (or removed), distinct dark
    colors would stop collapsing onto each other.
    """
    colors = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.uint8)

    out = lab(colors)

    assert np.array_equal(out[:, 0, 0], out[:, 0, 1])

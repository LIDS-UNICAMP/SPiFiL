"""Shared fixtures: small synthetic datasets, generated once per session.

The suite ships no images. Everything the tests need --- folders of
``<class>_<name>.png`` images with matching masks --- is drawn here from a
fixed seed, so a fresh clone can run ``pytest`` with nothing downloaded and
every run sees exactly the same pixels.

The three sets differ only in how many classes and images they hold, which is
the axis the tests actually care about:

``MULTICLASS``
    Six classes, three images each. Exercises class discovery and per-class
    budgets.
``TWOCLASS``
    Two classes, five images each.
``MINIMAL``
    Two classes, one image each --- the smallest input a full fit can run on,
    and what the Learner tests use to keep a two-layer fit at milliseconds.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from skimage.io import imsave

IMAGE_SHAPE = (48, 64)
"""``(height, width)`` of every generated image: small, to keep tests fast."""


@dataclass(frozen=True)
class SampleSet:
    """A generated folder pair, and the numbers the tests assert against."""

    name: str
    root: Path
    n_classes: int
    per_class: int

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def masks(self) -> Path:
        return self.root / "masks"

    @property
    def classes(self) -> list[int]:
        return list(range(1, self.n_classes + 1))

    def __len__(self) -> int:
        return self.n_classes * self.per_class


def _draw(label: int, index: int) -> tuple[np.ndarray, np.ndarray]:
    """One image and its mask: a textured blob of a class-dependent hue.

    Deterministic in ``(label, index)``. The blob gives the mask a non-convex
    interior, and the texture gives the superpixel algorithms something to cut
    along --- a flat image would segment into arbitrary tiles and make the
    seed-placement assertions meaningless.
    """
    height, width = IMAGE_SHAPE
    rng = np.random.default_rng(1000 * label + index)
    ys, xs = np.mgrid[0:height, 0:width]

    # An ellipse, off-center by class, dented by a lobe so it is not convex.
    cy, cx = height / 2 + 3 * (label % 3), width / 2 - 4 * (label % 2)
    ellipse = ((ys - cy) / (height * 0.32)) ** 2 + ((xs - cx) / (width * 0.28)) ** 2
    lobe = ((ys - cy + 9) / 7.0) ** 2 + ((xs - cx - 11) / 7.0) ** 2
    mask = (ellipse <= 1.0) & (lobe >= 1.0)

    # Class-dependent hue, plus stripes and noise so patches differ.
    base = np.array([60 + 30 * (label % 3), 90 + 25 * (label % 2), 140 - 20 * label])
    stripes = 18.0 * np.sin(xs / 3.0 + label) + 12.0 * np.cos(ys / 4.0 - index)
    image = base[None, None, :] + stripes[:, :, None]
    image = image + rng.normal(0.0, 6.0, size=(height, width, 3))

    return (
        np.clip(image, 0, 255).astype(np.uint8),
        (mask * 255).astype(np.uint8),
    )


def _generate(root: Path, name: str, n_classes: int, per_class: int) -> SampleSet:
    sample_set = SampleSet(name, root / name, n_classes, per_class)
    sample_set.images.mkdir(parents=True)
    sample_set.masks.mkdir(parents=True)

    for label in sample_set.classes:
        for index in range(per_class):
            image, mask = _draw(label, index)
            stem = f"{label:06d}_{index:08d}.png"
            imsave(sample_set.images / stem, image, check_contrast=False)
            imsave(sample_set.masks / stem, mask, check_contrast=False)
    return sample_set


_ROOT = Path(tempfile.mkdtemp(prefix="spifil-fixtures-"))
atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)

MULTICLASS = _generate(_ROOT, "multiclass", n_classes=6, per_class=3)
TWOCLASS = _generate(_ROOT, "twoclass", n_classes=2, per_class=5)
MINIMAL = _generate(_ROOT, "minimal", n_classes=2, per_class=1)


@pytest.fixture
def multiclass() -> SampleSet:
    """Six classes, three images each."""
    return MULTICLASS


@pytest.fixture
def twoclass() -> SampleSet:
    """Two classes, five images each."""
    return TWOCLASS


@pytest.fixture
def minimal() -> SampleSet:
    """Two classes, one image each."""
    return MINIMAL

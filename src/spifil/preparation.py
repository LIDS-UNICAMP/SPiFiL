"""Layer 0: color features, superpixels, and the initial seed set.

Per image: convert to multiband color features, segment those features into
superpixels inside the mask, reduce each superpixel to one seed, and label every
seed with the image's class.

Free functions rather than Learner methods, so layer 0 can be run and
inspected on its own --- which is what makes seed placement easy to visualize
--- while the Learner drives this same code.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from spifil.color import ColorTransform, LabNorm
from spifil.data import SpifilDataset
from spifil.superpixels.base import SeedExtractor, SuperpixelAlgorithm
from spifil.types import Seeds

__all__ = ["PreparedImage", "prepare", "prepare_sample"]


@dataclass(frozen=True)
class PreparedImage:
    """Everything layer 0 produces for one image."""

    name: str
    features: npt.NDArray[np.float32]
    """``(C, H, W)`` layer-0 multiband features: the color-transformed image."""
    superpixel_labels: npt.NDArray[np.int32]
    """``(H, W)`` region ids; retained because it is what makes seed placement
    reviewable, and it is not recoverable from the seeds alone."""
    seeds: Seeds
    """One seed per region, all carrying the image's class label."""


def prepare_sample(
    dataset: SpifilDataset,
    index: int,
    *,
    superpixels: SuperpixelAlgorithm,
    seed_extractor: SeedExtractor,
    n_superpixels: int = 100,
    color: ColorTransform | None = None,
) -> PreparedImage:
    """Run layer 0 for a single sample.

    Parameters
    ----------
    dataset
        Source of the image, its mask, and its class label.
    index
        Position in ``dataset``.
    superpixels
        Segmentation algorithm, e.g. :class:`~spifil.superpixels.slic.SLIC` or
        :class:`~spifil.superpixels.disf.DISF`.
    seed_extractor
        Rule for reducing a region to one pixel, e.g.
        :class:`~spifil.superpixels.centers.Medoids`.
    n_superpixels
        Target region count per image.
    color
        Color transform. Defaults to :class:`~spifil.color.LabNorm`.
    """
    color = color if color is not None else LabNorm()

    features = color(dataset.load_image(index))
    labels = superpixels(features, dataset.load_mask(index), n_superpixels)
    coords = seed_extractor(labels, features)

    return PreparedImage(
        name=dataset[index].name,
        features=features,
        superpixel_labels=labels,
        # Every seed inherits the image's class, whatever region it came
        # from: the mask guarantees it lies on the class of interest.
        seeds=Seeds.from_coords(
            torch.from_numpy(coords),
            label=dataset[index].label,
            grid=features.shape[1:],
        ),
    )


def prepare(
    dataset: SpifilDataset,
    *,
    superpixels: SuperpixelAlgorithm,
    seed_extractor: SeedExtractor,
    n_superpixels: int = 100,
    color: ColorTransform | None = None,
) -> Iterator[PreparedImage]:
    """Run layer 0 over a whole dataset, yielding one result per sample.

    Lazy: feature maps are large, and consumers either write them out or
    fold them into a running statistic.
    """
    for index in range(len(dataset)):
        yield prepare_sample(
            dataset,
            index,
            superpixels=superpixels,
            seed_extractor=seed_extractor,
            n_superpixels=n_superpixels,
            color=color,
        )

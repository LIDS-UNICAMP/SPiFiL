"""Superpixel segmentation and seed extraction — the two halves of layer 0."""

from spifil.superpixels.base import SeedExtractor, SuperpixelAlgorithm
from spifil.superpixels.centers import Centroids, GeodesicCenters, Medoids
from spifil.superpixels.disf import DISF
from spifil.superpixels.slic import SLIC

__all__ = [
    "DISF",
    "SLIC",
    "Centroids",
    "GeodesicCenters",
    "Medoids",
    "SeedExtractor",
    "SuperpixelAlgorithm",
]

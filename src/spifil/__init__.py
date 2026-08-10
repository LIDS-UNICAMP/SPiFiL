"""SPiFiL: Superpixel-based Filter Learning."""

from spifil.allocation import FilterAllocator, UniformAllocator
from spifil.callbacks import Callback, Checkpoint, CSVLogger, ProgressBar
from spifil.color import ColorTransform, Identity, LabNorm
from spifil.config import ArchSpec, LayerSpec
from spifil.data import Sample, SpifilDataset
from spifil.learner import (
    CancelFitException,
    CancelLayerException,
    LayerState,
    Learner,
)
from spifil.metrics import DistanceMetric, Euclidean, Mahalanobis
from spifil.nn import (
    ConvBlock,
    ConvBlockFactory,
    FilterBank,
    SpifilConvBlock,
    SpifilEncoder,
    SpifilNet,
    build_filters,
    load_encoder,
    load_model,
)
from spifil.patches import extract_patches
from spifil.preparation import PreparedImage, prepare, prepare_sample
from spifil.scoring import (
    DistanceSumScorer,
    FisherScorer,
    Scorer,
    per_class_ranks,
    scatter_ranks,
)
from spifil.selection import DiversitySelector, Selector, TopN
from spifil.strategies import FitStrategy, SequentialStrategy
from spifil.superpixels import (
    DISF,
    SLIC,
    Centroids,
    GeodesicCenters,
    Medoids,
    SeedExtractor,
    SuperpixelAlgorithm,
)
from spifil.types import PatchSet, Seeds

__version__ = "1.0.0"

__all__ = [
    "DISF",
    "SLIC",
    "ArchSpec",
    "CSVLogger",
    "Callback",
    "CancelFitException",
    "CancelLayerException",
    "Centroids",
    "Checkpoint",
    "ColorTransform",
    "ConvBlock",
    "ConvBlockFactory",
    "DistanceMetric",
    "DistanceSumScorer",
    "DiversitySelector",
    "Euclidean",
    "FilterAllocator",
    "FilterBank",
    "FisherScorer",
    "FitStrategy",
    "GeodesicCenters",
    "Identity",
    "LabNorm",
    "LayerSpec",
    "LayerState",
    "Learner",
    "Mahalanobis",
    "Medoids",
    "PatchSet",
    "PreparedImage",
    "ProgressBar",
    "Sample",
    "Scorer",
    "SeedExtractor",
    "Seeds",
    "Selector",
    "SequentialStrategy",
    "SpifilConvBlock",
    "SpifilDataset",
    "SpifilEncoder",
    "SpifilNet",
    "SuperpixelAlgorithm",
    "TopN",
    "UniformAllocator",
    "build_filters",
    "extract_patches",
    "load_encoder",
    "load_model",
    "per_class_ranks",
    "prepare",
    "prepare_sample",
    "scatter_ranks",
]

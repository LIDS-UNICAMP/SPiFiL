"""Structured configs.

These dataclasses are both the Python API's parameter objects and the schemas
Hydra type-checks its YAML against (registered in the ``ConfigStore`` below).

Components (scorer, selector, ...) are *not* fields here: Hydra instantiates
them from ``conf/`` and hands the Learner ready-made objects, so nothing below
ever holds a ``DictConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

__all__ = ["ArchSpec", "DataConfig", "LayerSpec", "SpifilConfig", "register_configs"]


@dataclass
class LayerSpec:
    """One convolutional layer: kernel, dilation, filter count, pooling."""

    kernel_size: int = 3
    out_channels: int = 16
    dilation: int = 1
    pool_type: str = "max"  # "max" | "avg" | "none"
    pool_size: int = 3
    pool_stride: int = 2
    relu: bool = True

    # Optional per-layer component overrides, set through the Python API.
    # ``None`` means "use the Learner's default".
    scorer: Any | None = None
    selector: Any | None = None
    metric: Any | None = None
    allocator: Any | None = None


@dataclass
class ArchSpec:
    """The stack of layers plus the z-score floor used when folding filters."""

    layers: list[LayerSpec] = field(default_factory=list)
    stdev_factor: float = 0.01


@dataclass
class DataConfig:
    """Image/mask folders. Class label is the integer prefix of each filename."""

    images_dir: str = "???"
    masks_dir: str = "???"


@dataclass
class SpifilConfig:
    """Root config. Component nodes are filled by Hydra's config groups."""

    data: DataConfig = field(default_factory=DataConfig)
    arch: ArchSpec = field(default_factory=ArchSpec)
    output_dir: str = "."
    device: str = "cpu"
    seed: int = 42
    n_superpixels: int = 100
    resume_from: int | None = None
    """Layer to restart from: checkpoints from here up are discarded before
    fitting. ``None`` resumes from whatever checkpoints exist."""

    color: Any = None
    superpixels: Any = None
    seed_extractor: Any = None
    metric: Any = None
    scorer: Any = None
    allocator: Any = None
    selector: Any = None
    strategy: Any = None


def register_configs() -> None:
    """Register the schemas with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="spifil_schema", node=SpifilConfig)

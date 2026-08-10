"""Hydra composes our configs, and the schema catches bad ones.

Establishes the Compose-API pattern the rest of the suite will use: no CLI, no
chdir, no run directories — just composition in-process.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf, ValidationError

import spifil
from spifil.config import ArchSpec, LayerSpec, register_configs

# Resolved from the installed package: conf/ ships inside spifil so the
# CLI works from a wheel.
CONF_DIR = str((Path(spifil.__file__).parent / "conf").resolve())


def compose_config(*overrides: str):  # type: ignore[no-untyped-def]
    register_configs()
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


def test_default_arch_is_the_documented_stack() -> None:
    cfg = compose_config("data.images_dir=/x", "data.masks_dir=/y")
    channels = [layer.out_channels for layer in cfg.arch.layers]
    assert channels == [16, 32, 32]
    assert cfg.arch.stdev_factor == pytest.approx(0.01)
    assert all(layer.kernel_size == 3 for layer in cfg.arch.layers)


def test_arch_is_swappable_by_config_group() -> None:
    cfg = compose_config("arch=flim2", "data.images_dir=/x", "data.masks_dir=/y")
    assert [layer.out_channels for layer in cfg.arch.layers] == [16, 32]


def test_missing_required_data_dirs_is_an_error() -> None:
    cfg = compose_config()
    with pytest.raises(Exception, match="mandatory|missing"):
        _ = cfg.data.images_dir


def test_schema_rejects_a_mistyped_field() -> None:
    with pytest.raises((ValidationError, ConfigCompositionException)):
        compose_config("data.images_dir=/x", "data.masks_dir=/y", "seed=not_an_int")


def test_layer_defaults_mirror_the_c_pipeline() -> None:
    """Pooling and activation default to 3x3 max-pool, stride 2, ReLU."""
    layer = LayerSpec()
    assert (layer.pool_type, layer.pool_size, layer.pool_stride) == ("max", 3, 2)
    assert layer.relu is True
    assert layer.dilation == 1


def test_per_layer_component_overrides_default_to_none() -> None:
    """'Unset' must be falsy so the Learner's `layer.scorer or self.scorer` works."""
    layer = LayerSpec()
    assert layer.scorer is None
    assert layer.selector is None
    assert layer.metric is None
    assert layer.allocator is None


def test_arch_spec_is_structured_config_compatible() -> None:
    arch = ArchSpec(layers=[LayerSpec(kernel_size=5, out_channels=8)])
    conf = OmegaConf.structured(arch)
    assert conf.layers[0].kernel_size == 5

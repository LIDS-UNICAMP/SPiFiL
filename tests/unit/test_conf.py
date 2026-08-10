"""Every layer-0 config group variant composes and instantiates.

Cheap insurance against the failure mode where a constructor signature changes
and the YAML that feeds it silently rots — nothing else type-checks the gap
between ``conf/`` and the classes it names. Every group and every variant is
covered; wiring a composed config into a Learner, and running the CLI for real,
is ``test_cli.py``.

Hydra lives at the boundary: these tests are the only place outside ``cli.py``
that may import it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import spifil
from spifil.allocation import FilterAllocator
from spifil.color import ColorTransform
from spifil.config import register_configs
from spifil.metrics import DistanceMetric
from spifil.scoring import Scorer
from spifil.selection import Selector
from spifil.strategies import FitStrategy
from spifil.superpixels.base import SeedExtractor, SuperpixelAlgorithm

# Resolved from the installed package, not the checkout: conf/ ships
# inside spifil so the CLI works from a wheel, and these tests should
# compose exactly what a user gets.
CONF_DIR = str((Path(spifil.__file__).parent / "conf").resolve())

REQUIRED = ["data.images_dir=images", "data.masks_dir=masks"]


@pytest.fixture(autouse=True)
def _registered() -> None:
    register_configs()


def _compose(overrides: list[str], *, with_hydra: bool = False):
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[*REQUIRED, *overrides],
            return_hydra_config=with_hydra,
        )


def test_defaults_are_the_documented_components() -> None:
    """The shipped defaults are the ones the README documents.

    Two of them are load-bearing and easy to change by accident. The default
    superpixel algorithm is ``slic``, so a plain install runs end to end
    without the optional DISF backend; and the default seed extractor is
    ``medoid``, the rule the SPiFiL paper defines. Swapping either changes
    every filter downstream without failing anything, which is exactly why
    they are asserted here.
    """
    cfg = _compose([])

    assert instantiate(cfg.color).__class__.__name__ == "LabNorm"
    assert instantiate(cfg.superpixels).__class__.__name__ == "SLIC"
    assert instantiate(cfg.seed_extractor).__class__.__name__ == "Medoids"
    assert instantiate(cfg.metric).__class__.__name__ == "Mahalanobis"
    assert instantiate(cfg.metric).estimator == "shrinkage"
    assert instantiate(cfg.scorer).__class__.__name__ == "FisherScorer"
    assert instantiate(cfg.allocator).__class__.__name__ == "UniformAllocator"
    assert instantiate(cfg.selector).__class__.__name__ == "DiversitySelector"
    assert instantiate(cfg.selector).alpha == 0.5
    assert instantiate(cfg.selector).pool_factor == 3
    assert instantiate(cfg.strategy).__class__.__name__ == "SequentialStrategy"
    assert cfg.n_superpixels == 100
    assert cfg.resume_from is None


@pytest.mark.parametrize("variant", ["disf", "slic"])
def test_superpixel_variants_instantiate_to_the_protocol(variant: str) -> None:
    cfg = _compose([f"superpixels={variant}"])

    assert isinstance(instantiate(cfg.superpixels), SuperpixelAlgorithm)


@pytest.mark.parametrize("variant", ["geodesic", "centroid", "medoid"])
def test_seed_extractor_variants_instantiate_to_the_protocol(variant: str) -> None:
    cfg = _compose([f"seed_extractor={variant}"])

    assert isinstance(instantiate(cfg.seed_extractor), SeedExtractor)


@pytest.mark.parametrize("variant", ["labnorm"])
def test_color_variants_instantiate_to_the_protocol(variant: str) -> None:
    cfg = _compose([f"color={variant}"])

    assert isinstance(instantiate(cfg.color), ColorTransform)


@pytest.mark.parametrize("variant", ["mahalanobis", "euclidean"])
def test_metric_variants_instantiate_to_the_protocol(variant: str) -> None:
    cfg = _compose([f"metric={variant}"])

    assert isinstance(instantiate(cfg.metric), DistanceMetric)


@pytest.mark.parametrize("variant", ["fisher", "distance_sum"])
def test_scorer_variants_instantiate_to_the_protocol(variant: str) -> None:
    cfg = _compose([f"scorer={variant}"])

    assert isinstance(instantiate(cfg.scorer), Scorer)


@pytest.mark.parametrize("variant", ["uniform"])
def test_allocator_variants_instantiate_to_the_protocol(variant: str) -> None:
    cfg = _compose([f"allocator={variant}"])

    assert isinstance(instantiate(cfg.allocator), FilterAllocator)


@pytest.mark.parametrize("variant", ["diversity", "topn"])
def test_selector_variants_instantiate_to_the_protocol(variant: str) -> None:
    cfg = _compose([f"selector={variant}"])

    assert isinstance(instantiate(cfg.selector), Selector)


@pytest.mark.parametrize("variant", ["sequential"])
def test_strategy_variants_instantiate_to_the_protocol(variant: str) -> None:
    cfg = _compose([f"strategy={variant}"])

    assert isinstance(instantiate(cfg.strategy), FitStrategy)


@pytest.mark.parametrize(
    ("variant", "n_layers"),
    [("flim2", 2), ("flim3", 3), ("flim4", 4), ("paper", 3)],
)
def test_arch_variants_compose(variant: str, n_layers: int) -> None:
    cfg = _compose([f"arch={variant}"])

    assert len(cfg.arch.layers) == n_layers
    assert cfg.arch.stdev_factor == 0.01


def test_component_parameters_are_overridable() -> None:
    cfg = _compose(["superpixels=disf", "superpixels.n_init_seeds=250"])

    assert instantiate(cfg.superpixels).n_init_seeds == 250


def test_working_directory_is_never_changed() -> None:
    """chdir=false is pinned so every path in a run stays explicit."""
    cfg = _compose([], with_hydra=True)

    assert cfg.hydra.job.chdir is False

"""The Hydra boundary: a composed config builds a Learner, and the CLI runs.

The design draws the line at ``cli.py`` — components and the Learner receive
instantiated objects and never a ``DictConfig``, so the package stays usable as
a plain library. :func:`test_no_dictconfig_reaches_the_learner` is that rule as
an assertion rather than a convention.

The multirun test shells out to the console script. It is the only test that
exercises ``@hydra.main`` — run directories, the resolved-config dump, sweep
subdirectories — and none of that is reachable through the Compose API.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

import spifil
from spifil.cli import build_learner
from spifil.config import register_configs
from spifil.strategies import SequentialStrategy
from tests.conftest import MINIMAL

# Resolved from the installed package, not the checkout: conf/ ships
# inside spifil so the CLI works from a wheel, and these tests should
# compose exactly what a user gets.
CONF_DIR = str((Path(spifil.__file__).parent / "conf").resolve())
PROJECT = Path(__file__).parents[2]


@pytest.fixture(autouse=True)
def _registered() -> None:
    register_configs()


def composed(overrides: list[str]) -> DictConfig:
    """Compose the root config against the tiny fixture dataset."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                f"data.images_dir={MINIMAL.images}",
                f"data.masks_dir={MINIMAL.masks}",
                "output_dir=.",
                *overrides,
            ],
        )


class TestBuildLearner:
    def test_defaults_build_the_documented_pipeline(self) -> None:
        learn = build_learner(composed(["superpixels=slic", "seed_extractor=centroid"]))

        assert learn.n_layers == 3
        assert learn.arch.stdev_factor == 0.01
        assert learn.scorer.__class__.__name__ == "FisherScorer"
        assert learn.metric.__class__.__name__ == "Mahalanobis"
        assert isinstance(learn.strategy, SequentialStrategy)
        assert len(learn.data) == 2

    def test_group_and_dotted_overrides_both_reach_the_learner(self) -> None:
        learn = build_learner(
            composed(
                [
                    "superpixels=slic",
                    "seed_extractor=centroid",
                    "scorer=distance_sum",
                    "selector.alpha=0.7",
                    "arch=flim2",
                    "n_superpixels=25",
                ]
            )
        )

        assert learn.scorer.__class__.__name__ == "DistanceSumScorer"
        assert learn.selector.alpha == 0.7  # type: ignore[attr-defined]
        assert learn.n_layers == 2
        assert learn.n_superpixels == 25

    def test_no_dictconfig_reaches_the_learner(self) -> None:
        """Hydra stays at the boundary."""
        learn = build_learner(composed(["superpixels=slic", "seed_extractor=centroid"]))

        candidates: list[Any] = [
            learn.arch,
            *learn.arch.layers,
            learn.color,
            learn.superpixels,
            learn.seed_extractor,
            learn.metric,
            learn.scorer,
            learn.allocator,
            learn.selector,
            learn.strategy,
        ]
        for component in candidates:
            assert not isinstance(component, DictConfig), component

    def test_arch_layers_become_real_layer_specs(self) -> None:
        learn = build_learner(
            composed(["arch=flim4", "superpixels=slic", "seed_extractor=centroid"])
        )

        assert [spec.out_channels for spec in learn.arch.layers] == [16, 32, 48, 64]
        assert all(spec.kernel_size == 3 for spec in learn.arch.layers)
        # Per-layer component overrides are a Python-API feature, so the
        # composed specs must leave them unset rather than half-wired.
        assert all(spec.scorer is None for spec in learn.arch.layers)


@pytest.mark.slow
class TestConsoleScript:
    def run_cli(
        self,
        tmp_path: Path,
        *,
        flags: list[str] | None = None,
        extra: list[str] | None = None,
    ) -> subprocess.CompletedProcess:
        # Hydra's parser puts `overrides` in a `nargs="*"` positional, so its
        # own flags have to come before the first override, not after.
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "spifil.cli",
                *(flags or []),
                f"data.images_dir={MINIMAL.images}",
                f"data.masks_dir={MINIMAL.masks}",
                "arch=flim2",
                "superpixels=slic",
                "seed_extractor=centroid",
                "n_superpixels=20",
                f"hydra.run.dir={tmp_path}/run",
                f"hydra.sweep.dir={tmp_path}/sweep",
                *(extra or []),
            ],
            cwd=PROJECT,
            capture_output=True,
            text=True,
        )

    def test_a_single_run_exports_a_model(self, tmp_path: Path) -> None:
        result = self.run_cli(tmp_path)
        assert result.returncode == 0, result.stderr

        run = tmp_path / "run"
        assert (run / "models" / "model.pt").exists()
        assert (run / "models" / "labels1.txt").exists()
        assert (run / "layers.csv").exists()
        assert (run / "checkpoints" / "layer2.done").exists()
        # Hydra writes the resolved config itself; no config-dump callback.
        assert (run / ".hydra" / "config.yaml").exists()

        described = json.loads((run / "models" / "architecture.json").read_text())
        assert len(described["layers"]) == 2

    def test_a_multirun_sweep_gives_one_directory_per_point(
        self, tmp_path: Path
    ) -> None:
        result = self.run_cli(
            tmp_path, flags=["--multirun"], extra=["selector.alpha=0.3,0.7"]
        )
        assert result.returncode == 0, result.stderr

        points = sorted(p for p in (tmp_path / "sweep").iterdir() if p.is_dir())
        assert len(points) == 2
        for point in points:
            assert (point / "models" / "model.pt").exists()
            assert (point / "models" / "labels1.txt").read_text().split()

        # Each point is its own experiment with its own resolved config; if the
        # sweep were collapsing, both directories would record the same alpha.
        recorded = [
            (point / ".hydra" / "overrides.yaml").read_text() for point in points
        ]
        assert "selector.alpha=0.3" in recorded[0]
        assert "selector.alpha=0.7" in recorded[1]

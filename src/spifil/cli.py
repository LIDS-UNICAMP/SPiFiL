"""The ``spifil-fit`` entry point.

Hydra composes the experiment and ``instantiate`` turns each config group into
an object, but nothing below this file ever sees a ``DictConfig``. The
Learner and every component receive constructed objects, so the package stays
a plain library that happens to ship a CLI.

``version_base`` is pinned so a Hydra upgrade cannot silently change composition
semantics, and ``hydra.job.chdir=false`` (set in ``conf/config.yaml``) keeps the
working directory put, Hydra's run directory is the experiment's output
directory, addressed explicitly rather than by ``cd``.
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from spifil.callbacks import Checkpoint, CSVLogger, ProgressBar
from spifil.config import ArchSpec, LayerSpec, register_configs
from spifil.data import SpifilDataset
from spifil.learner import Learner

__all__ = ["build_learner", "main"]

# Must run before Hydra composes anything: it is what type-checks conf/ against
# the dataclasses in spifil.config.
register_configs()


def build_learner(cfg: DictConfig) -> Learner:
    """Instantiate every component and wire them into a :class:`Learner`.

    Kept separate from :func:`main` so tests can compose a config with Hydra's
    Compose API and assert on the resulting objects without running a fit.
    """
    arch = ArchSpec(
        layers=[
            LayerSpec(**OmegaConf.to_container(layer, resolve=True))  # type: ignore[arg-type]
            for layer in cfg.arch.layers
        ],
        stdev_factor=float(cfg.arch.stdev_factor),
    )
    data = SpifilDataset.from_folders(cfg.data.images_dir, cfg.data.masks_dir)
    output_dir = str(cfg.output_dir)

    return Learner(
        data,
        arch,
        color=hydra.utils.instantiate(cfg.color),
        superpixels=hydra.utils.instantiate(cfg.superpixels),
        seed_extractor=hydra.utils.instantiate(cfg.seed_extractor),
        metric=hydra.utils.instantiate(cfg.metric),
        scorer=hydra.utils.instantiate(cfg.scorer),
        allocator=hydra.utils.instantiate(cfg.allocator),
        selector=hydra.utils.instantiate(cfg.selector),
        strategy=hydra.utils.instantiate(cfg.strategy),
        n_superpixels=int(cfg.n_superpixels),
        device=str(cfg.device),
        cbs=[
            Checkpoint(f"{output_dir}/checkpoints", resume_from=cfg.resume_from),
            CSVLogger(output_dir),
            ProgressBar(),
        ],
    )


# ``conf/`` lives *inside* the package, so it ships in the wheel and this path
# resolves the same from a source checkout and from site-packages. Pointing at
# the project root instead (``../../conf``) works only for an editable install:
# everywhere else the directory is simply absent and the console script dies at
# startup with "Primary config directory not found".
@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Fit a model and export it into Hydra's run directory."""
    torch.manual_seed(int(cfg.seed))

    learn = build_learner(cfg)
    learn.fit()
    learn.export(f"{cfg.output_dir}/models")


if __name__ == "__main__":  # pragma: no cover
    main()

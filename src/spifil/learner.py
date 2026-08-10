"""The orchestrator.

The :class:`Learner` owns exactly two things the components do not: the **order**
stages run in, and the **state** they hand each other. Everything else is
injected --- colour transform, superpixels, seed extractor, metric, scorer,
allocator, selector, conv block, fit strategy --- so swapping any one of them
is a constructor argument, never an edit here (every swappable concern
goes through a protocol).

Layer numbering starts at zero: **layer 0 is the colour transform's
output**, and layer ``L >= 1`` is conv block ``L``'s. So
``arch.layers[L - 1]`` is layer ``L``'s spec, and :attr:`Learner.state` is keyed
``0..n_layers``.

The ordering that matters most is **where scoring sits**.
A scoring pass over layer ``L``'s features uses **layer ``L+1``'s kernel
config** for its adjacency, because the patches it scores are the ones layer
``L+1`` will convolve. It therefore belongs to the *end* of layer ``L``, not
the start of layer ``L+1``: :meth:`prepare` scores layer 0, and :meth:`fit_layer`
scores layer ``L`` after encoding it, skipping the pass entirely for the last
layer, which has no successor to serve. That arrangement is what makes a
layer's ranks available before its own filters are selected.

Consequently the components a scoring pass uses are **layer ``L+1``'s**
(``spec.scorer``, ``spec.metric``): the ranks it computes exist to be consumed
by ``L+1``, so they are configured by whoever consumes them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from spifil import preparation
from spifil.allocation import FilterAllocator, UniformAllocator
from spifil.color import ColorTransform, LabNorm
from spifil.config import ArchSpec, LayerSpec
from spifil.data import SpifilDataset
from spifil.metrics import DistanceMetric, Mahalanobis
from spifil.nn.blocks import ConvBlockFactory, SpifilConvBlock
from spifil.nn.builder import FilterBank, build_filters
from spifil.nn.encoder import describe_color
from spifil.nn.model import FORMAT_VERSION, SpifilNet
from spifil.patches import extract_patches
from spifil.scoring import FisherScorer, Scorer, per_class_ranks, scatter_ranks
from spifil.selection import DiversitySelector, Selector, candidate_order
from spifil.strategies import FitStrategy, SequentialStrategy
from spifil.superpixels.base import SeedExtractor, SuperpixelAlgorithm
from spifil.superpixels.centers import Medoids
from spifil.superpixels.slic import SLIC
from spifil.types import PatchSet, Seeds

__all__ = [
    "CancelFitException",
    "CancelLayerException",
    "EVENTS",
    "LayerState",
    "Learner",
]

EVENTS = (
    "before_fit",
    "before_layer",
    "after_scoring",
    "after_selection",
    "after_model_build",
    "after_encode",
    "after_layer",
    "after_fit",
)
"""Callback events, in the order a full fit emits them."""


class CancelLayerException(Exception):
    """Raised by a callback in ``before_layer`` to skip the layer.

    The Checkpoint callback's resume path: it restores the layer's state and
    then cancels, which is how a finished layer is skipped on a resumed run.
    """


class CancelFitException(Exception):
    """Raised by a callback to stop the fit cleanly; ``after_fit`` still runs."""


@dataclass
class LayerState:
    """Everything one layer produced, addressable rather than loop-local.

    Holding each layer's outputs in a store rather than in local variables
    is what lets :meth:`Learner.refit_from` drop and rebuild a suffix of the
    stack, and what makes cross-layer analysis possible after a fit.

    Attributes
    ----------
    features
        One ``(C, H, W)`` tensor per image, in dataset order. Layer 0's are the
        colour transform's output; later layers' are the block's.
    seeds
        One :class:`~spifil.types.Seeds` per image, same order. Ranks are as of
        the last scoring pass — ``0`` for every seed until one has run.
    patches
        The scoring pass' global patch set: every seed of every image, this
        layer's features under the **next** layer's adjacency. ``None`` until
        the pass runs (and on the last layer, which never scores); rebuilt
        lazily by :meth:`Learner.fit_layer` when a resumed run needs it.
    scores
        ``(n,)`` raw scorer output aligned with :attr:`patches`' rows, or
        ``None``. A resumed run has ranks but no scores: a checkpoint stores
        the rank, which is what later layers consume.
    ranks
        ``(n,)`` per-class ranks aligned with :attr:`patches`' rows. Redundant
        with :attr:`seeds` by construction; selection needs them in
        patch-row order.
    bank
        The filters this layer was built from, ``None`` for layer 0.
    selected
        The patches those filters *are*, in filter order — so filter ``k``'s
        provenance (which seed of which image) is recoverable after the fit.
        ``None`` for layer 0 and for a layer restored from a checkpoint, which
        keeps the filters but not the patches behind them.
    superpixel_labels
        Layer 0 only: the ``(H, W)`` region maps the seeds were reduced from.
        Not recoverable from the seeds, and what makes seed placement
        reviewable.
    """

    features: list[Tensor]
    seeds: list[Seeds]
    patches: PatchSet | None = None
    scores: Tensor | None = None
    ranks: Tensor | None = None
    bank: FilterBank | None = None
    selected: PatchSet | None = None
    superpixel_labels: list[npt.NDArray[np.int32]] | None = field(default=None)


class Learner:
    """Fits a :class:`~spifil.nn.model.SpifilNet` layer by layer.

    Parameters
    ----------
    data
        The images, masks and class labels to learn from.
    arch
        The layer stack. ``arch.layers[L - 1]`` configures layer ``L``.
    superpixels, seed_extractor
        Layer 0: how to segment the colour features, and how to reduce each
        region to one seed. Default to
        :class:`~spifil.superpixels.slic.SLIC` (no optional dependency) and
        :class:`~spifil.superpixels.centers.Medoids`. Pass
        :class:`~spifil.superpixels.disf.DISF` to reproduce the paper.
    metric, scorer
        How seeds are ranked; default to
        :class:`~spifil.metrics.Mahalanobis` and
        :class:`~spifil.scoring.FisherScorer`.
    selector, allocator
        Which patches become filters and how many each class gets; default to
        :class:`~spifil.selection.DiversitySelector` and
        :class:`~spifil.allocation.UniformAllocator`.
    color
        Defaults to :class:`~spifil.color.LabNorm`. Its ``out_channels`` sets
        the first block's ``in_channels``.
    block_factory
        Builds each conv block; defaults to
        :class:`~spifil.nn.blocks.SpifilConvBlock`.
    strategy
        The layer loop; defaults to
        :class:`~spifil.strategies.SequentialStrategy`.
    cbs
        Callbacks, notified at every event in :data:`EVENTS`.
    n_superpixels
        Target number of regions per image, i.e. candidate seeds before
        ranking.
    device
        Where the tensor work happens. Feature maps and patches are moved here;
        seeds stay on the CPU, being small integer bookkeeping.

    Notes
    -----
    Per-layer component overrides are resolved through
    :meth:`component_for`: a :class:`~spifil.config.LayerSpec` may carry its own
    ``scorer``, ``selector``, ``metric`` or ``allocator``, and ``None`` means
    "use the Learner's". This is a Python-API feature until the Hydra wiring
    lands --- pass instantiated objects into the ``LayerSpec``.
    """

    def __init__(
        self,
        data: SpifilDataset,
        arch: ArchSpec,
        *,
        superpixels: SuperpixelAlgorithm | None = None,
        seed_extractor: SeedExtractor | None = None,
        metric: DistanceMetric | None = None,
        scorer: Scorer | None = None,
        selector: Selector | None = None,
        allocator: FilterAllocator | None = None,
        color: ColorTransform | None = None,
        block_factory: ConvBlockFactory = SpifilConvBlock,
        strategy: FitStrategy | None = None,
        cbs: Iterable[Any] = (),
        n_superpixels: int = 100,
        device: str | torch.device = "cpu",
    ) -> None:
        if not arch.layers:
            raise ValueError("Learner needs an ArchSpec with at least one layer")

        self.data = data
        self.arch = arch
        # The defaults are the ones conf/config.yaml composes, so the Python
        # API and `spifil-fit` describe the same pipeline.
        self.superpixels = superpixels if superpixels is not None else SLIC()
        self.seed_extractor = (
            seed_extractor if seed_extractor is not None else Medoids()
        )
        self.metric = metric if metric is not None else Mahalanobis()
        self.scorer = scorer if scorer is not None else FisherScorer()
        self.selector = selector if selector is not None else DiversitySelector()
        self.allocator = allocator if allocator is not None else UniformAllocator()
        self.color = color if color is not None else LabNorm()
        self.block_factory = block_factory
        self.strategy = strategy if strategy is not None else SequentialStrategy()
        self.cbs = list(cbs)
        self.n_superpixels = n_superpixels
        self.device = torch.device(device)

        self.model = SpifilNet(
            arch, in_channels=self.color.out_channels, block_factory=block_factory
        ).to(self.device)

        self.state: dict[int, LayerState] = {}
        self.layer: int | None = None
        """The layer currently being fitted --- what a callback reads to know
        where it is. ``None`` outside :meth:`fit_layer`."""
        self.selected: PatchSet | None = None
        """The patches the current layer's selector chose; set before
        ``after_selection`` so a callback can inspect the picks."""

    # ------------------------------------------------------------------
    # Component resolution
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """How many conv layers the architecture has; layers are ``1..n_layers``."""
        return len(self.arch.layers)

    def spec(self, layer: int) -> LayerSpec:
        """Layer ``layer``'s spec, 1-based."""
        if not 1 <= layer <= self.n_layers:
            raise ValueError(
                f"layer must be between 1 and {self.n_layers} (1-based), got {layer}"
            )
        return self.arch.layers[layer - 1]

    def component_for(self, layer: int, name: str) -> Any:
        """The component layer ``layer`` uses for ``name``, override or default.

        ``name`` is one of ``scorer``, ``selector``, ``metric``, ``allocator``.
        """
        override = getattr(self.spec(layer), name)
        return override if override is not None else getattr(self, name)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def dispatch(self, event: str) -> None:
        """Notify every callback of ``event``, in callback order.

        Callbacks may raise :class:`CancelLayerException` /
        :class:`CancelFitException`; those propagate to whoever owns the loop
        rather than being swallowed here.
        """
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}; expected one of {EVENTS}")
        for cb in self.cbs:
            handler = getattr(cb, event, None)
            if handler is not None:
                handler(self)

    # ------------------------------------------------------------------
    # Layer 0
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Run layer 0 and score it, unless a restored state already holds both.

        Idempotent: calling it twice does nothing the second time, so
        :meth:`fit` can call it unconditionally. Use :meth:`score_layer` to
        force a re-score after changing the scorer.

        A resumed run does not re-score layer 0. Scoring is deterministic, so
        the ranks it would recompute are the ones it just restored; skipping
        the pass only avoids paying O(N²) twice for the same answer.
        """
        if 0 not in self.state:
            prepared = list(
                preparation.prepare(
                    self.data,
                    superpixels=self.superpixels,
                    seed_extractor=self.seed_extractor,
                    n_superpixels=self.n_superpixels,
                    color=self.color,
                )
            )
            self.state[0] = LayerState(
                features=[
                    torch.from_numpy(image.features).to(self.device)
                    for image in prepared
                ],
                seeds=[image.seeds for image in prepared],
                superpixel_labels=[image.superpixel_labels for image in prepared],
            )
        if not any(bool((seeds.ranks > 0).any()) for seeds in self.state[0].seeds):
            self.score_layer(0)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def extract_layer_patches(self, layer: int) -> None:
        """Build ``state[layer].patches``/``.ranks`` — every seed, next layer's window.

        Split out from :meth:`score_layer` because a resumed run needs the
        patches without paying for the O(N²) scoring again: the ranks it
        would recompute are already in the restored seeds.
        """
        state = self.state[layer]
        # arch.layers is 0-based, so index `layer` is layer+1's spec — the
        # window the patches being scored will eventually be convolved with.
        spec = self.arch.layers[layer]
        patch_sets = [
            extract_patches(features, seeds, spec, image_id=image_id)
            for image_id, (features, seeds) in enumerate(
                zip(state.features, state.seeds, strict=True)
            )
        ]
        state.patches = PatchSet.cat(patch_sets).to(self.device)
        state.ranks = torch.cat([seeds.ranks for seeds in state.seeds]).to(self.device)

    def score_layer(self, layer: int) -> None:
        """Score layer ``layer``'s seeds and write per-class ranks back.

        One global dataset over all images, one covariance, and one rank per
        seed within its class.
        Seeds absent from the scored set keep rank 0 and drop out downstream —
        :func:`~spifil.scoring.scatter_ranks` enforces that.
        """
        if not 0 <= layer < self.n_layers:
            raise ValueError(
                f"only layers 0..{self.n_layers - 1} are scored (layer "
                f"{self.n_layers} has no successor to score for), got {layer}"
            )
        self.extract_layer_patches(layer)
        state = self.state[layer]
        assert state.patches is not None  # just built

        scorer = self.component_for(layer + 1, "scorer")
        metric = self.component_for(layer + 1, "metric")
        scores = scorer(state.patches, metric)
        ranks = per_class_ranks(scores, state.patches.labels)

        state.scores = scores
        state.ranks = ranks
        state.seeds = scatter_ranks(
            state.patches.to("cpu"), ranks.to("cpu"), state.seeds
        )

        # `layer` is the layer being *scored*, which is where a callback needs
        # to think it is. Restored rather than cleared because fit_layer scores
        # from inside its own layer and must stay there.
        outer, self.layer = self.layer, layer
        try:
            self.dispatch("after_scoring")
        finally:
            self.layer = outer

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Prepare if needed, then run the strategy over the layers."""
        self.dispatch("before_fit")
        try:
            self.prepare()
            self.strategy.run(self)
        except CancelFitException:
            pass
        self.dispatch("after_fit")

    def fit_layer(self, layer: int) -> None:
        """Select, build, encode and re-score one layer.

        Requires layer ``layer - 1``'s state, which is what carries the ranked
        seeds this layer selects from.
        """
        spec = self.spec(layer)
        if layer - 1 not in self.state:
            raise RuntimeError(
                f"layer {layer} needs layer {layer - 1}'s state; call "
                f"{'prepare()' if layer == 1 else f'fit_layer({layer - 1})'} first"
            )

        self.layer = layer
        try:
            self._fit_layer(layer, spec)
        except CancelLayerException:
            return
        finally:
            # Whatever leaves this method — a cancelled layer, a cancelled fit,
            # or a genuine error --- must not leave callbacks reading a layer
            # the Learner is no longer inside.
            self.layer = None

    def _fit_layer(self, layer: int, spec: LayerSpec) -> None:
        self.dispatch("before_layer")

        previous = self.state[layer - 1]
        if previous.patches is None:
            self.extract_layer_patches(layer - 1)
        patches, ranks = previous.patches, previous.ranks
        assert patches is not None and ranks is not None  # just ensured

        # Step 1: only ranked seeds are candidates.
        ranked = ranks > 0
        candidates, candidate_ranks = patches.select(ranked), ranks[ranked]
        if len(candidates) == 0:
            raise RuntimeError(
                f"layer {layer}: no seed of layer {layer - 1} carries a rank, so "
                "there is nothing to select filters from"
            )
        # A resumed run has ranks but no scores, which the default
        # score-blind allocator does not need. See LayerState.scores.
        scores = (
            previous.scores[ranked]
            if previous.scores is not None
            else torch.zeros(len(candidates), device=candidates.feats.device)
        )

        # Steps 2-3: budget per class, then the diversity-aware greedy pick.
        allocator = self.component_for(layer, "allocator")
        selector = self.component_for(layer, "selector")
        n_per_class = allocator(layer, spec, candidates, scores)
        mask = selector(candidates, candidate_ranks, n_per_class)

        # A selector returns a mask, which says nothing about order; the
        # filter bank needs one. Emit the picks in canonical candidate order
        # --- see `candidate_order`.
        picks = mask.nonzero(as_tuple=True)[0]
        self.selected = candidates.select(
            picks[candidate_order(candidates.select(picks), candidate_ranks[picks])]
        )
        self.dispatch("after_selection")

        # Step 4: fold the z-score into conv weights and install them.
        bank = build_filters(self.selected, spec, stdev_factor=self.arch.stdev_factor)
        self.model.set_filters(layer, bank)
        self.dispatch("after_model_build")

        # Steps 5-6: encode every image, and follow the seeds onto the pooled grid.
        block = self.model.block(layer)
        with torch.no_grad():
            features = [
                block(image.unsqueeze(0)).squeeze(0) for image in previous.features
            ]
        stride = projection_stride(spec)
        self.state[layer] = LayerState(
            features=features,
            seeds=[seeds.project(stride) for seeds in previous.seeds],
            bank=bank,
            selected=self.selected,
        )
        self.dispatch("after_encode")

        # Step 7: refresh the ranks the next layer will consume. The last layer
        # has no next layer, so its seeds keep the ranks projection gave them.
        if layer < self.n_layers:
            self.score_layer(layer)

        self.dispatch("after_layer")

    def refit_from(self, layer: int) -> None:
        """Drop layers ``>= layer`` and fit them again.

        State above the cut is invalid the moment ``layer``'s filters
        change, because it was encoded from features that no longer exist.
        """
        if not 1 <= layer <= self.n_layers:
            raise ValueError(
                f"layer must be between 1 and {self.n_layers} (1-based), got {layer}"
            )
        for stale in [key for key in self.state if key >= layer]:
            del self.state[stale]
        for current in range(layer, self.n_layers + 1):
            self.fit_layer(current)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, out_dir: str | Path) -> Path:
        """Write the fitted model as a self-describing bundle.

        Produces ``model.pt`` (a plain ``state_dict``), ``architecture.json``
        and ``labels{L}.txt`` — the class each filter came from.
        :func:`spifil.nn.load_model` reads it back as
        a :class:`~spifil.nn.model.SpifilNet`, and
        :func:`spifil.nn.load_encoder` as a
        :class:`~spifil.nn.encoder.SpifilEncoder` with the colour transform
        attached.

        ``architecture.json`` describes the stack **as fitted**, which is not
        always what the spec asked for: per-class integer division can
        under-produce filters and a narrow layer makes the next block narrower,
        so ``out_channels`` is what exists and ``target_out_channels`` what was
        requested. It also records **which colour transform** produced the
        features the filters were fitted on. That is not bookkeeping: transforms
        that agree on band count and on nothing else (Lab, RGB, another colour
        space) are interchangeable as far as tensor shapes are concerned, so
        without it a bundle can be fed the wrong input and merely produce
        wrong numbers.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), out / "model.pt")

        layers = []
        for layer in range(1, self.n_layers + 1):
            spec = self.spec(layer)
            block = self.model.block(layer)
            layers.append(
                {
                    "layer": layer,
                    "kernel_size": spec.kernel_size,
                    "dilation": spec.dilation,
                    "out_channels": block.out_channels,
                    "target_out_channels": spec.out_channels,
                    "pool_type": spec.pool_type,
                    "pool_size": spec.pool_size,
                    "pool_stride": spec.pool_stride,
                    "relu": spec.relu,
                }
            )
        (out / "architecture.json").write_text(
            json.dumps(
                {
                    "format": FORMAT_VERSION,
                    "in_channels": self.color.out_channels,
                    "color": describe_color(self.color),
                    "stdev_factor": self.arch.stdev_factor,
                    "layers": layers,
                },
                indent=2,
            )
            + "\n"
        )

        for layer in range(1, self.n_layers + 1):
            state = self.state.get(layer)
            bank = state.bank if state is not None else None
            if bank is None:
                continue
            (out / f"labels{layer}.txt").write_text(
                "".join(f"{label}\n" for label in bank.labels.tolist())
            )
        return out


def projection_stride(spec: LayerSpec) -> int:
    """How far a layer's seeds move — its pooling stride, or 1 if it never pools.

    A block with ``pool_type="none"`` does not shrink its grid, so its seeds
    stay where they are.
    """
    return 1 if spec.pool_type == "none" else spec.pool_stride

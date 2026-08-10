"""The Learner's orchestration: state store, events, overrides, refit, export.

These tests are about orchestration rather than numbers: the order stages run
in, what survives in :attr:`~spifil.learner.Learner.state`, and the API the
Learner offers around the loop.

They run on the two-image ``minimal`` fixture with SLIC and centroid seeds:
full images (so shapes, masks and the colour transform are exercised) with no
optional dependency, and small enough that a two-layer fit is milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from spifil import (
    SLIC,
    ArchSpec,
    Centroids,
    DiversitySelector,
    Euclidean,
    FisherScorer,
    LayerSpec,
    Learner,
    Mahalanobis,
    SpifilDataset,
    SpifilNet,
    UniformAllocator,
)
from spifil.learner import EVENTS, CancelFitException, projection_stride
from tests.conftest import MINIMAL


def make_learner(**kwargs: object) -> Learner:
    """A two-layer Learner over the tiny fixture, with PyIFT-free components."""
    data = SpifilDataset.from_folders(MINIMAL.images, MINIMAL.masks)
    defaults: dict[str, object] = {
        "superpixels": SLIC(),
        "seed_extractor": Centroids(),
        "metric": Mahalanobis(),
        "scorer": FisherScorer(),
        "selector": DiversitySelector(alpha=0.5, pool_factor=3),
        "n_superpixels": 20,
    }
    defaults.update(kwargs)
    arch = defaults.pop(
        "arch",
        ArchSpec(
            layers=[
                LayerSpec(kernel_size=3, out_channels=8),
                LayerSpec(kernel_size=3, out_channels=16),
            ]
        ),
    )
    return Learner(data, arch, **defaults)  # type: ignore[arg-type]


class Recorder:
    """A callback that remembers the event sequence and where it happened."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int | None]] = []

    def __getattr__(self, name: str):  # noqa: ANN202 - dispatch-time lookup
        if name not in EVENTS:
            raise AttributeError(name)

        def handler(learn: Learner) -> None:
            self.events.append((name, learn.layer))

        return handler


class TestFit:
    def test_fit_populates_every_layer(self) -> None:
        learn = make_learner()
        learn.fit()

        assert sorted(learn.state) == [0, 1, 2]
        assert learn.state[0].bank is None
        assert learn.state[1].bank is not None
        assert learn.state[2].bank is not None

    def test_features_shrink_by_the_pool_stride(self) -> None:
        learn = make_learner()
        learn.fit()

        height, width = learn.state[0].features[0].shape[1:]
        for layer in (1, 2):
            stride = projection_stride(learn.spec(layer))
            height, width = -(-height // stride), -(-width // stride)
            assert tuple(learn.state[layer].features[0].shape[1:]) == (height, width)
            assert learn.state[layer].seeds[0].grid == (height, width)

    def test_seeds_carry_ranks_after_scoring(self) -> None:
        learn = make_learner()
        learn.prepare()

        assert all((seeds.ranks > 0).any() for seeds in learn.state[0].seeds)
        assert learn.state[0].scores is not None

    def test_last_layer_is_not_rescored(self) -> None:
        """Step 7 is skipped for layer N: there is no layer N+1 to serve."""
        learn = make_learner()
        learn.fit()

        assert learn.state[1].scores is not None
        assert learn.state[2].scores is None
        assert learn.state[2].patches is None

    def test_filters_reach_the_model(self) -> None:
        learn = make_learner()
        learn.fit()

        bank = learn.state[1].bank
        assert bank is not None
        assert torch.equal(learn.model.block(1).conv.weight.detach(), bank.weight)

    def test_selected_patches_are_kept_as_provenance(self) -> None:
        learn = make_learner()
        learn.fit()

        for layer in (1, 2):
            state = learn.state[layer]
            bank, selected = state.bank, state.selected
            assert bank is not None and selected is not None
            assert len(selected) == len(bank)
            assert torch.equal(selected.labels, bank.labels)

    def test_fit_is_deterministic(self) -> None:
        first, second = make_learner(), make_learner()
        first.fit()
        second.fit()

        for layer in (1, 2):
            a, b = first.state[layer].bank, second.state[layer].bank
            assert a is not None and b is not None
            assert torch.equal(a.weight, b.weight)

    def test_prepare_is_idempotent(self) -> None:
        learn = make_learner()
        learn.prepare()
        ranks = [seeds.ranks.clone() for seeds in learn.state[0].seeds]

        learn.prepare()

        for before, after in zip(ranks, learn.state[0].seeds, strict=True):
            assert torch.equal(before, after.ranks)


class TestEvents:
    def test_full_fit_emits_the_documented_sequence(self) -> None:
        recorder = Recorder()
        learn = make_learner(cbs=[recorder])
        learn.fit()

        assert recorder.events == [
            ("before_fit", None),
            ("after_scoring", 0),  # prepare(): layer 0, for layer 1's window
            ("before_layer", 1),
            ("after_selection", 1),
            ("after_model_build", 1),
            ("after_encode", 1),
            ("after_scoring", 1),  # step 7, still inside layer 1
            ("after_layer", 1),
            ("before_layer", 2),
            ("after_selection", 2),
            ("after_model_build", 2),
            ("after_encode", 2),
            ("after_layer", 2),  # no re-score: layer 2 is the last
            ("after_fit", None),
        ]

    def test_scoring_reports_the_layer_being_scored(self) -> None:
        """``learn.layer`` is where a callback thinks it is, layer 0 included."""
        recorder = Recorder()
        learn = make_learner(cbs=[recorder])
        learn.prepare()

        assert recorder.events == [("after_scoring", 0)]
        assert learn.layer is None

    def test_cancel_fit_stops_the_loop_but_not_after_fit(self) -> None:
        class StopBeforeLayer2:
            def before_layer(self, learn: Learner) -> None:
                if learn.layer == 2:
                    raise CancelFitException

        recorder = Recorder()
        learn = make_learner(cbs=[StopBeforeLayer2(), recorder])
        learn.fit()

        assert sorted(learn.state) == [0, 1]
        assert recorder.events[-1] == ("after_fit", None)

    def test_callbacks_need_only_the_events_they_want(self) -> None:
        class OnlyOne:
            def __init__(self) -> None:
                self.seen = 0

            def after_fit(self, learn: Learner) -> None:
                self.seen += 1

        callback = OnlyOne()
        learn = make_learner(cbs=[callback])
        learn.fit()

        assert callback.seen == 1

    def test_unknown_event_is_rejected(self) -> None:
        learn = make_learner()
        with pytest.raises(ValueError, match="unknown event"):
            learn.dispatch("after_lunch")


class TestPerLayerOverrides:
    def test_layer_spec_component_wins_over_the_learner_default(self) -> None:
        override = Euclidean()
        arch = ArchSpec(
            layers=[
                LayerSpec(kernel_size=3, out_channels=8, metric=override),
                LayerSpec(kernel_size=3, out_channels=16),
            ]
        )
        default = Mahalanobis()
        learn = make_learner(arch=arch, metric=default)

        assert learn.component_for(1, "metric") is override
        assert learn.component_for(2, "metric") is default

    def test_an_override_actually_changes_the_result(self) -> None:
        """A different metric for layer 1 must reach the scoring pass.

        The scoring pass over layer 0 is configured by **layer 1**, whose ranks
        it produces (module docstring of ``spifil.learner``), so overriding
        layer 1's metric is what changes layer 0's ranks.
        """
        arch = ArchSpec(
            layers=[
                LayerSpec(kernel_size=3, out_channels=8, metric=Euclidean()),
                LayerSpec(kernel_size=3, out_channels=16),
            ]
        )
        overridden, plain = make_learner(arch=arch), make_learner()
        overridden.prepare()
        plain.prepare()

        assert not torch.equal(
            overridden.state[0].seeds[0].ranks, plain.state[0].seeds[0].ranks
        )


class TestRefit:
    def test_refit_from_rebuilds_the_suffix_identically(self) -> None:
        learn = make_learner()
        learn.fit()
        expected = [learn.state[layer].bank for layer in (1, 2)]

        learn.refit_from(2)

        assert sorted(learn.state) == [0, 1, 2]
        for layer, before in zip((1, 2), expected, strict=True):
            after = learn.state[layer].bank
            assert before is not None and after is not None
            assert torch.equal(before.weight, after.weight)

    def test_refit_from_drops_the_stale_layers_first(self) -> None:
        learn = make_learner()
        learn.fit()
        stale = learn.state[2]

        learn.refit_from(1)

        assert learn.state[2] is not stale

    def test_refit_from_rejects_out_of_range_layers(self) -> None:
        learn = make_learner()
        with pytest.raises(ValueError, match="1-based"):
            learn.refit_from(0)


class TestGuards:
    def test_fitting_before_preparing_says_so(self) -> None:
        learn = make_learner()
        with pytest.raises(RuntimeError, match=r"prepare\(\)"):
            learn.fit_layer(1)

    def test_skipping_a_layer_names_the_missing_one(self) -> None:
        learn = make_learner()
        learn.prepare()
        with pytest.raises(RuntimeError, match=r"fit_layer\(1\)"):
            learn.fit_layer(2)

    def test_the_last_layer_cannot_be_scored(self) -> None:
        learn = make_learner()
        learn.fit()
        with pytest.raises(ValueError, match="no successor"):
            learn.score_layer(2)

    def test_an_empty_arch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one layer"):
            make_learner(arch=ArchSpec(layers=[]))


class TestUnderProduction:
    def test_a_narrow_layer_rewires_the_next_block(self) -> None:
        """7 filters over 2 classes is 3 per class, so layer 1 emits 6, not 7.

        Integer division reaching all the way
        through the Learner: layer 2 has to convolve 6 channels, not 7.
        """
        arch = ArchSpec(
            layers=[
                LayerSpec(kernel_size=3, out_channels=7),
                LayerSpec(kernel_size=3, out_channels=16),
            ]
        )
        learn = make_learner(arch=arch)
        learn.fit()

        bank = learn.state[1].bank
        assert bank is not None and len(bank) == 6
        assert learn.state[1].features[0].shape[0] == 6
        assert learn.model.block(2).in_channels == 6


class TestExport:
    def test_export_writes_a_standalone_model(self, tmp_path: Path) -> None:
        learn = make_learner()
        learn.fit()

        out = learn.export(tmp_path / "models")

        state_dict = torch.load(out / "model.pt", weights_only=True)
        rebuilt = SpifilNet(learn.arch, in_channels=learn.color.out_channels)
        for layer in (1, 2):
            bank = learn.state[layer].bank
            assert bank is not None
            rebuilt.set_filters(layer, bank)
        rebuilt.load_state_dict(state_dict)

        image = learn.state[0].features[0].unsqueeze(0)
        with torch.no_grad():
            assert torch.equal(rebuilt(image), learn.model(image))

    def test_architecture_json_records_what_was_built(self, tmp_path: Path) -> None:
        arch = ArchSpec(
            layers=[
                LayerSpec(kernel_size=3, out_channels=7),
                LayerSpec(kernel_size=3, out_channels=16),
            ]
        )
        learn = make_learner(arch=arch)
        learn.fit()

        learn.export(tmp_path)
        described = json.loads((tmp_path / "architecture.json").read_text())

        assert described["in_channels"] == 3
        assert described["layers"][0]["out_channels"] == 6  # actually produced
        assert described["layers"][0]["target_out_channels"] == 7  # asked for

    def test_filter_labels_are_written_per_layer(self, tmp_path: Path) -> None:
        learn = make_learner()
        learn.fit()

        learn.export(tmp_path)

        for layer in (1, 2):
            bank = learn.state[layer].bank
            assert bank is not None
            written = (tmp_path / f"labels{layer}.txt").read_text().split()
            assert [int(label) for label in written] == bank.labels.tolist()

    def test_export_of_an_unfitted_learner_omits_labels(self, tmp_path: Path) -> None:
        learn = make_learner()

        learn.export(tmp_path)

        assert (tmp_path / "model.pt").exists()
        assert not (tmp_path / "labels1.txt").exists()


class TestStrategy:
    def test_a_custom_strategy_owns_the_loop(self) -> None:
        class OnlyLayer1:
            def run(self, learn: Learner) -> None:
                learn.fit_layer(1)

        learn = make_learner(strategy=OnlyLayer1())
        learn.fit()

        assert sorted(learn.state) == [0, 1]


class TestProjectionStride:
    @pytest.mark.parametrize(
        ("pool_type", "expected"), [("max", 2), ("avg", 2), ("none", 1)]
    )
    def test_a_block_that_does_not_pool_does_not_move_its_seeds(
        self, pool_type: str, expected: int
    ) -> None:
        spec = LayerSpec(kernel_size=3, pool_type=pool_type, pool_stride=2)

        assert projection_stride(spec) == expected

    def test_seeds_keep_their_grid_when_pooling_is_off(self) -> None:
        arch = ArchSpec(
            layers=[LayerSpec(kernel_size=3, out_channels=8, pool_type="none")]
        )
        learn = make_learner(arch=arch, allocator=UniformAllocator())
        learn.fit()

        assert learn.state[1].seeds[0].grid == learn.state[0].seeds[0].grid
        assert (
            learn.state[1].features[0].shape[1:] == learn.state[0].features[0].shape[1:]
        )

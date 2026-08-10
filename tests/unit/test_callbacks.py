"""Checkpoint resume semantics, CSV output, and progress reporting.

These are the mechanics — what lands on
disk, what a marker means, and what ``resume_from`` throws away.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
import torch

from spifil import Callback, Checkpoint, CSVLogger, Learner, ProgressBar
from spifil.learner import EVENTS
from tests.unit.test_learner import make_learner


def fitted(tmp_path: Path, **kwargs: object) -> Learner:
    learn = make_learner(**kwargs)
    learn.fit()
    return learn


class TestCheckpoint:
    def test_a_completed_fit_writes_a_payload_and_marker_per_layer(
        self, tmp_path: Path
    ) -> None:
        fitted(tmp_path, cbs=[Checkpoint(tmp_path)])

        for layer in (0, 1, 2):
            assert (tmp_path / f"layer{layer}.pt").exists()
            assert (tmp_path / f"layer{layer}.done").exists()

    def test_resuming_reuses_the_stored_layers(self, tmp_path: Path) -> None:
        """A fresh Learner over the same checkpoints skips every stored layer."""
        reference = fitted(tmp_path, cbs=[Checkpoint(tmp_path)])

        seen: list[int] = []

        class WatchLayers(Callback):
            def after_encode(self, learn: Learner) -> None:
                assert learn.layer is not None
                seen.append(learn.layer)

        resumed = make_learner(cbs=[Checkpoint(tmp_path), WatchLayers()])
        resumed.fit()

        assert seen == []  # nothing was re-encoded
        for layer in (1, 2):
            expected, got = reference.state[layer].bank, resumed.state[layer].bank
            assert expected is not None and got is not None
            assert torch.equal(got.weight, expected.weight)
            assert torch.equal(got.labels, expected.labels)

    def test_a_restored_layer_puts_its_filters_back_in_the_model(
        self, tmp_path: Path
    ) -> None:
        """Resume has to rebuild the model, not just the state store.

        The next layer encodes with it and ``export`` reads its weights, so a
        checkpoint that restored state but left the convolution at its random
        initialization would produce a model that disagrees with its own
        ``state[L].bank``.
        """
        reference = fitted(tmp_path, cbs=[Checkpoint(tmp_path)])

        resumed = make_learner(cbs=[Checkpoint(tmp_path)])
        resumed.fit()

        for layer in (1, 2):
            bank = reference.state[layer].bank
            assert bank is not None
            assert torch.equal(
                resumed.model.block(layer).conv.weight.detach(), bank.weight
            )

    def test_partial_progress_resumes_and_matches_an_uninterrupted_fit(
        self, tmp_path: Path
    ) -> None:
        class OnlyLayer1:
            def run(self, learn: Learner) -> None:
                learn.fit_layer(1)

        reference = fitted(tmp_path / "full")

        interrupted = make_learner(
            cbs=[Checkpoint(tmp_path / "cp")], strategy=OnlyLayer1()
        )
        interrupted.fit()
        assert sorted(interrupted.state) == [0, 1]
        assert not (tmp_path / "cp" / "layer2.done").exists()

        resumed = make_learner(cbs=[Checkpoint(tmp_path / "cp")])
        resumed.fit()

        for layer in (1, 2):
            expected, got = reference.state[layer].bank, resumed.state[layer].bank
            assert expected is not None and got is not None
            assert torch.equal(got.weight, expected.weight), f"layer {layer}"

    def test_resume_from_discards_that_layer_and_everything_above(
        self, tmp_path: Path
    ) -> None:
        """``resume_from=N``: markers from N up are deleted."""
        fitted(tmp_path, cbs=[Checkpoint(tmp_path)])

        make_learner(cbs=[Checkpoint(tmp_path, resume_from=2)]).fit()

        checkpoint = Checkpoint(tmp_path)
        assert checkpoint.is_complete(0)
        assert checkpoint.is_complete(1)
        assert checkpoint.is_complete(2)  # deleted, then re-run and re-written

    def test_resume_from_actually_re_runs_the_layer(self, tmp_path: Path) -> None:
        recorded: list[int] = []

        class WatchLayers(Callback):
            def after_encode(self, learn: Learner) -> None:
                assert learn.layer is not None
                recorded.append(learn.layer)

        fitted(tmp_path, cbs=[Checkpoint(tmp_path)])
        make_learner(cbs=[Checkpoint(tmp_path, resume_from=2), WatchLayers()]).fit()

        assert recorded == [2]

    def test_a_marker_without_a_payload_is_not_complete(self, tmp_path: Path) -> None:
        """Half a checkpoint is no checkpoint — the layer gets recomputed."""
        fitted(tmp_path, cbs=[Checkpoint(tmp_path)])
        (tmp_path / "layer2.pt").unlink()

        recorded: list[int] = []

        class WatchLayers(Callback):
            def after_encode(self, learn: Learner) -> None:
                assert learn.layer is not None
                recorded.append(learn.layer)

        make_learner(cbs=[Checkpoint(tmp_path), WatchLayers()]).fit()

        assert recorded == [2]

    def test_restored_seeds_keep_their_ranks(self, tmp_path: Path) -> None:
        """Ranks are what the next layer selects from, so they must survive."""
        reference = fitted(tmp_path, cbs=[Checkpoint(tmp_path)])

        restored = Checkpoint(tmp_path)._load(1)

        for expected, got in zip(reference.state[1].seeds, restored.seeds, strict=True):
            assert torch.equal(got.ranks, expected.ranks)
            assert torch.equal(got.coords, expected.coords)
            assert got.grid == expected.grid


class TestCSVLogger:
    def test_layer_summary_has_one_row_per_layer(self, tmp_path: Path) -> None:
        learn = fitted(tmp_path, cbs=[CSVLogger(tmp_path)])

        with (tmp_path / "layers.csv").open() as handle:
            rows = list(csv.DictReader(handle))

        assert [row["layer"] for row in rows] == ["1", "2"]
        for row, layer in zip(rows, (1, 2), strict=True):
            bank = learn.state[layer].bank
            assert bank is not None
            assert int(row["n_selected"]) == len(bank)
            assert row["labels"].split() == [
                str(label) for label in bank.labels.tolist()
            ]

    def test_scores_are_logged_for_every_scoring_pass(self, tmp_path: Path) -> None:
        learn = fitted(tmp_path, cbs=[CSVLogger(tmp_path)])

        with (tmp_path / "scores.csv").open() as handle:
            rows = list(csv.DictReader(handle))

        # One pass per layer that has a successor: layers 0 and 1 here.
        assert {row["layer"] for row in rows} == {"0", "1"}
        for layer in (0, 1):
            patches = learn.state[layer].patches
            assert patches is not None
            assert sum(row["layer"] == str(layer) for row in rows) == len(patches)

    def test_a_rerun_starts_from_empty_files(self, tmp_path: Path) -> None:
        """``before_fit`` truncates, so a second fit does not append to the first."""
        fitted(tmp_path, cbs=[CSVLogger(tmp_path)])
        first = (tmp_path / "layers.csv").read_text()

        fitted(tmp_path, cbs=[CSVLogger(tmp_path)])

        assert (tmp_path / "layers.csv").read_text() == first


class TestProgressBar:
    def test_it_reports_to_the_stream_it_is_given(self, tmp_path: Path) -> None:
        stream = io.StringIO()

        learn = fitted(tmp_path, cbs=[ProgressBar(stream=stream)])

        output = stream.getvalue()
        assert f"fitting {learn.n_layers} layers" in output
        assert "layer 1: built" in output
        assert "layer 2: encoded" in output
        assert output.rstrip().endswith("done")

    def test_a_narrow_layer_says_what_it_was_asked_for(self, tmp_path: Path) -> None:
        from spifil import ArchSpec, LayerSpec

        stream = io.StringIO()
        arch = ArchSpec(
            layers=[
                LayerSpec(kernel_size=3, out_channels=7),
                LayerSpec(kernel_size=3, out_channels=16),
            ]
        )
        fitted(tmp_path, arch=arch, cbs=[ProgressBar(stream=stream)])

        assert "layer 1: built 6 filters (target 7)" in stream.getvalue()


class TestCallbackBase:
    @pytest.mark.parametrize("event", EVENTS)
    def test_the_base_defines_every_event_as_a_no_op(self, event: str) -> None:
        callback = Callback()

        assert getattr(callback, event)(None) is None  # type: ignore[arg-type]

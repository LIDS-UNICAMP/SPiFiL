"""Loading a bundle back, and composing it with an ordinary torch model.

Two questions, and the second is the one that shapes the API. *Does it load* is
:func:`~spifil.nn.model.load_model`: rebuild the stack from
``architecture.json``, install ``model.pt``, get the same numbers. *Is it
usable* is :class:`~spifil.nn.encoder.SpifilEncoder`: a fitted network is only
half an artifact, because it eats layer-0 features rather than images, and a
downstream model handed the wrong half fails silently — the shapes match and
only the numbers are wrong.

Fixture and components match ``test_learner.py``: the two-image
``minimal`` fixture with SLIC and centroid seeds, so a full fit is
milliseconds and no optional dependency is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import Tensor, nn

from spifil import (
    SLIC,
    ArchSpec,
    Centroids,
    DiversitySelector,
    FisherScorer,
    Identity,
    LabNorm,
    LayerSpec,
    Learner,
    Mahalanobis,
    SpifilDataset,
    SpifilEncoder,
)
from spifil.nn.encoder import describe_color, load_encoder, resolve_color
from spifil.nn.model import FORMAT_VERSION, arch_from_json, load_model
from tests.conftest import MINIMAL


def make_learner(arch: ArchSpec | None = None, **kwargs: object) -> Learner:
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
    return Learner(
        data,
        arch
        or ArchSpec(
            layers=[
                LayerSpec(kernel_size=3, out_channels=8),
                LayerSpec(kernel_size=3, out_channels=16),
            ]
        ),
        **defaults,  # type: ignore[arg-type]
    )


class NeedsArguments:
    """A ``ColorTransform`` with no default configuration, for the loader tests.

    Module level rather than nested in the test that uses it: the loader
    resolves a transform by importing its path, and a class defined inside a
    function has none.
    """

    def __init__(self, bands: int) -> None:
        self.out_channels = bands

    def __call__(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        return np.transpose(image, (2, 0, 1)).astype(np.float32)


@pytest.fixture(scope="module")
def fitted() -> tuple[Learner, Tensor]:
    """A fitted Learner and one layer-0 feature map to encode with it."""
    learn = make_learner()
    learn.fit()
    return learn, learn.state[0].features[0].unsqueeze(0)


@pytest.fixture(scope="module")
def bundle(
    fitted: tuple[Learner, Tensor], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """An exported bundle directory, written once for the whole module."""
    learn, _ = fitted
    return learn.export(tmp_path_factory.mktemp("bundle"))


class TestLoadModel:
    def test_a_loaded_model_encodes_identically(
        self, fitted: tuple[Learner, Tensor], bundle: Path
    ) -> None:
        """Bit-equality, not a tolerance: same weights, same operations."""
        learn, image = fitted

        loaded = load_model(bundle)

        with torch.no_grad():
            assert torch.equal(loaded(image), learn.model(image))

    def test_every_prefix_of_the_stack_matches(
        self, fitted: tuple[Learner, Tensor], bundle: Path
    ) -> None:
        """``upto`` means the same thing on both sides, layer 0 included."""
        learn, image = fitted
        loaded = load_model(bundle)

        for upto in range(len(learn.model) + 1):
            with torch.no_grad():
                assert torch.equal(loaded(image, upto=upto), learn.model(image, upto))

    def test_an_under_produced_layer_round_trips(self, tmp_path: Path) -> None:
        """The shrinking case: ``out_channels`` is what was built, not asked for.

        7 filters over 2 classes is 3 each, so layer 1
        holds 6 and layer 2 convolves 6 channels. Rebuilding from the *spec*
        would produce an 8-channel block and fail the strict load — which is
        the point of recording both numbers.
        """
        learn = make_learner(
            ArchSpec(
                layers=[
                    LayerSpec(kernel_size=3, out_channels=7),
                    LayerSpec(kernel_size=3, out_channels=16),
                ]
            )
        )
        learn.fit()
        out = learn.export(tmp_path)

        loaded = load_model(out)

        assert loaded.block(1).out_channels == 6
        assert loaded.block(2).in_channels == 6
        image = learn.state[0].features[0].unsqueeze(0)
        with torch.no_grad():
            assert torch.equal(loaded(image), learn.model(image))

    def test_an_unfitted_model_round_trips_too(self, tmp_path: Path) -> None:
        """Export is not gated on a fit; the shapes are right from the start."""
        learn = make_learner()
        out = learn.export(tmp_path)

        loaded = load_model(out)

        state_dict = learn.model.state_dict()
        assert set(loaded.state_dict()) == set(state_dict)
        for key, value in loaded.state_dict().items():
            assert torch.equal(value, state_dict[key])

    def test_the_loaded_model_is_an_ordinary_module(self, bundle: Path) -> None:
        """What the artifact is for: composing with a task head.

        Frozen filters, a trainable head, gradients that reach the head and
        stop at the encoder — the whole downstream story in one assertion.
        """
        net = load_model(bundle).requires_grad_(False)
        head = nn.Conv2d(net.block(len(net)).out_channels, 2, kernel_size=1)
        model = nn.Sequential(net, head)

        out = model(torch.rand(1, 3, 24, 32))
        out.sum().backward()

        assert head.weight.grad is not None
        assert all(param.grad is None for param in net.parameters())


class TestBundleValidation:
    def test_a_newer_format_is_refused(self) -> None:
        described = {"format": FORMAT_VERSION + 1, "in_channels": 3, "layers": []}

        with pytest.raises(ValueError, match="format version"):
            arch_from_json(described)

    def test_an_empty_stack_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no layers"):
            arch_from_json({"in_channels": 3, "layers": []})

    def test_a_missing_file_names_what_is_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="architecture.json"):
            load_model(tmp_path)

    def test_weights_that_do_not_fit_the_description_fail_loudly(
        self, bundle: Path, tmp_path: Path
    ) -> None:
        """Strict loading is what makes the description a check, not a label."""
        (tmp_path / "model.pt").write_bytes((bundle / "model.pt").read_bytes())
        described = json.loads((bundle / "architecture.json").read_text())
        described["layers"][0]["out_channels"] += 1
        (tmp_path / "architecture.json").write_text(json.dumps(described))

        with pytest.raises(RuntimeError, match="size mismatch"):
            load_model(tmp_path)


class TestColorProvenance:
    """A bundle records which transform its features came from, and why."""

    def test_the_bundle_names_its_color_transform(self, bundle: Path) -> None:
        described = json.loads((bundle / "architecture.json").read_text())

        assert described["color"] == {
            "class": "spifil.color.LabNorm",
            "out_channels": 3,
        }

    def test_a_recorded_transform_is_rebuilt(self, bundle: Path) -> None:
        encoder = load_encoder(bundle)

        assert isinstance(encoder.color, LabNorm)

    def test_a_transform_of_the_wrong_width_is_refused(self, bundle: Path) -> None:
        """The check that a shape comparison cannot make.

        Lab and RGB both produce three bands, so nothing downstream would
        notice the swap. A band-count mismatch is the unambiguous half of that
        problem, and refusing it is free.
        """
        with pytest.raises(ValueError, match="fitted on 3-band"):
            load_encoder(bundle, color=Identity(out_channels=5))

    def test_an_explicit_transform_wins(self, bundle: Path) -> None:
        """The escape hatch for transforms the bundle cannot reconstruct."""
        override = Identity(out_channels=3)

        encoder = load_encoder(bundle, color=override)

        assert encoder.color is override

    def test_an_older_bundle_says_what_it_needs(self) -> None:
        with pytest.raises(ValueError, match="does not record its colour"):
            resolve_color(None)

    def test_a_parameterized_transform_asks_to_be_passed_in(self) -> None:
        """A constructor default is the dangerous case, not the loud one.

        ``Identity(out_channels=7)`` rebuilds as ``Identity()`` without
        complaint — a 3-band transform standing in for a 7-band one, wrong in
        exactly the silent way this whole field exists to prevent. Checking the
        rebuilt width against the recorded one is what catches it.
        """
        described = describe_color(Identity(out_channels=7))

        with pytest.raises(ValueError, match="produces 3-band"):
            resolve_color(described)

    def test_a_transform_needing_arguments_says_so(self) -> None:
        """The loud case: no default to construct from at all."""
        described = describe_color(NeedsArguments(bands=3))

        with pytest.raises(TypeError, match="needs constructor arguments"):
            resolve_color(described)

    def test_an_unimportable_transform_says_so(self) -> None:
        described = {"class": "not_a_module.Nope", "out_channels": 3}

        with pytest.raises(ImportError, match="cannot import"):
            resolve_color(described)


class TestEncoder:
    def test_images_go_in_and_features_come_out(
        self, fitted: tuple[Learner, Tensor], bundle: Path
    ) -> None:
        """The gap the encoder closes: raw pixels, not layer-0 features.

        Checked against the Learner's own transform + network, so a colour
        step applied differently here would show up as different numbers.
        """
        learn, _ = fitted
        image = learn.data.load_image(0)
        encoder = load_encoder(bundle)

        with torch.no_grad():
            encoded = encoder(image[None])
            expected = learn.model(
                torch.from_numpy(learn.color(image)).unsqueeze(0),
            )

        assert torch.equal(encoded, expected)

    def test_features_can_be_passed_straight_through(
        self, fitted: tuple[Learner, Tensor], bundle: Path
    ) -> None:
        """A float tensor is already transformed — for pipelines that preprocess."""
        learn, image = fitted
        encoder = load_encoder(bundle)

        with torch.no_grad():
            assert torch.equal(encoder(image), learn.model(image))

    def test_upto_reaches_the_intermediate_layers(self, bundle: Path) -> None:
        encoder = load_encoder(bundle)
        image = np.zeros((16, 20, 3), dtype=np.uint8)

        with torch.no_grad():
            transformed = encoder(image[None], upto=0)
            first = encoder(image[None], upto=1)

        assert transformed.shape[1] == encoder.in_channels
        assert first.shape[1] == encoder.net.block(1).out_channels

    def test_freezing_leaves_a_trainable_head(self, bundle: Path) -> None:
        encoder = load_encoder(bundle).freeze()
        head = nn.Conv2d(encoder.out_channels, 2, kernel_size=1)

        out = head(encoder(np.zeros((1, 16, 20, 3), dtype=np.uint8)))
        out.sum().backward()

        assert head.weight.grad is not None
        assert all(param.grad is None for param in encoder.parameters())

    def test_a_torch_transform_runs_in_the_graph(self, bundle: Path) -> None:
        """Nothing here changes when the colour transform becomes torch-native.

        The numpy branch is a boundary convenience, not the design: an
        ``nn.Module`` transform is used directly on the batch, which is what a
        GPU-side or learnable transform would be.
        """

        class Scaled(nn.Module):
            out_channels = 3

            def forward(self, images: Tensor) -> Tensor:
                return images.permute(0, 3, 1, 2).float() / 255.0

        net = load_model(bundle)
        encoder = SpifilEncoder(Scaled(), net)  # type: ignore[arg-type]
        images = torch.randint(0, 255, (1, 16, 20, 3), dtype=torch.uint8)

        with torch.no_grad():
            encoded = encoder(images)
            expected = net(images.permute(0, 3, 1, 2).float() / 255.0)

        assert torch.equal(encoded, expected)


class TestIdentityTransform:
    """The 'my data is already features' case — RGB, other spaces, sensors."""

    def test_channels_move_to_the_front(self) -> None:
        image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)

        features = Identity(out_channels=3)(image)

        assert features.shape == (3, 2, 4)
        assert features.dtype == np.float32
        assert np.array_equal(features, np.transpose(image, (2, 0, 1)))

    def test_a_single_band_image_is_accepted(self) -> None:
        features = Identity(out_channels=1)(np.zeros((3, 5), dtype=np.uint8))

        assert features.shape == (1, 3, 5)

    def test_the_wrong_band_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expects"):
            Identity(out_channels=3)(np.zeros((3, 5, 4), dtype=np.uint8))

    def test_it_fits_a_whole_pipeline(self, tmp_path: Path) -> None:
        """The proof it is a real ``ColorTransform``: fit, export, reload.

        Values differ from a LabNorm fit and are not meant to match anything —
        what is being checked is that an alternative transform travels
        through the Learner, the bundle and back without a special case.
        """
        learn = make_learner(color=Identity(out_channels=3))
        learn.fit()
        out = learn.export(tmp_path)

        encoder = load_encoder(out)

        assert isinstance(encoder.color, Identity)
        image = learn.data.load_image(0)
        with torch.no_grad():
            assert torch.equal(
                encoder(image[None]),
                learn.model(torch.from_numpy(learn.color(image)).unsqueeze(0)),
            )

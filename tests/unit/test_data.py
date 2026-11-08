"""Dataset discovery: class parsing, mask pairing, ordering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spifil.data import SpifilDataset, parse_class_label


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("000001_00000012", 1),
        ("000002_00000094", 2),
        ("3_sample", 3),
        ("12", 12),
        ("-4_neg", -4),
        ("cyst_03", 0),  # atoi returns 0 when there is no leading number
        ("", 0),
    ],
)
def test_parse_class_label_matches_atoi(stem: str, expected: int) -> None:
    assert parse_class_label(stem) == expected


def test_from_folders_pairs_masks_and_labels(multiclass) -> None:
    data = SpifilDataset.from_folders(multiclass.images, multiclass.masks)

    assert len(data) == 18
    assert data.classes == [1, 2, 3, 4, 5, 6]
    assert all(s.mask_path is not None for s in data)
    assert all(s.label == parse_class_label(s.name) for s in data)


def test_from_folders_is_sorted(twoclass) -> None:
    """Discovery order does not depend on how the filesystem lists a dir."""
    data = SpifilDataset.from_folders(twoclass.images, twoclass.masks)
    names = [s.name for s in data]

    assert names == sorted(names)


def test_missing_mask_warns_and_falls_back(tmp_path: Path, twoclass) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "000001_a.png").write_bytes(next(twoclass.images.iterdir()).read_bytes())

    with pytest.warns(UserWarning, match="No mask found"):
        data = SpifilDataset.from_folders(images, tmp_path / "empty_masks")

    assert data[0].mask_path is None
    assert data.load_mask(0) is None


def test_mask_extension_fallback(tmp_path: Path, twoclass) -> None:
    """A .jpg image pairs with a .png mask, via the stem fallback."""
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    source = next(twoclass.images.iterdir())
    (images / "000007_x.jpg").write_bytes(source.read_bytes())
    (masks / "000007_x.png").write_bytes(source.read_bytes())

    data = SpifilDataset.from_folders(images, masks)

    assert data[0].label == 7
    assert data[0].mask_path == masks / "000007_x.png"


def test_non_image_files_and_dotfiles_are_ignored(tmp_path: Path, twoclass) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "000001_a.png").write_bytes(next(twoclass.images.iterdir()).read_bytes())
    (images / "notes.txt").write_text("ignore me")
    (images / ".hidden.png").write_text("ignore me too")

    data = SpifilDataset.from_folders(images, None)

    assert [s.name for s in data] == ["000001_a"]


def test_load_image_is_rgb_uint8(twoclass) -> None:
    data = SpifilDataset.from_folders(twoclass.images, twoclass.masks)
    image = data.load_image(0)

    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype.name == "uint8"


def test_missing_images_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        SpifilDataset.from_folders(tmp_path / "nope")


def _write_descriptor(tmp_path: Path, splits: dict[str, list[str]]) -> Path:
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(splits))
    return path


def test_json_descriptor_keeps_only_the_subset(tmp_path: Path, twoclass) -> None:
    names = sorted(p.name for p in twoclass.images.iterdir())
    descriptor = _write_descriptor(tmp_path, {"train": names[:2], "test": names[2:]})

    data = SpifilDataset.from_folders(
        twoclass.images, twoclass.masks, json_descriptor=descriptor, subset="train"
    )

    assert [s.image_path.name for s in data] == names[:2]


@pytest.mark.parametrize("given", ["json_descriptor", "subset"])
def test_json_descriptor_and_subset_must_come_together(
    tmp_path: Path, twoclass, given: str
) -> None:
    descriptor = _write_descriptor(tmp_path, {"train": []})
    kwargs = {"json_descriptor": descriptor, "subset": "train"}

    with pytest.raises(ValueError, match="must be given together"):
        SpifilDataset.from_folders(twoclass.images, **{given: kwargs[given]})


def test_unknown_subset_lists_the_available_ones(tmp_path: Path, twoclass) -> None:
    descriptor = _write_descriptor(tmp_path, {"train": [], "test": []})

    with pytest.raises(ValueError, match=r"available: \['test', 'train'\]"):
        SpifilDataset.from_folders(
            twoclass.images, json_descriptor=descriptor, subset="val"
        )


def test_json_descriptor_warns_on_missing_files(tmp_path: Path, twoclass) -> None:
    present = sorted(p.name for p in twoclass.images.iterdir())[0]
    descriptor = _write_descriptor(tmp_path, {"train": [present, "ghost.png"]})

    with pytest.warns(UserWarning, match="ghost.png"):
        data = SpifilDataset.from_folders(
            twoclass.images, twoclass.masks, json_descriptor=descriptor, subset="train"
        )

    assert [s.image_path.name for s in data] == [present]

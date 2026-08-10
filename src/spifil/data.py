"""Dataset discovery: image/mask pairing and class-from-filename.

List the images in a folder, take each one's class from the integer prefix of
its name, and pair it with a mask of the same name::

    images/000001_00000012.png   ->  class 1
    masks/000001_00000012.png        the region superpixels are drawn inside

No annotation beyond the masks the dataset already ships with is needed.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from skimage.io import imread

__all__ = ["IMAGE_EXTENSIONS", "Sample", "SpifilDataset", "parse_class_label"]

IMAGE_EXTENSIONS = (".png", ".ppm", ".pgm", ".jpg", ".tif")
"""Image extensions discovered by default, matched case-insensitively."""


def parse_class_label(name: str) -> int:
    """Class label from the integer prefix of a filename stem.

    Reads an optional sign followed by digits and stops at the first character
    that is not one, giving 0 when there are none — so ``"000001_00000012"``
    is class 1, and ``"cyst_03"`` is class 0 because it does not *start* with
    its number.

    Examples
    --------
    >>> parse_class_label("000002_00000094")
    2
    """
    match = re.match(r"[+-]?[0-9]+", name)
    return int(match.group()) if match else 0


@dataclass(frozen=True)
class Sample:
    """One image, its class, and the mask restricting superpixel extraction."""

    name: str
    """Filename stem — the key every output artifact is named after."""
    image_path: Path
    mask_path: Path | None
    label: int


class SpifilDataset(Sequence[Sample]):
    """An ordered collection of :class:`Sample` records.

    Indexing yields metadata; pixels are read on demand via :meth:`load_image`
    and :meth:`load_mask` so that a dataset stays cheap to construct and hold.
    """

    def __init__(self, samples: Sequence[Sample]) -> None:
        self._samples = list(samples)

    @classmethod
    def from_folders(
        cls,
        images_dir: str | Path,
        masks_dir: str | Path | None = None,
        *,
        extensions: Sequence[str] = IMAGE_EXTENSIONS,
    ) -> SpifilDataset:
        """Discover samples in ``images_dir``, pairing masks from ``masks_dir``.

        A mask is looked up first under the image's full filename, then under
        ``<stem>.png``, so a ``.jpg`` image pairs with a ``.png`` mask. Images
        with no mask are kept with ``mask_path=None``, meaning "use the whole
        image", and warn.

        Parameters
        ----------
        images_dir
            Folder of images named ``<class_id>_<name>.<ext>``.
        masks_dir
            Folder of binary masks. ``None`` skips masking entirely.
        extensions
            Accepted image extensions, matched case-insensitively.

        Notes
        -----
        Samples are returned in sorted filename order, so a run does not
        depend on the order the filesystem happens to list a directory in.
        """
        images_dir = Path(images_dir)
        if not images_dir.is_dir():
            raise NotADirectoryError(f"images_dir does not exist: {images_dir}")
        masks_dir = Path(masks_dir) if masks_dir is not None else None

        suffixes = {e.lower() for e in extensions}
        paths = sorted(
            p
            for p in images_dir.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in suffixes
        )

        samples = []
        for path in paths:
            samples.append(
                Sample(
                    name=path.stem,
                    image_path=path,
                    mask_path=_find_mask(masks_dir, path),
                    label=parse_class_label(path.stem),
                )
            )
        return cls(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Sample:  # type: ignore[override]
        return self._samples[index]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    @property
    def labels(self) -> list[int]:
        """Class label of every sample, in dataset order."""
        return [s.label for s in self._samples]

    @property
    def classes(self) -> list[int]:
        """The distinct class ids, sorted ascending."""
        return sorted(set(self.labels))

    def load_image(self, index: int) -> npt.NDArray[np.uint8]:
        """Read sample ``index`` as an ``(H, W, 3)`` uint8 RGB array."""
        image = imread(self[index].image_path)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        return np.ascontiguousarray(image)

    def load_mask(self, index: int) -> npt.NDArray[np.int32] | None:
        """Read sample ``index``'s mask as ``(H, W)`` int32, or ``None``.

        Values are passed through as stored --- any nonzero value counts as
        inside the region --- collapsing a color mask to its first channel.
        """
        path = self[index].mask_path
        if path is None:
            return None
        mask = imread(path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return np.ascontiguousarray(mask.astype(np.int32))


def _find_mask(masks_dir: Path | None, image_path: Path) -> Path | None:
    if masks_dir is None:
        return None
    for candidate in (
        masks_dir / image_path.name,
        masks_dir / f"{image_path.stem}.png",
    ):
        if candidate.is_file():
            return candidate
    warnings.warn(
        f"No mask found for {image_path.stem} in {masks_dir}; using the full image.",
        stacklevel=3,
    )
    return None

"""Coordinate flattening, unflattening and projection onto a pooled grid."""

from __future__ import annotations

import pytest
import torch

from spifil.coords import from_linear, project_to_grid, to_linear


@pytest.mark.parametrize("grid", [(100, 100), (7, 13), (5, 7, 13), (3, 5, 7, 13)])
def test_linear_round_trip(grid: tuple[int, ...]) -> None:
    """2D, 3D and 4D grids all round-trip through linear indices."""
    n = int(torch.tensor(grid).prod())
    indices = torch.arange(n)
    coords = from_linear(indices, grid)
    assert coords.shape == (n, len(grid))
    torch.testing.assert_close(to_linear(coords, grid), indices)


def test_linear_matches_c_convention() -> None:
    """C computes ``x = elem % xsize``, ``y = elem // xsize``; coords are (y, x)."""
    xsize, ysize = 100, 80
    elem = torch.tensor([0, 1, 99, 100, 8000 - 1])
    coords = from_linear(elem, (ysize, xsize))
    expected_y, expected_x = elem // xsize, elem % xsize
    torch.testing.assert_close(coords[:, 0], expected_y)
    torch.testing.assert_close(coords[:, 1], expected_x)


def test_project_to_grid_halves_coordinates() -> None:
    coords = torch.tensor([[0, 0], [1, 1], [2, 3], [99, 98]])
    projected, grid = project_to_grid(coords, stride=2, grid=(100, 100))
    torch.testing.assert_close(
        projected, torch.tensor([[0, 0], [0, 0], [1, 1], [49, 49]])
    )
    assert grid == (50, 50)


def test_project_to_grid_ceils_odd_sizes() -> None:
    """A pooled grid is sized with ceil(size / stride)."""
    _, grid = project_to_grid(torch.tensor([[0, 0]]), stride=2, grid=(99, 101))
    assert grid == (50, 51)


def test_project_to_grid_stride_one_is_identity() -> None:
    coords = torch.tensor([[3, 4], [5, 6]])
    projected, grid = project_to_grid(coords, stride=1, grid=(10, 10))
    torch.testing.assert_close(projected, coords)
    assert grid == (10, 10)


def test_rejects_coords_of_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"coords must be \(n, 2\)"):
        to_linear(torch.tensor([[1, 2, 3]]), (10, 10))

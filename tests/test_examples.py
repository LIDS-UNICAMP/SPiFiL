"""Execute the example notebook, so it cannot rot quietly.

An example that no longer runs is worse than no example: it is documentation
that lies, and nothing about the package's own test suite would notice. So the
notebook is executed here top to bottom, and any cell that raises fails the
build.

It runs on the datasets bundled in ``examples/data``, so it passes on a fresh
clone with nothing downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOK = EXAMPLES / "quickstart.ipynb"


def read_notebook(nbformat: object) -> object:
    """Parse the notebook at its current format version."""
    return nbformat.read(str(NOTEBOOK), as_version=4)  # type: ignore[attr-defined]


def test_the_quickstart_notebook_runs() -> None:
    """Every cell, in order."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")  # no display in CI; figures are still built

    notebook = read_notebook(nbformat)
    client = nbclient.NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        # Run from examples/, which is where a reader opens the notebook.
        resources={"metadata": {"path": str(EXAMPLES)}},
    )

    client.execute()


def test_the_notebook_is_committed_without_outputs() -> None:
    """Keeps the diffs readable and the outputs honest.

    A notebook carrying stored outputs shows numbers from whenever it was last
    run by hand, which drift from what the code now produces — and the images
    make every edit a binary diff. The execution test above is what proves it
    still works, so the committed copy does not need to carry the evidence.
    """
    nbformat = pytest.importorskip("nbformat")
    notebook = read_notebook(nbformat)

    for number, cell in enumerate(notebook.cells, start=1):  # type: ignore[attr-defined]
        if cell.cell_type != "code":
            continue
        assert not cell.outputs, f"cell {number} has stored outputs; clear them"
        assert cell.execution_count is None, f"cell {number} has an execution count"

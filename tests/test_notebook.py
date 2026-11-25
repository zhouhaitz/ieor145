"""Run the teammate notebook top to bottom from the repo root and check it raises no errors."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "generate_synthetic_data.ipynb"

pytestmark = pytest.mark.slow


def test_notebook_runs_without_errors():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    NotebookClient(notebook, kernel_name="python3", resources={"metadata": {"path": str(REPO_ROOT)}}).execute()
    printed_text = "".join(
        output.get("text", "") for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    )
    assert "customers_per_day" in printed_text


def test_notebook_is_saved_with_cleared_outputs():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(not cell.outputs and cell.execution_count is None for cell in code_cells)

"""Run the teammate notebooks top to bottom from the repo root and check they raise no errors."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "generate_synthetic_data.ipynb"
POLICY_TEMPLATE_PATH = REPO_ROOT / "notebooks" / "policy_template.ipynb"

pytestmark = pytest.mark.slow


def test_notebook_runs_without_errors():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    NotebookClient(notebook, kernel_name="python3", resources={"metadata": {"path": str(REPO_ROOT)}}).execute()
    printed_text = "".join(
        output.get("text", "") for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    )
    assert "customers_per_day" in printed_text


def test_policy_template_runs_and_compares_all_policies():
    notebook = nbformat.read(POLICY_TEMPLATE_PATH, as_version=4)
    NotebookClient(notebook, kernel_name="python3", resources={"metadata": {"path": str(REPO_ROOT)}}).execute()
    displayed_text = "".join(
        str(output.get("data", {}).get("text/plain", ""))
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
    assert "my_policy" in displayed_text and "myopic_mnl" in displayed_text


@pytest.mark.parametrize("notebook_path", [NOTEBOOK_PATH, POLICY_TEMPLATE_PATH])
def test_notebook_is_saved_with_cleared_outputs(notebook_path):
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(not cell.outputs and cell.execution_count is None for cell in code_cells)

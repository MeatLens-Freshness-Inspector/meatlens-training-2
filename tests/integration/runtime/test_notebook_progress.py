from __future__ import annotations

import json
from pathlib import Path

from meatlens_pork_pipeline import notebook_progress
from tests.support.notebook_test_utils import execute_notebook


NOTEBOOK_PATHS = [
    Path("00_shared_setup.ipynb"),
    Path("01_manifest_and_dataset_audit.ipynb"),
    Path("02_roi_preprocessing.ipynb"),
    Path("03_build_cross_rotation_splits.ipynb"),
    Path("04_train_8fold_mobilenetv3small.ipynb"),
    Path("05_regenerate_metrics_and_reports.ipynb"),
    Path("06_train_final_deployment_model.ipynb"),
    Path("07_export_onnx.ipynb"),
    Path("08_inference_smoke_test.ipynb"),
]


def test_shared_setup_exposes_notebook_progress_helpers() -> None:
    namespace = execute_notebook(
        Path("00_shared_setup.ipynb"),
        overrides={"NOTEBOOK_TEST_MODE": True},
        cwd=Path.cwd(),
    )

    assert callable(namespace["start_notebook_cell_progress"])
    assert callable(namespace["advance_notebook_cell_progress"])
    assert callable(namespace["finish_notebook_cell_progress"])
    assert callable(namespace["iter_notebook_progress"])

    progress = namespace["start_notebook_cell_progress"]("test-notebook", "test-cell", total_steps=2)
    namespace["advance_notebook_cell_progress"](progress, "halfway")
    namespace["finish_notebook_cell_progress"](progress, "done")

    assert list(namespace["iter_notebook_progress"]([1, 2], "items", total=2)) == [1, 2]


def test_plain_progress_emits_time_heartbeat_without_checkpoint(capsys) -> None:
    current_time = [0.0]
    progress = notebook_progress._PlainProgressBar(
        total=100,
        description="images",
        unit="image",
        emit_interval_seconds=5.0,
        clock=lambda: current_time[0],
    )
    capsys.readouterr()

    progress.update()
    assert capsys.readouterr().out == ""

    current_time[0] = 5.0
    progress.update()
    output = capsys.readouterr().out
    assert "[RUNNING] images [2/100 image]" in output

    progress.update()
    assert capsys.readouterr().out == ""


def test_every_notebook_code_cell_starts_cell_progress() -> None:
    for notebook_path in NOTEBOOK_PATHS:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]

        assert code_cells, f"{notebook_path} should contain code cells"

        for cell in code_cells:
            source = "".join(cell.get("source", []))
            assert "start_notebook_cell_progress(" in source, (
                f"{notebook_path} is missing a visible cell-level progress start hook"
            )
            assert "finish_notebook_cell_progress(" in source, (
                f"{notebook_path} is missing a visible cell-level progress finish hook"
            )

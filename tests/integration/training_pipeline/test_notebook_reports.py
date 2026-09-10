from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tests.support.notebook_test_utils import execute_notebook


def test_report_notebook_exposes_progress_phase_markers() -> None:
    notebook = json.loads(Path("05_regenerate_metrics_and_reports.ipynb").read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "[REPORT] Loading prediction CSVs" in code_source
    assert "[REPORT] Summarizing folds" in code_source
    assert "[REPORT] Writing report artifacts" in code_source


def test_regenerate_metrics_and_reports_writes_summary_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "o"
    predictions_dir = output_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    prediction_path_1 = predictions_dir / "processed_roi8_cnn_only_fold1_seed42_test_predictions.csv"
    prediction_path_2 = predictions_dir / "processed_roi8_cnn_only_fold2_seed42_test_predictions.csv"

    pd.DataFrame(
        [
            {"true_label": "fresh", "predicted_label": "fresh"},
            {"true_label": "not fresh", "predicted_label": "not fresh"},
            {"true_label": "spoiled", "predicted_label": "spoiled"},
        ]
    ).to_csv(prediction_path_1, index=False)
    pd.DataFrame(
        [
            {"true_label": "fresh", "predicted_label": "not fresh"},
            {"true_label": "not fresh", "predicted_label": "not fresh"},
            {"true_label": "spoiled", "predicted_label": "spoiled"},
        ]
    ).to_csv(prediction_path_2, index=False)

    seed_metrics_path = output_root / "processed_roi8_cnn_only_seed_metrics.csv"
    pd.DataFrame(
        [
            {"fold": "fold1", "seed": 42, "predictions_path": str(prediction_path_1)},
            {"fold": "fold2", "seed": 42, "predictions_path": str(prediction_path_2)},
        ]
    ).to_csv(seed_metrics_path, index=False)

    execute_notebook(
        Path("05_regenerate_metrics_and_reports.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "EIGHTFOLD_OUTPUT_ROOT": str(output_root),
            "SEED_METRICS_PATH": str(seed_metrics_path),
        },
        cwd=Path.cwd(),
    )

    assert (output_root / "processed_roi8_cnn_only_fold_summary.csv").exists()
    assert (output_root / "processed_roi8_cnn_only_prediction_distribution.csv").exists()
    assert (output_root / "processed_roi8_cnn_only_overall_confusion_matrix.csv").exists()
    assert (output_root / "processed_roi8_cnn_only_overall_confusion_matrix.png").exists()


def test_regenerate_reports_writes_procedure_summary_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "o"
    predictions_root = output_root / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    prediction_path = predictions_root / "processed_roi8_cnn_only_fold1_seed42_test_predictions.csv"
    pd.DataFrame(
        [
            {"true_label": "fresh", "predicted_label": "spoiled"},
            {"true_label": "not fresh", "predicted_label": "not fresh"},
            {"true_label": "spoiled", "predicted_label": "fresh"},
        ]
    ).to_csv(prediction_path, index=False)

    metrics_path = output_root / "processed_roi8_cnn_only_seed_metrics.csv"
    pd.DataFrame(
        [
            {
                "fold": "fold1",
                "seed": 42,
                "predictions_path": str(prediction_path),
                "input_mode": "processed_hsv_lab_threshold_roi_224",
                "fine_tune_fraction": 0.25,
            }
        ]
    ).to_csv(metrics_path, index=False)

    execute_notebook(
        Path("05_regenerate_metrics_and_reports.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "EIGHTFOLD_OUTPUT_ROOT": str(output_root),
            "SEED_METRICS_PATH": str(metrics_path),
        },
        cwd=Path.cwd(),
    )

    assert (output_root / "processed_roi8_cnn_only_worst_fold_summary.csv").exists()
    assert (output_root / "processed_roi8_cnn_only_severe_error_summary.csv").exists()
    assert (output_root / "processed_roi8_cnn_only_procedure_summary.csv").exists()

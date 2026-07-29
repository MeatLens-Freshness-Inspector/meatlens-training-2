from __future__ import annotations

from pathlib import Path


def test_training_pipeline_usage_is_notebook_first() -> None:
    content = Path("docs/training-pipeline-usage.md").read_text(encoding="utf-8")

    assert "00_shared_setup.ipynb" in content
    assert "08_inference_smoke_test.ipynb" in content
    assert "Official evaluation metrics come from the 8-fold cross-rotation workflow." in content
    assert "RTX 4050" in content
    assert "environment.windows-gpu.yml" in content
    assert "meatlens-tf210-gpu" in content
    assert "numpy<2" in content
    assert "data/processed_hsv_lab_threshold_roi_224/processing_summary.csv" in content


def test_training_pipeline_usage_describes_procedure_improvements() -> None:
    content = Path("docs/training-pipeline-usage.md").read_text(encoding="utf-8")

    assert "raw_center_crop_224" in content
    assert "severe-error rate" in content
    assert "worst-fold macro-F1" in content
    assert "sample-aware validation" in content

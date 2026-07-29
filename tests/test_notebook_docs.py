from __future__ import annotations

from pathlib import Path


def test_training_pipeline_usage_is_notebook_first() -> None:
    content = Path("docs/training-pipeline-usage.md").read_text(encoding="utf-8")

    assert "00_shared_setup.ipynb" in content
    assert "08_inference_smoke_test.ipynb" in content
    assert "Official evaluation metrics come from the 8-fold cross-rotation workflow." in content
    assert "RTX 4050" in content

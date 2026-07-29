from pathlib import Path

from meatlens_pork_pipeline.config import (
    IMAGE_CROP_MODE,
    INPUT_SIZE,
    LABEL_ORDER,
    MODEL_INPUT_MODE,
    build_run_paths,
)


def test_build_run_paths_uses_required_output_contract(tmp_path: Path) -> None:
    paths = build_run_paths(
        dataset_slug="pilot",
        seed=42,
        timestamp="20260729_130500",
        root=tmp_path,
    )

    assert LABEL_ORDER == ("fresh", "not fresh", "spoiled")
    assert INPUT_SIZE == (224, 224)
    assert MODEL_INPUT_MODE == "cnn_only"
    assert IMAGE_CROP_MODE == "preprocessed_hsv_lab_threshold_roi_224"
    assert paths.processed_dataset_root == tmp_path / "processed_datasets" / "pilot"
    assert paths.run_root == tmp_path / "runs" / "20260729_130500_seed42"
    assert paths.models_dir == paths.run_root / "models"
    assert paths.metrics_dir == paths.run_root / "metrics"

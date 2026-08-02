from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from tests.support.notebook_test_utils import execute_notebook


def test_manifest_audit_rebuilds_local_paths_from_processing_summary(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed_hsv_lab_threshold_roi_224"
    fresh_dir = processed_root / "sample 1" / "fresh"
    spoiled_dir = processed_root / "sample 2" / "spoiled"
    fresh_dir.mkdir(parents=True, exist_ok=True)
    spoiled_dir.mkdir(parents=True, exist_ok=True)

    fresh_image = fresh_dir / "fresh_001.jpg"
    spoiled_image = spoiled_dir / "spoiled_001.png"
    Image.new("RGB", (320, 240), color=(180, 80, 80)).save(fresh_image)
    Image.new("RGB", (240, 320), color=(80, 80, 180)).save(spoiled_image)

    manifest_path = processed_root / "processing_summary.csv"
    pd.DataFrame(
        [
            {
                "image_file_name": fresh_image.name,
                "sample_number": "1",
                "label": " Fresh ",
                "sample_id": "sample_1",
                "processed_output_file": r"E:\old\sample 1\fresh\fresh_001.jpg",
            },
            {
                "image_file_name": spoiled_image.name,
                "sample_number": "2",
                "label": "SPOILED",
                "sample_id": "sample_2",
                "processed_output_file": r"E:\old\sample 2\spoiled\spoiled_001.png",
            },
        ]
    ).to_csv(manifest_path, index=False)

    audited_manifest_path = tmp_path / "generated_splits" / "audited_manifest.csv"

    execute_notebook(
        Path("01_manifest_and_dataset_audit.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "MANIFEST_PATH": str(manifest_path),
            "PROCESSED_ROI_ROOT": str(processed_root),
            "AUDITED_MANIFEST_PATH": str(audited_manifest_path),
            "GENERATED_SPLITS_ROOT": str(tmp_path / "generated_splits"),
        },
        cwd=Path.cwd(),
    )

    audited_df = pd.read_csv(audited_manifest_path)
    assert audited_df["label"].tolist() == ["fresh", "spoiled"]
    assert audited_df["source_manifest_type"].tolist() == ["processed_summary", "processed_summary"]
    assert audited_df["sample_id"].tolist() == ["sample_1", "sample_2"]
    assert audited_df["local_image_path"].map(Path).map(Path.exists).all()


def test_roi_preprocessing_generates_processed_dataset_from_raw_manifest(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_images"
    raw_root.mkdir(parents=True, exist_ok=True)
    source_image = raw_root / "raw_001.jpg"
    Image.new("RGB", (480, 360), color=(190, 110, 110)).save(source_image)

    audited_manifest_path = tmp_path / "generated_splits" / "audited_manifest.csv"
    audited_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "image_name": source_image.name,
                "image_file_name": source_image.name,
                "label": "fresh",
                "sample_number": "1",
                "sample_id": "sample_1",
                "local_image_path": str(source_image),
                "source_manifest_type": "raw_manifest",
            }
        ]
    ).to_csv(audited_manifest_path, index=False)

    processed_root = tmp_path / "data" / "processed_hsv_lab_threshold_roi_224"
    processed_manifest_path = tmp_path / "generated_splits" / "processed_manifest.csv"
    preprocessing_summary_path = processed_root / "preprocessing_summary.csv"
    preprocessing_failures_path = processed_root / "preprocessing_failures.csv"

    execute_notebook(
        Path("02_roi_preprocessing.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "FORCE_REPROCESS": True,
            "AUDITED_MANIFEST_PATH": str(audited_manifest_path),
            "PROCESSED_ROI_ROOT": str(processed_root),
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "PREPROCESSING_SUMMARY_PATH": str(preprocessing_summary_path),
            "PREPROCESSING_FAILURES_PATH": str(preprocessing_failures_path),
            "GENERATED_SPLITS_ROOT": str(tmp_path / "generated_splits"),
        },
        cwd=Path.cwd(),
    )

    processed_df = pd.read_csv(processed_manifest_path)
    assert processed_df["label"].tolist() == ["fresh"]
    assert processed_df["local_image_path"].map(Path).map(Path.exists).all()
    output_image = Path(processed_df.loc[0, "local_image_path"])
    assert Image.open(output_image).size == (224, 224)
    assert preprocessing_summary_path.exists()
    assert preprocessing_failures_path.exists()


def test_roi_preprocessing_can_generate_raw_center_crop_branch(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_images"
    raw_root.mkdir(parents=True, exist_ok=True)
    source_image = raw_root / "raw_001.jpg"
    Image.new("RGB", (480, 360), color=(190, 110, 110)).save(source_image)

    audited_manifest_path = tmp_path / "generated_splits" / "audited_manifest.csv"
    audited_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "image_name": source_image.name,
                "image_file_name": source_image.name,
                "label": "fresh",
                "sample_number": "1",
                "sample_id": "sample_1",
                "local_image_path": str(source_image),
                "source_manifest_type": "raw_manifest",
            }
        ]
    ).to_csv(audited_manifest_path, index=False)

    raw_center_crop_root = tmp_path / "data" / "raw_center_crop_224"
    processed_manifest_path = tmp_path / "generated_splits" / "processed_manifest.csv"
    preprocessing_summary_path = raw_center_crop_root / "preprocessing_summary.csv"
    preprocessing_failures_path = raw_center_crop_root / "preprocessing_failures.csv"

    execute_notebook(
        Path("02_roi_preprocessing.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "FORCE_REPROCESS": True,
            "INPUT_MODE": "raw_center_crop_224",
            "AUDITED_MANIFEST_PATH": str(audited_manifest_path),
            "RAW_CENTER_CROP_ROOT": str(raw_center_crop_root),
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "PREPROCESSING_SUMMARY_PATH": str(preprocessing_summary_path),
            "PREPROCESSING_FAILURES_PATH": str(preprocessing_failures_path),
            "GENERATED_SPLITS_ROOT": str(tmp_path / "generated_splits"),
        },
        cwd=Path.cwd(),
    )

    processed_df = pd.read_csv(processed_manifest_path)
    output_image = Path(processed_df.loc[0, "local_image_path"])

    assert processed_df.loc[0, "input_mode"] == "raw_center_crop_224"
    assert output_image.exists()
    assert Image.open(output_image).size == (224, 224)

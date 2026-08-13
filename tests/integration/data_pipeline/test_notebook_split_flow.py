from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.support.notebook_test_utils import execute_notebook


def test_build_cross_rotation_splits_creates_official_fold_files(tmp_path: Path) -> None:
    processed_manifest_path = tmp_path / "generated_splits" / "processed_manifest.csv"
    processed_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for sample_number in range(1, 9):
        for label in ("fresh", "not fresh", "spoiled"):
            rows.append(
                {
                    "image_file_name": f"sample_{sample_number}_{label.replace(' ', '_')}.jpg",
                    "label": label,
                    "sample_number": str(sample_number),
                    "sample_id": f"sample_{sample_number}",
                    "local_image_path": str(tmp_path / f"sample_{sample_number}_{label.replace(' ', '_')}.jpg"),
                }
            )

    pd.DataFrame(rows).to_csv(processed_manifest_path, index=False)

    generated_splits_root = tmp_path / "generated_splits"
    execute_notebook(
        Path("03_build_cross_rotation_splits.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "GENERATED_SPLITS_ROOT": str(generated_splits_root),
        },
        cwd=Path.cwd(),
    )

    fold1_train = pd.read_csv(generated_splits_root / "fold1_train.csv")
    fold1_val = pd.read_csv(generated_splits_root / "fold1_val.csv")
    fold1_test = pd.read_csv(generated_splits_root / "fold1_test.csv")
    leakage_df = pd.read_csv(generated_splits_root / "cross_rotation_leakage_check.csv")

    assert set(fold1_test["sample_id"]) == {"sample_1"}
    assert set(fold1_val["sample_id"]) == {"sample_2"}
    assert set(fold1_train["sample_id"]) == {
        "sample_3",
        "sample_4",
        "sample_5",
        "sample_6",
        "sample_7",
        "sample_8",
    }
    assert not leakage_df[["train_val_overlap", "train_test_overlap", "val_test_overlap"]].to_numpy().any()
    assert (generated_splits_root / "cross_rotation_summary.csv").exists()
    assert (generated_splits_root / "all_sampled_images.csv").exists()


def test_build_cross_rotation_splits_preserves_roboflow_native_splits(tmp_path: Path) -> None:
    generated_splits_root = tmp_path / "generated_splits"
    processed_manifest_path = generated_splits_root / "processed_manifest.csv"
    generated_splits_root.mkdir(parents=True)

    rows = [
        {
            "image_file_name": "train.jpg",
            "label": "fresh",
            "local_image_path": str(tmp_path / "train.jpg"),
            "roboflow_split": "train",
        },
        {
            "image_file_name": "valid.jpg",
            "label": "not fresh",
            "local_image_path": str(tmp_path / "valid.jpg"),
            "roboflow_split": "valid",
        },
        {
            "image_file_name": "test.jpg",
            "label": "spoiled",
            "local_image_path": str(tmp_path / "test.jpg"),
            "roboflow_split": "test",
        },
    ]
    pd.DataFrame(rows).to_csv(processed_manifest_path, index=False)

    execute_notebook(
        Path("03_build_cross_rotation_splits.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "DATASET_SOURCE": "roboflow",
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "GENERATED_SPLITS_ROOT": str(generated_splits_root),
        },
        cwd=Path.cwd(),
    )

    fold1_train = pd.read_csv(generated_splits_root / "fold1_train.csv")
    fold1_val = pd.read_csv(generated_splits_root / "fold1_val.csv")
    fold1_test = pd.read_csv(generated_splits_root / "fold1_test.csv")

    assert fold1_train["image_file_name"].tolist() == ["train.jpg"]
    assert fold1_val["image_file_name"].tolist() == ["valid.jpg"]
    assert fold1_test["image_file_name"].tolist() == ["test.jpg"]
    assert set(fold1_train["split_type"]) == {"roboflow_native"}

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


def test_build_cross_rotation_splits_generates_eight_roboflow_folds(tmp_path: Path) -> None:
    generated_splits_root = tmp_path / "generated_splits"
    processed_manifest_path = generated_splits_root / "processed_manifest.csv"
    generated_splits_root.mkdir(parents=True)

    rows = []
    labels = ("fresh", "not fresh", "spoiled")
    native_splits = ("train", "valid", "test")
    for index in range(24):
        filename = f"image_{index}.jpg"
        rows.append(
            {
                "image_file_name": filename,
                "label": labels[index % len(labels)],
                "local_image_path": str(tmp_path / filename),
                "roboflow_split": native_splits[index % len(native_splits)],
            }
        )
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

    summary_df = pd.read_csv(generated_splits_root / "cross_rotation_summary.csv")
    assert summary_df["fold"].tolist() == [f"fold{i}" for i in range(1, 9)]

    for fold_index in range(1, 9):
        split_frames = [
            pd.read_csv(generated_splits_root / f"fold{fold_index}_{split}.csv")
            for split in ("train", "val", "test")
        ]
        image_sets = [set(frame["image_file_name"]) for frame in split_frames]
        assert not (image_sets[0] & image_sets[1])
        assert not (image_sets[0] & image_sets[2])
        assert not (image_sets[1] & image_sets[2])
        assert all(set(frame["split_type"]) == {"roboflow_stratified_8fold"} for frame in split_frames)

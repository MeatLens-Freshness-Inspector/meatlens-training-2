from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from meatlens_pork_pipeline.roboflow import (
    build_roboflow_manifest,
    normalize_roboflow_row,
)


def test_normalize_roboflow_row_maps_half_fresh_to_not_fresh(tmp_path: Path) -> None:
    image_root = tmp_path / "train"
    image_root.mkdir()
    image_path = image_root / "sample.jpg"
    image_path.write_bytes(b"image")

    normalized = normalize_roboflow_row(
        pd.Series({"filename": "sample.jpg", "FRESH": 0, "HALF-FRESH": 1, "SPOILED": 0}),
        split="train",
        image_root=tmp_path,
    )

    assert normalized["label"] == "not fresh"
    assert normalized["roboflow_split"] == "train"
    assert normalized["local_image_path"] == str(image_path)
    assert normalized["source_manifest_type"] == "roboflow_manifest"


def test_build_roboflow_manifest_reads_all_native_splits(tmp_path: Path) -> None:
    for split, label_column in (("train", "FRESH"), ("valid", "HALF-FRESH"), ("test", "SPOILED")):
        split_root = tmp_path / split
        split_root.mkdir()
        filename = f"{split}.jpg"
        (split_root / filename).write_bytes(b"image")
        pd.DataFrame(
            [{"filename": filename, "FRESH": int(label_column == "FRESH"), "HALF-FRESH": int(label_column == "HALF-FRESH"), "SPOILED": int(label_column == "SPOILED")}]
        ).to_csv(split_root / "_classes.csv", index=False)

    manifest = build_roboflow_manifest(tmp_path)

    assert manifest["label"].tolist() == ["fresh", "not fresh", "spoiled"]
    assert manifest["roboflow_split"].tolist() == ["train", "valid", "test"]


def test_normalize_roboflow_row_rejects_ambiguous_status(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one active status label"):
        normalize_roboflow_row(
            pd.Series({"filename": "sample.jpg", "FRESH": 1, "HALF-FRESH": 1, "SPOILED": 0}),
            split="train",
            image_root=tmp_path,
        )

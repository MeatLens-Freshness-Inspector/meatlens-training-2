"""Helpers for importing the local Roboflow export."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd


ROBOFLOW_STATUS_COLUMNS = ("FRESH", "HALF-FRESH", "SPOILED")
ROBOFLOW_LABEL_MAP = {
    "FRESH": "fresh",
    "HALF-FRESH": "not fresh",
    "SPOILED": "spoiled",
}
ROBOFLOW_SPLITS = ("train", "valid", "test")


def _is_active(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_roboflow_row(
    row: pd.Series | Mapping[str, object],
    *,
    split: str,
    image_root: Path,
) -> dict[str, object]:
    """Convert one Roboflow CSV row into the project manifest schema."""

    filename = str(row.get("filename", "")).strip()
    if not filename:
        raise ValueError("Roboflow row is missing a filename.")

    active_statuses = [
        status for status in ROBOFLOW_STATUS_COLUMNS if _is_active(row.get(status, 0))
    ]
    if len(active_statuses) != 1:
        raise ValueError(
            f"Roboflow row {filename!r} must have exactly one active status label; "
            f"got {active_statuses or 'none'}."
        )

    normalized_split = str(split).strip().lower()
    if normalized_split not in ROBOFLOW_SPLITS:
        raise ValueError(f"Unsupported Roboflow split: {split!r}")

    image_path = image_root / normalized_split / filename
    if not image_path.is_file():
        raise FileNotFoundError(f"Roboflow image referenced by CSV was not found: {image_path}")

    return {
        "image_name": filename,
        "image_file_name": filename,
        "sample_number": "",
        "sample_id": "",
        "local_image_path": str(image_path),
        "label": ROBOFLOW_LABEL_MAP[active_statuses[0]],
        "roboflow_split": normalized_split,
        "source_manifest_type": "roboflow_manifest",
    }


def build_roboflow_manifest(dataset_root: Path) -> pd.DataFrame:
    """Read ``train``, ``valid`` and ``test`` from a Roboflow export."""

    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Roboflow dataset root was not found: {dataset_root}")

    records: list[dict[str, object]] = []
    for split in ROBOFLOW_SPLITS:
        split_root = dataset_root / split
        classes_path = split_root / "_classes.csv"
        if not classes_path.is_file():
            raise FileNotFoundError(f"Roboflow classes CSV was not found: {classes_path}")

        classes_df = pd.read_csv(classes_path, dtype=str).fillna("")
        if "filename" not in classes_df.columns:
            raise ValueError(f"Roboflow classes CSV has no filename column: {classes_path}")
        records.extend(
            normalize_roboflow_row(row, split=split, image_root=dataset_root)
            for _, row in classes_df.iterrows()
        )

    if not records:
        raise ValueError(f"Roboflow dataset contains no labeled images: {dataset_root}")
    return pd.DataFrame(records)

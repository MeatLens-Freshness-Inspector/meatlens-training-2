from pathlib import Path

import pandas as pd


NORMALIZED_LABELS = {
    "fresh": "fresh",
    "not fresh": "not fresh",
    "not_fresh": "not fresh",
    "spoiled": "spoiled",
}


def normalize_label(value: str) -> str:
    key = value.strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    if key not in NORMALIZED_LABELS:
        raise ValueError(f"Unsupported label: {value}")
    return NORMALIZED_LABELS[key]


def _load_table(manifest_path: Path) -> pd.DataFrame:
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(manifest_path, dtype=str).fillna("")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(manifest_path, dtype=str).fillna("")
    raise ValueError(f"Unsupported manifest file: {manifest_path}")


def _resolve_image_path(image_name: str, images_root: Path) -> Path:
    matches = sorted(path for path in images_root.rglob(image_name) if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one match for {image_name}, found {len(matches)}")
    return matches[0]


def load_manifest(manifest_path: Path, images_root: Path) -> pd.DataFrame:
    manifest_df = _load_table(manifest_path)
    required_columns = {"image_name", "label"}
    missing_columns = required_columns.difference(manifest_df.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")

    manifest_df = manifest_df.copy()
    manifest_df["label"] = manifest_df["label"].map(normalize_label)
    manifest_df["resolved_image_path"] = manifest_df["image_name"].map(
        lambda image_name: str(_resolve_image_path(str(image_name), images_root))
    )
    return manifest_df

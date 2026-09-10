from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from .config import INPUT_SIZE

IMAGE_PATH_COLUMNS = ("processed_image_path", "local_image_path", "processed_output_file")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_PATH_MARKERS = {"data", "roboflow dataset", "generated_splits", "training outputs"}


def _rebase_missing_path(image_path: Path, project_root: Path) -> Path:
    if image_path.exists() or not image_path.is_absolute():
        return image_path

    normalized_parts = [part.casefold() for part in image_path.parts]
    for index, part in enumerate(normalized_parts):
        if part not in PROJECT_PATH_MARKERS:
            continue
        candidate = project_root.joinpath(*image_path.parts[index:])
        if candidate.exists():
            return candidate
    return image_path


def resolve_image_path(
    row: Mapping[str, object],
    *,
    project_root: Path | None = None,
) -> Path:
    for column_name in IMAGE_PATH_COLUMNS:
        value = row.get(column_name)
        if value is None:
            continue

        value_text = str(value).strip()
        if not value_text or value_text.lower() == "nan":
            continue

        return _rebase_missing_path(Path(value_text), project_root or PROJECT_ROOT)

    raise KeyError(
        "Could not resolve an image path from the provided row. "
        f"Expected one of {list(IMAGE_PATH_COLUMNS)}."
    )


def load_image_array(
    row_or_path: Mapping[str, object] | str | Path,
    target_size: tuple[int, int] = INPUT_SIZE,
    *,
    project_root: Path | None = None,
) -> np.ndarray:
    if isinstance(row_or_path, Mapping):
        image_path = resolve_image_path(row_or_path, project_root=project_root)
    else:
        image_path = _rebase_missing_path(Path(row_or_path), project_root or PROJECT_ROOT)

    image = Image.open(image_path).convert("RGB")
    if image.size != target_size:
        image = image.resize(target_size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)

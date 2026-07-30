from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from .config import INPUT_SIZE

IMAGE_PATH_COLUMNS = ("processed_image_path", "local_image_path", "processed_output_file")


def resolve_image_path(row: Mapping[str, object]) -> Path:
    for column_name in IMAGE_PATH_COLUMNS:
        value = row.get(column_name)
        if value is None:
            continue

        value_text = str(value).strip()
        if not value_text or value_text.lower() == "nan":
            continue

        return Path(value_text)

    raise KeyError(
        "Could not resolve an image path from the provided row. "
        f"Expected one of {list(IMAGE_PATH_COLUMNS)}."
    )


def load_image_array(
    row_or_path: Mapping[str, object] | str | Path,
    target_size: tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    if isinstance(row_or_path, Mapping):
        image_path = resolve_image_path(row_or_path)
    else:
        image_path = Path(row_or_path)

    image = Image.open(image_path).convert("RGB")
    if image.size != target_size:
        image = image.resize(target_size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)

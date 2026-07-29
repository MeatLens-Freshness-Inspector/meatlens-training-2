from pathlib import Path

import numpy as np
from PIL import Image

from apply_hsv_lab_threshold_roi_batch import (
    apply_background_fill,
    clean_mask,
    failure_flag,
    get_hsv_channels,
    preprocess_center_square_resize_224,
)


def _get_lab_channels(image_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = image_uint8.astype(np.float32) / 255.0
    rgb_linear = np.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92,
    )

    x = (
        (rgb_linear[:, :, 0] * 0.4124564)
        + (rgb_linear[:, :, 1] * 0.3575761)
        + (rgb_linear[:, :, 2] * 0.1804375)
    ) / 0.95047
    y = (
        (rgb_linear[:, :, 0] * 0.2126729)
        + (rgb_linear[:, :, 1] * 0.7151522)
        + (rgb_linear[:, :, 2] * 0.0721750)
    ) / 1.00000
    z = (
        (rgb_linear[:, :, 0] * 0.0193339)
        + (rgb_linear[:, :, 1] * 0.1191920)
        + (rgb_linear[:, :, 2] * 0.9503041)
    ) / 1.08883

    delta = 6.0 / 29.0
    delta_cubed = delta**3
    scale = 1.0 / (3.0 * delta**2)
    offset = 4.0 / 29.0

    def transform(channel: np.ndarray) -> np.ndarray:
        return np.where(channel > delta_cubed, np.cbrt(channel), (channel * scale) + offset)

    fx = transform(x)
    fy = transform(y)
    fz = transform(z)
    l = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return l.astype(np.float32), a.astype(np.float32), b.astype(np.float32)


def process_image(path: Path, background_mode: str = "gray") -> tuple[np.ndarray, dict[str, object]]:
    image = Image.open(path).convert("RGB")
    processed = preprocess_center_square_resize_224(image)
    image_uint8 = np.array(processed, dtype=np.uint8)

    _, saturation, value = get_hsv_channels(image_uint8)
    _, a_channel, _ = _get_lab_channels(image_uint8)
    raw_mask = (
        (saturation >= 20)
        & (value >= 35)
        & ~((value >= 235) & (saturation <= 30))
        & (a_channel >= 6.0)
    )
    final_mask, quality = clean_mask(raw_mask)
    failed = failure_flag(final_mask, quality)

    if failed or not final_mask.any():
        output_uint8 = image_uint8.copy()
    else:
        output_uint8 = apply_background_fill(image_uint8, final_mask, mode=background_mode)

    metadata: dict[str, object] = {
        "segmentation_failed": failed,
        "mask_area_ratio": float(quality["mask_area_ratio"]),
        "center_overlap_ratio": float(quality["center_overlap_ratio"]),
        "number_of_components": int(quality["number_of_components"]),
        "touches_border": bool(quality["touches_border"]),
    }
    return output_uint8, metadata

__all__ = ["process_image"]

"""TensorFlow input pipelines for image training and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import INPUT_SIZE, LABEL_ORDER
from .image_io import resolve_image_path


def dataframe_to_paths_and_labels(
    dataframe: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """Resolve manifest rows once, preserving their order and labels."""

    required_columns = {"label"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing_columns)}")

    paths: list[str] = []
    labels: list[int] = []
    for row in dataframe.to_dict(orient="records"):
        label = str(row["label"])
        if label not in LABEL_ORDER:
            raise ValueError(f"Unsupported label: {label!r}")
        paths.append(str(resolve_image_path(row)))
        labels.append(LABEL_ORDER.index(label))
    return paths, np.asarray(labels, dtype=np.int32)


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_image(
        tf.io.read_file(path),
        channels=3,
        expand_animations=False,
    )
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, INPUT_SIZE, method="bilinear")
    image = tf.ensure_shape(tf.cast(image, tf.float32), (*INPUT_SIZE, 3))
    return image, tf.cast(label, tf.int32)


def build_image_dataset(
    dataframe: pd.DataFrame,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    cache_mode: str | Path = "memory",
    seed: int | None = None,
) -> tf.data.Dataset:
    """Build a decoded, optionally cached, batched and prefetched image dataset.

    Caching is deliberately applied after deterministic decoding/resizing and
    before shuffling and model-owned random augmentation. This keeps the cache
    reusable across epochs without freezing augmentation choices.
    """

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    paths, labels = dataframe_to_paths_and_labels(dataframe)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

    if isinstance(cache_mode, Path):
        cache_mode.parent.mkdir(parents=True, exist_ok=True)
        dataset = dataset.cache(str(cache_mode))
    elif cache_mode == "memory":
        dataset = dataset.cache()
    elif cache_mode != "none":
        raise ValueError("cache_mode must be 'none', 'memory', or a pathlib.Path")

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=max(len(paths), 1),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


__all__ = ["build_image_dataset", "dataframe_to_paths_and_labels"]

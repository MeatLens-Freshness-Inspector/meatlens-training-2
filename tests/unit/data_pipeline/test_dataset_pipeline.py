from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from meatlens_pork_pipeline.dataset_pipeline import (
    build_image_dataset,
    dataframe_to_paths_and_labels,
)


def test_dataframe_to_paths_and_labels_preserves_row_order_and_label_indices(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (224, 224), (255, 0, 0)).save(first)
    Image.new("RGB", (224, 224), (0, 0, 255)).save(second)
    frame = pd.DataFrame(
        [
            {"local_image_path": str(first), "label": "spoiled"},
            {"local_image_path": str(second), "label": "fresh"},
        ]
    )

    paths, labels = dataframe_to_paths_and_labels(frame)

    assert paths == [str(first), str(second)]
    np.testing.assert_array_equal(labels, [2, 0])


def test_build_image_dataset_returns_static_float_images_and_integer_labels(tmp_path):
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (16, 12), (10, 20, 30)).save(image)
    frame = pd.DataFrame([{"local_image_path": str(image), "label": "fresh"}])

    dataset = build_image_dataset(frame, batch_size=1, shuffle=False, cache_mode="none")
    images, labels = next(iter(dataset))

    assert images.shape == (1, 224, 224, 3)
    assert images.dtype == np.float32
    assert labels.shape == (1,)
    assert labels.dtype == np.int32
    assert int(labels[0]) == 0


def test_build_image_dataset_uses_cache_before_batching_when_requested(tmp_path):
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (224, 224), (10, 20, 30)).save(image)
    frame = pd.DataFrame([{"local_image_path": str(image), "label": "fresh"}])

    dataset = build_image_dataset(frame, batch_size=1, shuffle=False, cache_mode="memory")

    cache_dataset = dataset._input_dataset._input_dataset
    assert "CacheDataset" in type(cache_dataset).__name__

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from meatlens_pork_pipeline.embeddings import build_linear_ovr_classifier_model, cache_dataframe_embeddings
from meatlens_pork_pipeline.evaluation import evaluate_model
from meatlens_pork_pipeline.image_io import resolve_image_path
from meatlens_pork_pipeline.training import train_model
import meatlens_pork_pipeline.training as training_module


def _build_toy_feature_extractor() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")
    x = tf.keras.layers.Rescaling(scale=1.0 / 255.0, name="toy_rescale")(inputs)
    outputs = tf.keras.layers.GlobalAveragePooling2D(name="toy_avg_pool")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="toy_feature_extractor")


def _write_color_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224), color).save(path)


def _build_split_csv(tmp_path: Path, split_name: str, items_per_class: int) -> Path:
    rows: list[dict[str, object]] = []
    label_colors = {
        "fresh": (230, 70, 70),
        "not fresh": (70, 230, 70),
        "spoiled": (70, 70, 230),
    }

    for label, color in label_colors.items():
        for index in range(items_per_class):
            image_path = tmp_path / split_name / label / f"{label.replace(' ', '_')}_{index}.jpg"
            _write_color_image(image_path, color)
            rows.append(
                {
                    "label": label,
                    "local_image_path": str(image_path),
                }
            )

    csv_path = tmp_path / f"{split_name}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_resolve_image_path_accepts_split_csv_aliases(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    _write_color_image(image_path, (200, 100, 90))

    local_row = {"local_image_path": str(image_path)}
    processed_output_row = {"processed_output_file": str(image_path)}

    assert resolve_image_path(local_row) == image_path
    assert resolve_image_path(processed_output_row) == image_path


def test_cache_dataframe_embeddings_writes_feature_arrays(tmp_path: Path) -> None:
    image_a = tmp_path / "cache_inputs" / "fresh.jpg"
    image_b = tmp_path / "cache_inputs" / "spoiled.jpg"
    _write_color_image(image_a, (255, 0, 0))
    _write_color_image(image_b, (0, 0, 255))

    cache_df = pd.DataFrame(
        [
            {"label": "fresh", "processed_output_file": str(image_a)},
            {"label": "spoiled", "processed_output_file": str(image_b)},
        ]
    )

    cache_path = cache_dataframe_embeddings(
        df=cache_df,
        feature_extractor=_build_toy_feature_extractor(),
        output_path=tmp_path / "embedding_cache" / "split_embeddings.npz",
        batch_size=2,
    )

    payload = np.load(cache_path, allow_pickle=True)
    assert payload["features"].shape == (2, 3)
    assert payload["labels"].tolist() == [0, 2]
    assert payload["paths"].tolist() == [str(image_a), str(image_b)]


def test_train_model_cached_embeddings_saves_image_classifier(tmp_path: Path, monkeypatch) -> None:
    train_csv = _build_split_csv(tmp_path, "train", items_per_class=4)
    val_csv = _build_split_csv(tmp_path, "val", items_per_class=2)
    test_csv = _build_split_csv(tmp_path, "test", items_per_class=2)

    monkeypatch.setattr(training_module, "build_feature_extractor_model", lambda **_: _build_toy_feature_extractor())

    artifacts = train_model(
        train_csv=train_csv,
        val_csv=val_csv,
        output_dir=tmp_path / "models",
        seed=42,
        training_strategy="cached_embeddings_sgd_v1",
    )

    assert artifacts.model_h5_path.exists()
    assert artifacts.history_csv_path.exists()
    assert artifacts.embedding_cache_dir == tmp_path / "models" / "embedding_cache"
    assert (artifacts.embedding_cache_dir / "train_embeddings.npz").exists()
    assert (artifacts.embedding_cache_dir / "val_embeddings.npz").exists()

    evaluation = evaluate_model(
        model_h5_path=artifacts.model_h5_path,
        test_csv=test_csv,
        output_dir=tmp_path / "evaluation",
        batch_size=2,
    )

    assert evaluation.test_count == 6
    assert evaluation.metrics["accuracy"] >= 0.99
    assert evaluation.confusion_matrix_csv_path.exists()
    assert evaluation.confusion_matrix_png_path.exists()


def test_linear_ovr_keras_model_avoids_zero_probability_rows_for_extreme_negative_logits() -> None:
    model = build_linear_ovr_classifier_model(
        feature_dim=2,
        num_classes=3,
        scaler_mean=np.zeros(2, dtype=np.float32),
        scaler_scale=np.ones(2, dtype=np.float32),
        coefficients=np.zeros((3, 2), dtype=np.float32),
        intercept=np.array([-200.0, -200.0, -200.0], dtype=np.float32),
    )

    probabilities = model.predict(np.zeros((2, 2), dtype=np.float32), verbose=0)

    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        probabilities,
        np.full((2, 3), 1.0 / 3.0, dtype=np.float32),
        atol=1e-6,
    )

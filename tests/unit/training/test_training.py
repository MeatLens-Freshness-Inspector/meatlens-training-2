import tensorflow as tf
import pandas as pd

from meatlens_pork_pipeline.training import (
    build_mobilenetv3small_model,
    compute_class_weights,
    _build_end_to_end_datasets,
)
import pytest


def test_build_mobilenetv3small_model_has_three_class_softmax() -> None:
    model = build_mobilenetv3small_model(weights=None)
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 3)
    assert model.layers[-1].activation.__name__ == "softmax"


def test_build_mobilenetv3small_model_uses_stable_training_defaults() -> None:
    model = build_mobilenetv3small_model(weights=None)

    assert "dense_128" not in {layer.name for layer in model.layers}
    assert model.optimizer.learning_rate.numpy() == pytest.approx(1e-4)
    assert model.loss.label_smoothing == pytest.approx(0.0)


def test_build_mobilenetv3small_model_uses_nonfused_batchnorm_for_deterministic_gpu_finetuning() -> None:
    model = build_mobilenetv3small_model(weights=None)

    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    batch_norm_layers = [
        layer for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]

    assert batch_norm_layers
    assert all(getattr(layer, "fused", None) is False for layer in batch_norm_layers)


def test_compute_class_weights_returns_all_three_indices() -> None:
    class_weights = compute_class_weights(["fresh", "fresh", "not fresh", "spoiled"])
    assert set(class_weights) == {0, 1, 2}


def test_build_mobilenetv3small_model_can_own_training_augmentation() -> None:
    model = build_mobilenetv3small_model(weights=None, augmentation=True)

    augmentation = model.get_layer("training_augmentation")
    assert {
        layer.__class__.__name__ for layer in augmentation.layers
    } == {"RandomFlip", "RandomRotation", "RandomZoom", "RandomTranslation"}


def test_end_to_end_datasets_use_tensorflow_pipeline_for_each_split(monkeypatch) -> None:
    calls = []

    def fake_build(dataframe, *, batch_size, shuffle, cache_mode, seed):
        calls.append((dataframe, batch_size, shuffle, cache_mode, seed))
        return f"dataset-{len(calls)}"

    monkeypatch.setattr("meatlens_pork_pipeline.training.build_image_dataset", fake_build)
    train_df = pd.DataFrame([{"label": "fresh", "local_image_path": "train.jpg"}])
    val_df = pd.DataFrame([{"label": "spoiled", "local_image_path": "val.jpg"}])

    train_dataset, val_dataset = _build_end_to_end_datasets(
        train_df,
        val_df,
        batch_size=64,
        cache_mode="none",
        seed=42,
    )

    assert (train_dataset, val_dataset) == ("dataset-1", "dataset-2")
    assert calls == [
        (train_df, 64, True, "none", 42),
        (val_df, 64, False, "none", 42),
    ]

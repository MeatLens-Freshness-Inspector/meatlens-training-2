from __future__ import annotations

import pytest
import tensorflow as tf
from tensorflow.keras import layers

from meatlens_pork_pipeline.modeling import (
    build_classification_head,
    build_classification_loss,
)


def test_build_classification_head_linear_variant_adds_only_prediction_layer() -> None:
    inputs = tf.keras.Input(shape=(16,), name="features")
    outputs = build_classification_head(inputs, num_classes=3, head_variant="linear_v1")
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    assert model.output_shape == (None, 3)
    assert all(layer.name != "dense_128" for layer in model.layers)


def test_build_classification_head_mlp_variant_preserves_hidden_dense_layer() -> None:
    inputs = tf.keras.Input(shape=(16,), name="features")
    outputs = build_classification_head(inputs, num_classes=3, head_variant="mlp_v1")
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    assert any(layer.name == "dense_128" for layer in model.layers)


def test_build_classification_head_rejects_unknown_variant() -> None:
    inputs = tf.keras.Input(shape=(16,), name="features")

    with pytest.raises(ValueError, match="Unsupported head_variant"):
        build_classification_head(inputs, num_classes=3, head_variant="unknown")


def test_build_classification_loss_uses_requested_label_smoothing() -> None:
    loss = build_classification_loss(label_smoothing=0.05)

    assert isinstance(loss, tf.keras.losses.CategoricalCrossentropy)
    assert loss.label_smoothing == pytest.approx(0.05)


def test_build_classification_loss_rejects_invalid_label_smoothing() -> None:
    with pytest.raises(ValueError, match="label_smoothing"):
        build_classification_loss(label_smoothing=1.2)

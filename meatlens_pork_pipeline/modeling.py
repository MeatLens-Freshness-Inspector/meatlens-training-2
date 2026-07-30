from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def build_classification_head(
    features: tf.Tensor,
    num_classes: int,
    head_variant: str = "linear_v1",
) -> tf.Tensor:
    if head_variant == "linear_v1":
        features = layers.Dropout(0.2, name="dropout_1")(features)
        return layers.Dense(num_classes, activation="softmax", name="predictions")(features)

    if head_variant == "mlp_v1":
        features = layers.Dropout(0.2, name="dropout_1")(features)
        features = layers.Dense(128, activation="relu", name="dense_128")(features)
        features = layers.Dropout(0.1, name="dropout_2")(features)
        return layers.Dense(num_classes, activation="softmax", name="predictions")(features)

    raise ValueError(
        f"Unsupported head_variant {head_variant!r}. Expected one of ['linear_v1', 'mlp_v1']."
    )


def build_classification_loss(
    label_smoothing: float = 0.0,
) -> tf.keras.losses.CategoricalCrossentropy:
    smoothing = float(label_smoothing)
    if not 0.0 <= smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0.0, 1.0). Got {smoothing}.")

    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)

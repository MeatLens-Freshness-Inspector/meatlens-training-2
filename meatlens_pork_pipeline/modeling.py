from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="meatlens")
class MacroF1Metric(tf.keras.metrics.Metric):
    """Unweighted macro-F1 accumulated over a complete validation pass."""

    def __init__(self, num_classes: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
            dtype=tf.float32,
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        del sample_weight
        if y_true.shape.rank is not None and y_true.shape.rank > 1:
            # Sparse Keras labels may arrive as (batch, 1); only a final
            # dimension greater than one represents one-hot class labels.
            if y_true.shape[-1] == 1:
                y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.int32)
            else:
                y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        else:
            y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        batch_confusion = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            dtype=self.confusion_matrix.dtype,
        )
        self.confusion_matrix.assign_add(batch_confusion)

    def result(self) -> tf.Tensor:
        true_positive = tf.linalg.diag_part(self.confusion_matrix)
        actual = tf.reduce_sum(self.confusion_matrix, axis=1)
        predicted = tf.reduce_sum(self.confusion_matrix, axis=0)
        precision = tf.math.divide_no_nan(true_positive, predicted)
        recall = tf.math.divide_no_nan(true_positive, actual)
        f1 = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
        return tf.reduce_mean(f1)

    def reset_state(self) -> None:
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


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

    if head_variant == "training1_mlp_v1":
        features = layers.Dropout(0.30, name="image_dropout")(features)
        features = layers.Dense(128, activation="relu", name="dense_128")(features)
        features = layers.Dropout(0.30, name="dense_dropout")(features)
        return layers.Dense(num_classes, activation="softmax", name="classification_head")(features)

    raise ValueError(
        f"Unsupported head_variant {head_variant!r}. Expected one of "
        "['linear_v1', 'mlp_v1', 'training1_mlp_v1']."
    )


def build_classification_loss(
    label_smoothing: float = 0.0,
) -> tf.keras.losses.CategoricalCrossentropy:
    smoothing = float(label_smoothing)
    if not 0.0 <= smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0.0, 1.0). Got {smoothing}.")

    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)

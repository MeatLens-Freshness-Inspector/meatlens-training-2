from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small

from .config import INPUT_SIZE, LABEL_ORDER
from .image_io import load_image_array, resolve_image_path
from .modeling import build_classification_head, build_classification_loss


@tf.keras.utils.register_keras_serializable(package="meatlens")
class StandardizeFeatures(layers.Layer):
    def __init__(self, mean: list[float] | np.ndarray, scale: list[float] | np.ndarray, **kwargs) -> None:
        super().__init__(**kwargs)
        mean_array = np.asarray(mean, dtype=np.float32).reshape((-1,))
        scale_array = np.asarray(scale, dtype=np.float32).reshape((-1,))
        scale_array[scale_array == 0.0] = 1.0
        self._mean_values = mean_array.tolist()
        self._scale_values = scale_array.tolist()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        mean = tf.reshape(tf.constant(self._mean_values, dtype=inputs.dtype), shape=(1, -1))
        scale = tf.reshape(tf.constant(self._scale_values, dtype=inputs.dtype), shape=(1, -1))
        return tf.math.divide_no_nan(inputs - mean, scale)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "mean": self._mean_values,
                "scale": self._scale_values,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="meatlens")
class NormalizeProbabilities(layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        row_sums = tf.reduce_sum(inputs, axis=-1, keepdims=True)
        return tf.math.divide_no_nan(inputs, row_sums)


def build_feature_extractor_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    backbone = MobileNetV3Small(
        include_top=False,
        include_preprocessing=False,
        weights=weights,
        input_shape=input_shape,
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image_input")
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv3_rescale")(inputs)
    x = backbone(x, training=False)
    outputs = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="meatlens_feature_extractor")


def build_embedding_classifier_model(
    feature_dim: int,
    num_classes: int = 3,
    learning_rate: float = 1e-4,
    label_smoothing: float = 0.0,
    head_variant: str = "linear_v1",
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(feature_dim,), name="feature_input")
    outputs = build_classification_head(inputs, num_classes=num_classes, head_variant=head_variant)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="meatlens_embedding_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=build_classification_loss(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    return model


def cache_dataframe_embeddings(
    df: pd.DataFrame,
    feature_extractor: tf.keras.Model,
    output_path: Path,
    batch_size: int = 32,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = df.to_dict(orient="records")
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_arrays = np.stack([load_image_array(row, target_size=INPUT_SIZE) for row in batch_rows], axis=0)
        batch_features = feature_extractor.predict(batch_arrays, verbose=0)
        features.append(np.asarray(batch_features, dtype=np.float32))
        labels.extend(LABEL_ORDER.index(str(row["label"])) for row in batch_rows)
        paths.extend(str(resolve_image_path(row)) for row in batch_rows)

    feature_array = np.concatenate(features, axis=0) if features else np.zeros((0, 0), dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int32)
    path_array = np.asarray(paths, dtype=object)

    np.savez_compressed(
        output_path,
        features=feature_array,
        labels=label_array,
        paths=path_array,
    )
    return output_path


def load_cached_embeddings(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=True) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int32)
    return features, labels


def assemble_image_classifier_model(
    feature_extractor: tf.keras.Model,
    classifier_model: tf.keras.Model,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), name="image_input")
    features = feature_extractor(inputs, training=False)
    outputs = classifier_model(features, training=False)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="meatlens_mobilenetv3small_cached_embeddings")


def fit_linear_ovr_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
    alpha: float = 1e-4,
) -> tuple[StandardScaler, SGDClassifier]:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(train_features)
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        penalty="l2",
        max_iter=5000,
        tol=1e-4,
        class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(scaled_features, train_labels)
    return scaler, classifier


def predict_linear_ovr_probabilities(
    features: np.ndarray,
    scaler: StandardScaler,
    classifier: SGDClassifier,
) -> np.ndarray:
    scaled_features = scaler.transform(features)
    logits = np.asarray(classifier.decision_function(scaled_features), dtype=np.float32)
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)

    sigmoid_scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    row_sums = sigmoid_scores.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        sigmoid_scores,
        row_sums,
        out=np.full_like(sigmoid_scores, 1.0 / sigmoid_scores.shape[1]),
        where=row_sums != 0,
    )
    return np.asarray(probabilities, dtype=np.float32)


def build_linear_ovr_classifier_model(
    feature_dim: int,
    num_classes: int,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    coefficients: np.ndarray,
    intercept: np.ndarray,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(feature_dim,), name="feature_input")
    standardized = StandardizeFeatures(mean=scaler_mean, scale=scaler_scale, name="feature_standardization")(inputs)
    logits_layer = layers.Dense(num_classes, activation=None, name="ovr_logits")
    logits = logits_layer(standardized)
    sigmoid_scores = layers.Activation("sigmoid", name="ovr_sigmoid")(logits)
    outputs = NormalizeProbabilities(name="predictions")(sigmoid_scores)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="meatlens_embedding_classifier_sgd")
    logits_layer.set_weights(
        [
            np.asarray(coefficients, dtype=np.float32).T,
            np.asarray(intercept, dtype=np.float32),
        ]
    )
    return model


def build_linear_ovr_image_classifier_model(
    feature_extractor: tf.keras.Model,
    scaler: StandardScaler,
    classifier: SGDClassifier,
) -> tf.keras.Model:
    classifier_model = build_linear_ovr_classifier_model(
        feature_dim=int(classifier.coef_.shape[1]),
        num_classes=int(classifier.coef_.shape[0]),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
        coefficients=np.asarray(classifier.coef_, dtype=np.float32),
        intercept=np.asarray(classifier.intercept_, dtype=np.float32),
    )
    return assemble_image_classifier_model(feature_extractor, classifier_model)

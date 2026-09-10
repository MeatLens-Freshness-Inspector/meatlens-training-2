from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight as sklearn_compute_class_weight
from tensorflow.keras import callbacks, layers
from tensorflow.keras.applications import MobileNetV3Small

from .augmentation import build_training_augmentation
from .config import (
    END_TO_END_DEFAULTS,
    INPUT_SIZE,
    LABEL_ORDER,
    ROBOFLOW_CACHED_BASELINE_STRATEGY,
    TRAINING1_COMPATIBLE_STRATEGY,
)
from .embeddings import (
    assemble_image_classifier_model,
    build_embedding_classifier_model,
    build_feature_extractor_model,
    build_linear_ovr_image_classifier_model,
    cache_dataframe_embeddings,
    fit_linear_ovr_classifier,
    load_cached_embeddings,
    predict_linear_ovr_probabilities,
)
from .image_io import load_image_array
from .modeling import build_classification_head, build_classification_loss


@dataclass(frozen=True)
class TrainingArtifacts:
    model_h5_path: Path
    checkpoint_path: Path
    history_csv_path: Path
    train_count: int
    val_count: int
    class_weights: dict[int, float]
    training_strategy: str
    embedding_cache_dir: Path | None = None


class CsvImageSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        shuffle: bool = False,
        use_augmentation: bool = False,
        augmentation_preset: str = "geometry_only_v1",
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.use_augmentation = bool(use_augmentation)
        self.augmentation = build_training_augmentation(augmentation_preset)
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]

        rows = batch_df.to_dict(orient="records")
        images = np.stack(
            [load_image_array(row, target_size=INPUT_SIZE) for row in rows],
            axis=0,
        ).astype(np.float32)
        if self.use_augmentation:
            images = self.augmentation(images, training=True).numpy()
        labels = [LABEL_ORDER.index(str(row["label"])) for row in rows]

        return (
            np.stack(images, axis=0),
            tf.keras.utils.to_categorical(labels, num_classes=len(LABEL_ORDER)),
        )

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indexes)


def _disable_backbone_batch_norm_fusion(backbone: tf.keras.Model) -> None:
    """Use deterministic-compatible BatchNorm kernels during fine-tuning.

    TensorFlow 2.10 has no deterministic GPU gradient for fused BatchNorm when
    the layer is called with ``training=False``. The backbone deliberately
    keeps BatchNorm statistics frozen during fine-tuning, so use the unfused
    implementation before the backbone is connected to the classifier graph.
    Newer Keras versions no longer expose ``fused`` but allow the attribute;
    setting it keeps this code compatible across the supported environments.
    """
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.fused = False


def build_mobilenetv3small_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 3,
    weights: str | None = "imagenet",
    learning_rate: float = 1e-4,
    label_smoothing: float = 0.0,
    head_variant: str = "linear_v1",
) -> tf.keras.Model:
    backbone = MobileNetV3Small(
        include_top=False,
        include_preprocessing=False,
        weights=weights,
        input_shape=input_shape,
    )
    _disable_backbone_batch_norm_fusion(backbone)
    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image_input")
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv3_rescale")(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = build_classification_head(x, num_classes=num_classes, head_variant=head_variant)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="meatlens_mobilenetv3small_cnn_only")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=build_classification_loss(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(labels: pd.Series | list[str]) -> dict[int, float]:
    label_list = [str(label) for label in labels]
    label_indices = np.array([LABEL_ORDER.index(label) for label in label_list], dtype=int)
    weights = sklearn_compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(range(len(LABEL_ORDER))), dtype=int),
        y=label_indices,
    )
    return {int(index): float(weight) for index, weight in enumerate(weights)}


def _set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("Expected a nested backbone model in the classifier.")


class ValidationMacroF1Callback(tf.keras.callbacks.Callback):
    """Compute macro-F1 on the unaugmented validation sequence after each epoch."""

    def __init__(self, validation_sequence: CsvImageSequence) -> None:
        super().__init__()
        self.validation_sequence = validation_sequence

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        del epoch
        logs = logs if logs is not None else {}
        y_true: list[int] = []
        y_pred: list[int] = []
        for batch_index in range(len(self.validation_sequence)):
            images, labels = self.validation_sequence[batch_index]
            probabilities = self.model(images, training=False).numpy()
            y_true.extend(np.argmax(labels, axis=1).tolist())
            y_pred.extend(np.argmax(probabilities, axis=1).tolist())
        logs["val_f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )


def _build_callbacks(checkpoint_path: Path) -> list[callbacks.Callback]:
    return [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def _history_to_frame(
    phase_histories: list[tuple[str, tf.keras.callbacks.History]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    epoch_offset = 0
    for phase_name, history in phase_histories:
        epoch_count = len(history.epoch)
        for index in range(epoch_count):
            row: dict[str, object] = {
                "phase": phase_name,
                "epoch": epoch_offset + index + 1,
            }
            for metric_name, values in history.history.items():
                row[metric_name] = float(values[index])
            rows.append(row)
        epoch_offset += epoch_count
    return pd.DataFrame(rows)


def _linear_history_frame(
    train_labels: np.ndarray,
    train_probabilities: np.ndarray,
    val_labels: np.ndarray,
    val_probabilities: np.ndarray,
    phase_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "phase": phase_name,
                "epoch": 1,
                "accuracy": float(accuracy_score(train_labels, train_probabilities.argmax(axis=1))),
                "loss": float(log_loss(train_labels, train_probabilities, labels=np.arange(len(LABEL_ORDER)))),
                "val_accuracy": float(accuracy_score(val_labels, val_probabilities.argmax(axis=1))),
                "val_loss": float(log_loss(val_labels, val_probabilities, labels=np.arange(len(LABEL_ORDER)))),
            }
        ]
    )


def train_model(
    train_csv: Path,
    val_csv: Path,
    output_dir: Path,
    seed: int,
    epochs_head: int = 8,
    epochs_fine: int = 20,
    head_lr: float = 5e-4,
    fine_tune_lr: float = 1e-5,
    label_smoothing: float = 0.0,
    head_variant: str = "linear_v1",
    training_strategy: str = "cached_embeddings_sgd_v1",
    weights: str | None = "imagenet",
    batch_size: int = 32,
    augmentation: bool = True,
    fine_tune_fraction: float = 0.25,
) -> TrainingArtifacts:
    _set_training_seed(seed)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "meatlens_mobilenetv3small_pork_cnn_only_best.keras"
    model_h5_path = output_dir / "meatlens_mobilenetv3small_pork_cnn_only.keras"
    history_csv_path = output_dir / "training_history.csv"

    if training_strategy in ("end_to_end",):
        training_strategy = TRAINING1_COMPATIBLE_STRATEGY
    elif training_strategy in ("cached_embeddings_sgd_v1", ROBOFLOW_CACHED_BASELINE_STRATEGY):
        training_strategy = ROBOFLOW_CACHED_BASELINE_STRATEGY

    class_weights = compute_class_weights(train_df["label"])
    if training_strategy == TRAINING1_COMPATIBLE_STRATEGY:
        train_sequence = CsvImageSequence(
            train_df,
            batch_size=batch_size,
            shuffle=True,
            use_augmentation=augmentation,
            augmentation_preset="geometry_only_v1",
        )
        val_sequence = CsvImageSequence(
            val_df,
            batch_size=batch_size,
            shuffle=False,
            use_augmentation=False,
            augmentation_preset="geometry_only_v1",
        )
        model = build_mobilenetv3small_model(
            weights=weights,
            learning_rate=head_lr,
            label_smoothing=0.0,
            head_variant="training1_mlp_v1",
        )
        fit_callbacks = [
            ValidationMacroF1Callback(val_sequence),
            callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=END_TO_END_DEFAULTS["monitor"],
                mode="max",
                save_best_only=True,
            ),
            callbacks.EarlyStopping(
                monitor=END_TO_END_DEFAULTS["monitor"],
                mode="max",
                patience=4,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor=END_TO_END_DEFAULTS["monitor"],
                mode="max",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
            ),
        ]
        head_history = model.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=epochs_head,
            class_weight=class_weights,
            callbacks=fit_callbacks,
            verbose=2,
        )

        backbone = _find_backbone(model)
        backbone.trainable = True
        fine_tune_at = max(
            int(len(backbone.layers) * (1.0 - fine_tune_fraction)),
            1,
        )
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=fine_tune_lr
            ),
            loss=build_classification_loss(label_smoothing=0.0),
            metrics=["accuracy"],
        )
        fine_history = model.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=epochs_fine,
            class_weight=class_weights,
            callbacks=fit_callbacks,
            verbose=2,
        )
        history_df = _history_to_frame([("head", head_history), ("fine_tune", fine_history)])
        history_df.to_csv(history_csv_path, index=False)

        if checkpoint_path.exists():
            best_model = tf.keras.models.load_model(checkpoint_path, compile=False)
        else:
            best_model = model
        best_model.save(model_h5_path, include_optimizer=False)
        return TrainingArtifacts(
            model_h5_path=model_h5_path,
            checkpoint_path=checkpoint_path,
            history_csv_path=history_csv_path,
            train_count=len(train_df),
            val_count=len(val_df),
            class_weights=class_weights,
            training_strategy=TRAINING1_COMPATIBLE_STRATEGY,
        )

    if training_strategy == ROBOFLOW_CACHED_BASELINE_STRATEGY:
        embedding_cache_dir = output_dir / "embedding_cache"
        feature_extractor = build_feature_extractor_model(weights=weights)
        train_cache_path = embedding_cache_dir / "train_embeddings.npz"
        val_cache_path = embedding_cache_dir / "val_embeddings.npz"

        if not train_cache_path.exists():
            cache_dataframe_embeddings(
                df=train_df,
                feature_extractor=feature_extractor,
                output_path=train_cache_path,
                batch_size=32,
            )
        if not val_cache_path.exists():
            cache_dataframe_embeddings(
                df=val_df,
                feature_extractor=feature_extractor,
                output_path=val_cache_path,
                batch_size=32,
            )

        train_features, train_labels = load_cached_embeddings(train_cache_path)
        val_features, val_labels = load_cached_embeddings(val_cache_path)

        scaler, classifier = fit_linear_ovr_classifier(
            train_features=train_features,
            train_labels=train_labels,
            seed=seed,
        )
        train_probabilities = predict_linear_ovr_probabilities(train_features, scaler, classifier)
        val_probabilities = predict_linear_ovr_probabilities(val_features, scaler, classifier)

        history_df = _linear_history_frame(
            train_labels=train_labels,
            train_probabilities=train_probabilities,
            val_labels=val_labels,
            val_probabilities=val_probabilities,
            phase_name="cached_embeddings_sgd",
        )
        history_df.to_csv(history_csv_path, index=False)

        best_model = build_linear_ovr_image_classifier_model(feature_extractor, scaler, classifier)
        best_model.save(checkpoint_path, include_optimizer=False)
        best_model.save(model_h5_path, include_optimizer=False)

        return TrainingArtifacts(
            model_h5_path=model_h5_path,
            checkpoint_path=checkpoint_path,
            history_csv_path=history_csv_path,
            train_count=len(train_df),
            val_count=len(val_df),
            class_weights=class_weights,
            training_strategy=ROBOFLOW_CACHED_BASELINE_STRATEGY,
            embedding_cache_dir=embedding_cache_dir,
        )

    if training_strategy == "cached_embeddings_v1":
        embedding_cache_dir = output_dir / "embedding_cache"
        head_checkpoint_path = output_dir / "meatlens_mobilenetv3small_pork_cnn_only_head_best.keras"
        feature_extractor = build_feature_extractor_model(weights=weights)
        train_cache_path = embedding_cache_dir / "train_embeddings.npz"
        val_cache_path = embedding_cache_dir / "val_embeddings.npz"

        if not train_cache_path.exists():
            cache_dataframe_embeddings(
                df=train_df,
                feature_extractor=feature_extractor,
                output_path=train_cache_path,
                batch_size=32,
            )
        if not val_cache_path.exists():
            cache_dataframe_embeddings(
                df=val_df,
                feature_extractor=feature_extractor,
                output_path=val_cache_path,
                batch_size=32,
            )

        train_features, train_labels = load_cached_embeddings(train_cache_path)
        val_features, val_labels = load_cached_embeddings(val_cache_path)

        classifier_model = build_embedding_classifier_model(
            feature_dim=int(train_features.shape[1]),
            num_classes=len(LABEL_ORDER),
            learning_rate=head_lr,
            label_smoothing=label_smoothing,
            head_variant=head_variant,
        )
        fit_callbacks = _build_callbacks(head_checkpoint_path)
        target_epochs = max(epochs_head + epochs_fine, 1)
        history = classifier_model.fit(
            train_features,
            tf.keras.utils.to_categorical(train_labels, num_classes=len(LABEL_ORDER)),
            validation_data=(
                val_features,
                tf.keras.utils.to_categorical(val_labels, num_classes=len(LABEL_ORDER)),
            ),
            epochs=target_epochs,
            class_weight=class_weights,
            callbacks=fit_callbacks,
            verbose=2,
        )

        history_df = _history_to_frame([("cached_embeddings", history)])
        history_df.to_csv(history_csv_path, index=False)

        best_classifier = tf.keras.models.load_model(head_checkpoint_path, compile=False)
        best_model = assemble_image_classifier_model(feature_extractor, best_classifier)
        best_model.save(checkpoint_path, include_optimizer=False)
        best_model.save(model_h5_path, include_optimizer=False)

        return TrainingArtifacts(
            model_h5_path=model_h5_path,
            checkpoint_path=checkpoint_path,
            history_csv_path=history_csv_path,
            train_count=len(train_df),
            val_count=len(val_df),
            class_weights=class_weights,
            training_strategy=training_strategy,
            embedding_cache_dir=embedding_cache_dir,
        )

    if training_strategy != "cached_embeddings_v1":
        raise ValueError(
            f"Unsupported training_strategy {training_strategy!r}. "
            "Expected one of ['training1_compatible_end_to_end', "
            "'roboflow_cached_baseline_v1', 'cached_embeddings_v1']."
        )

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight as sklearn_compute_class_weight
from tensorflow.keras import callbacks, layers
from tensorflow.keras.applications import MobileNetV3Small

from .config import INPUT_SIZE, LABEL_ORDER


@dataclass(frozen=True)
class TrainingArtifacts:
    model_h5_path: Path
    checkpoint_path: Path
    history_csv_path: Path
    train_count: int
    val_count: int
    class_weights: dict[int, float]


class CsvImageSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]

        images: list[np.ndarray] = []
        labels: list[int] = []
        for row in batch_df.to_dict(orient="records"):
            image = Image.open(row["processed_image_path"]).convert("RGB")
            if image.size != INPUT_SIZE:
                image = image.resize(INPUT_SIZE, Image.BILINEAR)
            images.append(np.asarray(image, dtype=np.float32))
            labels.append(LABEL_ORDER.index(str(row["label"])))

        return (
            np.stack(images, axis=0),
            tf.keras.utils.to_categorical(labels, num_classes=len(LABEL_ORDER)),
        )

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indexes)


def build_mobilenetv3small_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 3,
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
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="meatlens_mobilenetv3small_cnn_only")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
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


def train_model(
    train_csv: Path,
    val_csv: Path,
    output_dir: Path,
    seed: int,
    epochs_head: int = 8,
    epochs_fine: int = 12,
) -> TrainingArtifacts:
    _set_training_seed(seed)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "meatlens_mobilenetv3small_pork_cnn_only_best.keras"
    model_h5_path = output_dir / "meatlens_mobilenetv3small_pork_cnn_only.keras"
    history_csv_path = output_dir / "training_history.csv"

    train_sequence = CsvImageSequence(train_df, batch_size=32, shuffle=True)
    val_sequence = CsvImageSequence(val_df, batch_size=32, shuffle=False)
    class_weights = compute_class_weights(train_df["label"])

    model = build_mobilenetv3small_model(weights="imagenet")
    fit_callbacks = _build_callbacks(checkpoint_path)
    phase_histories: list[tuple[str, tf.keras.callbacks.History]] = []

    if epochs_head > 0:
        head_history = model.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=epochs_head,
            class_weight=class_weights,
            callbacks=fit_callbacks,
            verbose=2,
        )
        phase_histories.append(("head", head_history))

    if epochs_fine > 0:
        backbone = _find_backbone(model)
        backbone.trainable = True
        fine_tune_at = max(int(len(backbone.layers) * 0.75), 1)
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        fine_history = model.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=epochs_head + epochs_fine,
            initial_epoch=epochs_head,
            class_weight=class_weights,
            callbacks=fit_callbacks,
            verbose=2,
        )
        phase_histories.append(("fine_tune", fine_history))

    history_df = _history_to_frame(phase_histories)
    history_df.to_csv(history_csv_path, index=False)

    best_model = tf.keras.models.load_model(checkpoint_path, compile=False)
    best_model.save(model_h5_path, include_optimizer=False)

    return TrainingArtifacts(
        model_h5_path=model_h5_path,
        checkpoint_path=checkpoint_path,
        history_csv_path=history_csv_path,
        train_count=len(train_df),
        val_count=len(val_df),
        class_weights=class_weights,
    )

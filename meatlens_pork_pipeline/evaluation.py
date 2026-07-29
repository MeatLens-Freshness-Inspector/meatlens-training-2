from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_mplconfig_dir = Path(tempfile.gettempdir()) / "meatlens_mplconfig"
_mplconfig_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import LABEL_ORDER


@dataclass(frozen=True)
class EvaluationSummary:
    metrics_json_path: Path
    predictions_csv_path: Path
    confusion_matrix_csv_path: Path
    normalized_confusion_matrix_csv_path: Path
    confusion_matrix_png_path: Path
    metrics: dict[str, object]
    test_count: int


def summarize_predictions(prediction_df: pd.DataFrame) -> dict[str, object]:
    y_true = prediction_df["true_label"].astype(str)
    y_pred = prediction_df["predicted_label"].astype(str)
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABEL_ORDER))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(LABEL_ORDER),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": matrix,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(LABEL_ORDER),
            output_dict=True,
            zero_division=0,
        ),
    }


def _batched_image_arrays(test_df: pd.DataFrame, batch_size: int) -> tuple[list[np.ndarray], list[pd.DataFrame]]:
    image_batches: list[np.ndarray] = []
    frame_batches: list[pd.DataFrame] = []
    for start in range(0, len(test_df), batch_size):
        batch_df = test_df.iloc[start : start + batch_size].copy()
        batch_arrays: list[np.ndarray] = []
        for row in batch_df.to_dict(orient="records"):
            image = Image.open(row["processed_image_path"]).convert("RGB")
            batch_arrays.append(np.asarray(image, dtype=np.float32))
        image_batches.append(np.stack(batch_arrays, axis=0))
        frame_batches.append(batch_df)
    return image_batches, frame_batches


def _normalized_confusion_matrix(confusion: np.ndarray) -> np.ndarray:
    row_sums = confusion.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(confusion, row_sums, where=row_sums != 0)
    normalized[row_sums.squeeze(axis=1) == 0] = 0.0
    return normalized


def _save_confusion_matrix_plot(confusion: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def evaluate_model(
    model_h5_path: Path,
    test_csv: Path,
    output_dir: Path,
    batch_size: int = 32,
) -> EvaluationSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_h5_path, compile=False)
    test_df = pd.read_csv(test_csv)

    image_batches, frame_batches = _batched_image_arrays(test_df, batch_size=batch_size)
    prediction_rows: list[dict[str, object]] = []
    for batch_arrays, batch_df in zip(image_batches, frame_batches, strict=True):
        probabilities = model.predict(batch_arrays, verbose=0)
        predicted_indices = probabilities.argmax(axis=1)
        for row, predicted_index, probability_vector in zip(
            batch_df.to_dict(orient="records"),
            predicted_indices,
            probabilities,
            strict=True,
        ):
            predicted_label = LABEL_ORDER[int(predicted_index)]
            prediction_rows.append(
                {
                    **row,
                    "true_label": str(row["label"]),
                    "predicted_label": predicted_label,
                    "confidence": float(probability_vector[int(predicted_index)]),
                    "fresh_probability": float(probability_vector[0]),
                    "not_fresh_probability": float(probability_vector[1]),
                    "spoiled_probability": float(probability_vector[2]),
                }
            )

    prediction_df = pd.DataFrame(prediction_rows)
    summary = summarize_predictions(prediction_df)
    confusion = np.asarray(summary["confusion_matrix"], dtype=int)
    normalized = _normalized_confusion_matrix(confusion)

    predictions_csv_path = output_dir / "test_predictions.csv"
    metrics_json_path = output_dir / "test_metrics.json"
    confusion_matrix_csv_path = output_dir / "confusion_matrix.csv"
    normalized_confusion_matrix_csv_path = output_dir / "confusion_matrix_normalized.csv"
    confusion_matrix_png_path = output_dir / "confusion_matrix.png"

    prediction_df.to_csv(predictions_csv_path, index=False)
    pd.DataFrame(confusion, index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(confusion_matrix_csv_path)
    pd.DataFrame(normalized, index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(
        normalized_confusion_matrix_csv_path
    )
    _save_confusion_matrix_plot(confusion, confusion_matrix_png_path)

    metrics_payload = {
        "accuracy": summary["accuracy"],
        "macro_precision": summary["macro_precision"],
        "macro_recall": summary["macro_recall"],
        "macro_f1": summary["macro_f1"],
        "classification_report": summary["classification_report"],
        "test_count": int(len(prediction_df)),
    }
    metrics_json_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return EvaluationSummary(
        metrics_json_path=metrics_json_path,
        predictions_csv_path=predictions_csv_path,
        confusion_matrix_csv_path=confusion_matrix_csv_path,
        normalized_confusion_matrix_csv_path=normalized_confusion_matrix_csv_path,
        confusion_matrix_png_path=confusion_matrix_png_path,
        metrics=metrics_payload,
        test_count=len(prediction_df),
    )

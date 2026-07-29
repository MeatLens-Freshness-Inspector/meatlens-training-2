from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

from .config import IMAGE_CROP_MODE, INPUT_SIZE, LABEL_ORDER, MODEL_INPUT_MODE


def build_onnx_metadata(
    seed: int,
    train_count: int,
    val_count: int,
    test_count: int,
    class_weights: dict[int, float],
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "model_name": "meatlens_mobilenetv3small_pork_cnn_only",
        "backbone": "MobileNetV3Small",
        "model_input_mode": MODEL_INPUT_MODE,
        "image_crop_mode": IMAGE_CROP_MODE,
        "input_shape": [INPUT_SIZE[0], INPUT_SIZE[1], 3],
        "target_size": [INPUT_SIZE[0], INPUT_SIZE[1]],
        "labels": list(LABEL_ORDER),
        "label_order": list(LABEL_ORDER),
        "label_to_index": {label: index for index, label in enumerate(LABEL_ORDER)},
        "index_to_label": {str(index): label for index, label in enumerate(LABEL_ORDER)},
        "split_mode": "random_70_15_15",
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "seed": seed,
        "class_weights": {str(index): float(weight) for index, weight in class_weights.items()},
        "test_accuracy": float(metrics["accuracy"]),
        "test_macro_precision": float(metrics["macro_precision"]),
        "test_macro_recall": float(metrics["macro_recall"]),
        "test_macro_f1": float(metrics["macro_f1"]),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def export_model_to_onnx(model_h5_path: Path, onnx_path: Path, opset: int = 13) -> Path:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_h5_path, compile=False)
    tf2onnx.convert.from_keras(
        model,
        input_signature=(
            tf.TensorSpec(
                (None, INPUT_SIZE[0], INPUT_SIZE[1], 3),
                tf.float32,
                name="image_input",
            ),
        ),
        opset=opset,
        output_path=str(onnx_path),
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    return onnx_path


def smoke_test_onnx(onnx_path: Path, sample_batch: np.ndarray) -> dict[str, object]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: sample_batch.astype(np.float32)})
    return {
        "input_name": input_name,
        "output_shapes": [list(output.shape) for output in outputs],
    }


def write_metadata_json(metadata: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path

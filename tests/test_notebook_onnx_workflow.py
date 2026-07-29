from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from tests.notebook_test_utils import execute_notebook


def test_export_and_smoke_test_onnx_workflow(tmp_path: Path) -> None:
    model_path = tmp_path / "meatlens_test_model.keras"
    onnx_path = tmp_path / "meatlens_test_model.onnx"
    metadata_path = tmp_path / "meatlens_test_model_metadata.json"
    smoke_summary_path = tmp_path / "onnx_smoke_summary.json"

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 3), name="image_input"),
            tf.keras.layers.Rescaling(scale=1.0 / 255.0),
            tf.keras.layers.Conv2D(4, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.save(model_path)

    execute_notebook(
        Path("07_export_onnx.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "MODEL_PATH": str(model_path),
            "ONNX_PATH": str(onnx_path),
            "ONNX_METADATA_PATH": str(metadata_path),
            "BASE_METADATA": {"seed": 42},
        },
        cwd=Path.cwd(),
    )

    execute_notebook(
        Path("08_inference_smoke_test.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "ONNX_PATH": str(onnx_path),
            "SMOKE_SUMMARY_PATH": str(smoke_summary_path),
        },
        cwd=Path.cwd(),
    )

    summary = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
    assert onnx_path.exists()
    assert metadata_path.exists()
    assert summary["output_shapes"] == [[1, 3]]

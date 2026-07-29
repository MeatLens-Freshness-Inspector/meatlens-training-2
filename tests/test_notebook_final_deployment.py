from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from tests.notebook_test_utils import execute_notebook


def test_train_final_deployment_model_writes_final_artifacts(tmp_path: Path) -> None:
    generated_splits_root = tmp_path / "generated_splits"
    processed_manifest_path = generated_splits_root / "processed_manifest.csv"
    processed_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    label_colors = {
        "fresh": (210, 120, 120),
        "not fresh": (170, 140, 110),
        "spoiled": (110, 150, 110),
    }
    rows: list[dict[str, object]] = []
    for sample_number in range(1, 9):
        for label, color in label_colors.items():
            image_dir = tmp_path / "data" / "processed_hsv_lab_threshold_roi_224" / f"sample {sample_number}" / label
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"sample_{sample_number}_{label.replace(' ', '_')}.jpg"
            Image.new("RGB", (224, 224), color=color).save(image_path)
            rows.append(
                {
                    "image_file_name": image_path.name,
                    "label": label,
                    "sample_number": str(sample_number),
                    "sample_id": f"sample_{sample_number}",
                    "local_image_path": str(image_path),
                }
            )

    pd.DataFrame(rows).to_csv(processed_manifest_path, index=False)

    training_outputs_root = tmp_path / "training_outputs"
    execute_notebook(
        Path("06_train_final_deployment_model.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "SKIP_GPU_CHECK": True,
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "TRAINING_OUTPUTS_ROOT": str(training_outputs_root),
            "RUN_SEEDS": [42],
            "MODEL_WEIGHTS": None,
            "BATCH_SIZE": 4,
            "EPOCHS_HEAD": 1,
            "EPOCHS_FINE": 1,
            "FINAL_VAL_SIZE": 0.25,
            "USE_TRAINING_AUGMENTATION": False,
        },
        cwd=Path.cwd(),
    )

    output_root = training_outputs_root / "mobilenetv3small_8samples_final_deployment_cnn_only"
    assert (output_root / "models" / "meatlens_final_8samples_cnn_only_mobilenetv3small.keras").exists()
    assert (output_root / "final_validation_predictions.csv").exists()
    assert (output_root / "final_training_history.csv").exists()
    assert (output_root / "deployment_metadata.json").exists()

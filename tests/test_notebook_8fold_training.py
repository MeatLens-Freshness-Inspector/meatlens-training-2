from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from tests.notebook_test_utils import execute_notebook


def test_train_8fold_mobilenetv3small_writes_fold_artifacts(tmp_path: Path) -> None:
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

    execute_notebook(
        Path("03_build_cross_rotation_splits.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "GENERATED_SPLITS_ROOT": str(generated_splits_root),
        },
        cwd=Path.cwd(),
    )

    training_outputs_root = tmp_path / "training_outputs"
    execute_notebook(
        Path("04_train_8fold_mobilenetv3small.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "SKIP_GPU_CHECK": True,
            "GENERATED_SPLITS_ROOT": str(generated_splits_root),
            "TRAINING_OUTPUTS_ROOT": str(training_outputs_root),
            "RUN_SEEDS": [42],
            "SELECT_FOLDS": ["fold1"],
            "MODEL_WEIGHTS": None,
            "BATCH_SIZE": 4,
            "EPOCHS_HEAD": 1,
            "EPOCHS_FINE": 1,
            "USE_TRAINING_AUGMENTATION": False,
        },
        cwd=Path.cwd(),
    )

    output_root = training_outputs_root / "mobilenetv3small_8fold_processed_roi_cnn_only"
    metrics_path = output_root / "processed_roi8_cnn_only_seed_metrics.csv"
    metrics_df = pd.read_csv(metrics_path)

    assert metrics_df.loc[0, "fold"] == "fold1"
    assert int(metrics_df.loc[0, "seed"]) == 42
    assert {"accuracy", "macro_precision", "macro_recall", "macro_f1"}.issubset(metrics_df.columns)
    assert (output_root / "models" / "processed_roi8_cnn_only_fold1_seed42.keras").exists()
    assert (output_root / "predictions" / "processed_roi8_cnn_only_fold1_seed42_test_predictions.csv").exists()
    assert (output_root / "figures" / "processed_roi8_cnn_only_fold1_seed42_confusion_matrix.png").exists()

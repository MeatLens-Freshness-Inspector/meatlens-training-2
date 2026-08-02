from __future__ import annotations

import json
from pathlib import Path


def test_shared_setup_notebook_exposes_core_contract() -> None:
    notebook = json.loads(Path("00_shared_setup.ipynb").read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "LABEL_ORDER = ['fresh', 'not fresh', 'spoiled']" in code_source
    assert "RUN_SEEDS = [42, 123, 2026]" in code_source
    assert "INPUT_SHAPE = (224, 224, 3)" in code_source
    assert "REQUIRED_TRAINING_GPU_SUBSTRING = 'RTX 4050'" in code_source
    assert "tf.config.list_physical_devices('GPU')" in code_source
    assert "TensorFlow does not currently see any GPU devices." in code_source
    assert "def override(name: str, default: object) -> object:" in code_source
    assert "def inspect_default_processed_dataset(" in code_source
    assert "DEFAULT_PROCESSED_DATASET_SUMMARY = inspect_default_processed_dataset()" in code_source
    assert "DEFAULT_PROCESSED_DATASET_READY = bool(DEFAULT_PROCESSED_DATASET_SUMMARY['ready'])" in code_source
    assert "tf.config.experimental.enable_op_determinism()" in code_source
    assert "def enforce_training_gpu(required_name_substring: str = REQUIRED_TRAINING_GPU_SUBSTRING) -> list[str]:" in code_source


def test_shared_setup_notebook_exposes_procedure_improvement_contract() -> None:
    notebook = json.loads(Path("00_shared_setup.ipynb").read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "INPUT_MODE = str(override('INPUT_MODE', 'processed_hsv_lab_threshold_roi_224'))" in code_source
    assert "RAW_CENTER_CROP_ROOT = Path(str(override('RAW_CENTER_CROP_ROOT', DATA_ROOT / 'raw_center_crop_224')))" in code_source
    assert "AUGMENTATION_PRESET = str(override('AUGMENTATION_PRESET', 'geometry_only_v1'))" in code_source
    assert "SEVERE_ERROR_LABEL_PAIRS = [('fresh', 'spoiled'), ('spoiled', 'fresh')]" in code_source
    assert "HEAD_LR = float(override('HEAD_LR', 1e-4))" in code_source
    assert "TRAINING_STRATEGY = str(override('TRAINING_STRATEGY', 'cached_embeddings_sgd_v1'))" in code_source
    assert "def resolve_input_root(input_mode: str) -> Path:" in code_source
    assert "def resolve_fine_tune_fraction(value: object) -> float:" in code_source
    assert "def build_sample_heldout_validation_split(" in code_source


def test_test_suite_architecture_combines_module_and_feature_grouping() -> None:
    expected_directories = [
        Path("tests/support"),
        Path("tests/unit/augmentation"),
        Path("tests/unit/cli"),
        Path("tests/unit/config"),
        Path("tests/unit/evaluation"),
        Path("tests/unit/image_ops"),
        Path("tests/unit/manifest"),
        Path("tests/unit/modeling"),
        Path("tests/unit/onnx_export"),
        Path("tests/unit/splits"),
        Path("tests/unit/training"),
        Path("tests/unit/windows_tf_bootstrap"),
        Path("tests/integration/data_pipeline"),
        Path("tests/integration/training_pipeline"),
        Path("tests/integration/export_pipeline"),
        Path("tests/integration/runtime"),
    ]

    for directory in expected_directories:
        assert directory.is_dir(), f"Expected test directory {directory} to exist"

    assert (Path("tests/support/notebook_test_utils.py")).exists()

# MobileNetV3-Small Procedure Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the notebook-first `MobileNetV3Small` pork-freshness pipeline with stronger training procedure controls, a less-processed input branch, sample-aware deployment validation, and better procedure-selection reports while preserving ONNX export.

**Architecture:** Keep the current `00` to `08` notebook suite as the only operational interface and extend it in place. The main changes are shared experiment knobs in `00_shared_setup.ipynb`, a second preprocessing branch in `02_roi_preprocessing.ipynb`, explicit conservative augmentation and richer metrics in `04_train_8fold_mobilenetv3small.ipynb`, stronger aggregate reports in `05_regenerate_metrics_and_reports.ipynb`, and sample-aware final validation plus input-mode metadata in `06` through `08`.

**Tech Stack:** Python 3.10, Jupyter notebooks, TensorFlow 2.10.1 on native Windows RTX 4050, pandas, numpy, pillow, scikit-learn, matplotlib, seaborn, tf2onnx, onnx, onnxruntime, pytest

## Global Constraints

- Keep only `MobileNetV3Small`.
- Keep model mode `cnn_only`.
- Keep output classes exactly `fresh`, `not fresh`, and `spoiled`.
- Keep required primary metrics: `accuracy`, macro `precision`, macro `recall`, macro `F1`, and confusion matrix.
- Preserve the official `8-fold` sample-held-out cross-rotation workflow for all thesis reporting.
- Do not weaken sample-level separation with random image-level fold splitting.
- Keep ONNX export and ONNX Runtime smoke testing intact.
- Support two input modes in this pass:
  - `processed_hsv_lab_threshold_roi_224`
  - `raw_center_crop_224`
- Keep the processed ROI branch as the default because the repo already ships with `data/processed_hsv_lab_threshold_roi_224`.
- Use conservative training-only augmentation with no aggressive hue or saturation shifts.
- Support explicit fine-tuning settings `0.0`, `0.25`, and `1.0`.
- Report procedure-level quality using mean macro `F1`, standard deviation, worst-fold macro `F1`, and severe-error rate.
- Define severe-error rate as only:
  - `fresh -> spoiled`
  - `spoiled -> fresh`
- Final deployment validation must be sample-aware, not random-image only.
- Do not add ordinal loss, cost-sensitive loss, hard-example mining, calibration, temporal smoothing, or new backbones in this pass.

## File Map

**Modify**

- `00_shared_setup.ipynb`
  - add shared experiment knobs and sample-aware split helpers
- `02_roi_preprocessing.ipynb`
  - add `raw_center_crop_224` generation alongside the existing processed ROI path
- `04_train_8fold_mobilenetv3small.ipynb`
  - tighten augmentation, add input-mode and fine-tune-mode support, and save severe-error metrics
- `05_regenerate_metrics_and_reports.ipynb`
  - add worst-fold, severe-error, and procedure summary reports
- `06_train_final_deployment_model.ipynb`
  - replace random image-level validation with sample-aware validation
- `07_export_onnx.ipynb`
  - carry input-mode metadata into ONNX metadata output
- `08_inference_smoke_test.ipynb`
  - verify the exported artifact still matches the expected input and output contract
- `docs/training-pipeline-usage.md`
  - document the new experiment knobs and the raw-center-crop prerequisite
- `tests/test_notebook_suite_structure.py`
  - assert the shared setup notebook exposes the new controls
- `tests/test_notebook_dataset_flow.py`
  - assert `02` can generate a less-processed branch
- `tests/test_notebook_8fold_training.py`
  - assert the training notebook emits the new metrics and respects input mode
- `tests/test_notebook_reports.py`
  - assert the report notebook emits worst-fold and severe-error summaries
- `tests/test_notebook_final_deployment.py`
  - assert final validation splits on `sample_id`
- `tests/test_notebook_onnx_workflow.py`
  - assert ONNX metadata includes the chosen input mode
- `tests/test_notebook_docs.py`
  - assert docs cover the new experiment surface

**Do Not Touch**

- `docs/windows_instructions.md`
- `docs/MeatLens_Training_Procedure_Recommendations_COMPLETE.md`
- `tmp/`

---

### Task 1: Extend Shared Experiment Controls In `00_shared_setup.ipynb`

**Files:**
- Modify: `00_shared_setup.ipynb`
- Modify: `tests/test_notebook_suite_structure.py`
- Test: `tests/test_notebook_suite_structure.py`

**Interfaces:**
- Consumes:
  - `override(name: str, default: object) -> object`
  - `ensure_dir(path: Path) -> Path`
- Produces:
  - `INPUT_MODE: str`
  - `RAW_CENTER_CROP_ROOT: Path`
  - `INPUT_MODE_ROOTS: dict[str, Path]`
  - `AUGMENTATION_PRESET: str`
  - `SEVERE_ERROR_LABEL_PAIRS: list[tuple[str, str]]`
  - `resolve_input_root(input_mode: str) -> Path`
  - `resolve_fine_tune_fraction(value: object) -> float`
  - `build_sample_heldout_validation_split(processed_df: pd.DataFrame, val_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_suite_structure.py
from __future__ import annotations

import json
from pathlib import Path


def test_shared_setup_notebook_exposes_procedure_improvement_contract() -> None:
    notebook = json.loads(Path("00_shared_setup.ipynb").read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "INPUT_MODE = str(override('INPUT_MODE', 'processed_hsv_lab_threshold_roi_224'))" in code_source
    assert "RAW_CENTER_CROP_ROOT = Path(str(override('RAW_CENTER_CROP_ROOT', DATA_ROOT / 'raw_center_crop_224')))" in code_source
    assert "AUGMENTATION_PRESET = str(override('AUGMENTATION_PRESET', 'conservative_v1'))" in code_source
    assert "SEVERE_ERROR_LABEL_PAIRS = [('fresh', 'spoiled'), ('spoiled', 'fresh')]" in code_source
    assert "def resolve_input_root(input_mode: str) -> Path:" in code_source
    assert "def resolve_fine_tune_fraction(value: object) -> float:" in code_source
    assert "def build_sample_heldout_validation_split(" in code_source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_suite_structure.py::test_shared_setup_notebook_exposes_procedure_improvement_contract -v`

Expected: `FAIL` because `00_shared_setup.ipynb` does not yet expose the new input-mode, severe-error, or sample-held-out split helpers.

- [ ] **Step 3: Write minimal implementation**

```python
# 00_shared_setup.ipynb
INPUT_MODE = str(override('INPUT_MODE', 'processed_hsv_lab_threshold_roi_224'))
RAW_CENTER_CROP_ROOT = Path(str(override('RAW_CENTER_CROP_ROOT', DATA_ROOT / 'raw_center_crop_224')))
INPUT_MODE_ROOTS = {
    'processed_hsv_lab_threshold_roi_224': PROCESSED_ROI_ROOT,
    'raw_center_crop_224': RAW_CENTER_CROP_ROOT,
}
AUGMENTATION_PRESET = str(override('AUGMENTATION_PRESET', 'conservative_v1'))
SEVERE_ERROR_LABEL_PAIRS = [('fresh', 'spoiled'), ('spoiled', 'fresh')]


def resolve_input_root(input_mode: str) -> Path:
    normalized = str(input_mode).strip()
    if normalized not in INPUT_MODE_ROOTS:
        raise ValueError(
            f'Unsupported input_mode {normalized!r}. Expected one of {sorted(INPUT_MODE_ROOTS)}'
        )
    return INPUT_MODE_ROOTS[normalized]


def resolve_fine_tune_fraction(value: object) -> float:
    fraction = float(value)
    allowed = {0.0, 0.25, 1.0}
    if fraction not in allowed:
        raise ValueError(f'Fine-tune fraction must be one of {sorted(allowed)}. Got {fraction}.')
    return fraction


def build_sample_heldout_validation_split(
    processed_df: pd.DataFrame,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if 'sample_id' not in processed_df.columns:
        raise ValueError('Processed manifest must contain sample_id for sample-held-out validation.')

    grouped = (
        processed_df[['sample_id', 'label']]
        .drop_duplicates()
        .groupby('sample_id')['label']
        .agg(lambda values: tuple(sorted(values)))
        .reset_index()
    )
    if grouped.empty:
        raise ValueError('Cannot build a validation split from an empty processed manifest.')

    candidate_count = max(1, int(round(len(grouped) * float(val_size))))
    shuffled = grouped.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    selected_sample_ids = shuffled.head(candidate_count)['sample_id'].astype(str).tolist()

    train_df = processed_df[~processed_df['sample_id'].astype(str).isin(selected_sample_ids)].copy()
    val_df = processed_df[processed_df['sample_id'].astype(str).isin(selected_sample_ids)].copy()

    if train_df.empty or val_df.empty:
        raise ValueError('Sample-held-out deployment split produced an empty train or validation partition.')

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_suite_structure.py::test_shared_setup_notebook_exposes_procedure_improvement_contract -v`

Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add 00_shared_setup.ipynb tests/test_notebook_suite_structure.py
git commit -m "feat: add shared procedure experiment controls"
```

### Task 2: Add A Less-Processed `raw_center_crop_224` Branch In `02_roi_preprocessing.ipynb`

**Files:**
- Modify: `02_roi_preprocessing.ipynb`
- Modify: `tests/test_notebook_dataset_flow.py`
- Test: `tests/test_notebook_dataset_flow.py`

**Interfaces:**
- Consumes:
  - `TARGET_SIZE: tuple[int, int]`
  - `INPUT_MODE: str`
  - `resolve_input_root(input_mode: str) -> Path`
  - `ensure_dir(path: Path) -> Path`
- Produces:
  - `preprocess_raw_center_crop_image(path: Path) -> tuple[np.ndarray, dict[str, object]]`
  - `preprocess_audited_manifest(audited_df: pd.DataFrame, output_root: Path, input_mode: str, background_mode: str = 'gray') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_dataset_flow.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from tests.notebook_test_utils import execute_notebook


def test_roi_preprocessing_can_generate_raw_center_crop_branch(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_images"
    raw_root.mkdir(parents=True, exist_ok=True)
    source_image = raw_root / "raw_001.jpg"
    Image.new("RGB", (480, 360), color=(190, 110, 110)).save(source_image)

    audited_manifest_path = tmp_path / "generated_splits" / "audited_manifest.csv"
    audited_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "image_name": source_image.name,
                "image_file_name": source_image.name,
                "label": "fresh",
                "sample_number": "1",
                "sample_id": "sample_1",
                "local_image_path": str(source_image),
                "source_manifest_type": "raw_manifest",
            }
        ]
    ).to_csv(audited_manifest_path, index=False)

    raw_center_crop_root = tmp_path / "data" / "raw_center_crop_224"
    processed_manifest_path = tmp_path / "generated_splits" / "processed_manifest.csv"
    preprocessing_summary_path = raw_center_crop_root / "preprocessing_summary.csv"
    preprocessing_failures_path = raw_center_crop_root / "preprocessing_failures.csv"

    execute_notebook(
        Path("02_roi_preprocessing.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "FORCE_REPROCESS": True,
            "INPUT_MODE": "raw_center_crop_224",
            "AUDITED_MANIFEST_PATH": str(audited_manifest_path),
            "RAW_CENTER_CROP_ROOT": str(raw_center_crop_root),
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "PREPROCESSING_SUMMARY_PATH": str(preprocessing_summary_path),
            "PREPROCESSING_FAILURES_PATH": str(preprocessing_failures_path),
            "GENERATED_SPLITS_ROOT": str(tmp_path / "generated_splits"),
        },
        cwd=Path.cwd(),
    )

    processed_df = pd.read_csv(processed_manifest_path)
    output_image = Path(processed_df.loc[0, "local_image_path"])

    assert processed_df.loc[0, "input_mode"] == "raw_center_crop_224"
    assert output_image.exists()
    assert Image.open(output_image).size == (224, 224)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_dataset_flow.py::test_roi_preprocessing_can_generate_raw_center_crop_branch -v`

Expected: `FAIL` because `02_roi_preprocessing.ipynb` only supports the processed ROI branch.

- [ ] **Step 3: Write minimal implementation**

```python
# 02_roi_preprocessing.ipynb
def preprocess_raw_center_crop_image(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    image = Image.open(path).convert('RGB')
    if image.size != TARGET_SIZE:
        image = image.resize(TARGET_SIZE, Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8), {
        'segmentation_failed': False,
        'mask_area_ratio': '',
        'center_overlap_ratio': '',
        'number_of_components': '',
        'touches_border': '',
    }


def preprocess_audited_manifest(
    audited_df: pd.DataFrame,
    output_root: Path,
    input_mode: str,
    background_mode: str = 'gray',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for row in audited_df.to_dict(orient='records'):
        source_path = Path(str(row['local_image_path']))
        output_path = build_processed_output_path(pd.Series(row), output_root)
        ensure_dir(output_path.parent)

        try:
            if input_mode == 'processed_hsv_lab_threshold_roi_224':
                output_uint8, metadata = preprocess_roi_image(source_path, background_mode=background_mode)
            elif input_mode == 'raw_center_crop_224':
                output_uint8, metadata = preprocess_raw_center_crop_image(source_path)
            else:
                raise ValueError(f'Unsupported input_mode: {input_mode}')

            Image.fromarray(output_uint8).save(output_path)
            processed_record = dict(row)
            processed_record['local_image_path'] = str(output_path)
            processed_record['processed_output_file'] = str(output_path)
            processed_record['input_mode'] = input_mode
            processed_record.update(metadata)
            records.append(processed_record)

            summary_rows.append(
                {
                    'image_file_name': row['image_file_name'],
                    'sample_number': row.get('sample_number', ''),
                    'sample_id': row.get('sample_id', ''),
                    'label': row['label'],
                    'input_mode': input_mode,
                    'processed_output_file': str(output_path),
                    **metadata,
                }
            )
        except Exception as exc:
            failure_rows.append({'image_file_name': row.get('image_file_name', ''), 'error': str(exc)})

    return pd.DataFrame(records), pd.DataFrame(summary_rows), pd.DataFrame(failure_rows)


OUTPUT_ROOT = resolve_input_root(INPUT_MODE)
PREPROCESSING_SUMMARY_PATH = Path(str(override('PREPROCESSING_SUMMARY_PATH', OUTPUT_ROOT / 'preprocessing_summary.csv')))
PREPROCESSING_FAILURES_PATH = Path(str(override('PREPROCESSING_FAILURES_PATH', OUTPUT_ROOT / 'preprocessing_failures.csv')))

if use_existing_processed and INPUT_MODE == 'processed_hsv_lab_threshold_roi_224':
    processed_df, summary_df, failures_df = normalize_existing_processed_manifest(audited_df)
    processed_df['input_mode'] = 'processed_hsv_lab_threshold_roi_224'
else:
    processed_df, summary_df, failures_df = preprocess_audited_manifest(
        audited_df,
        output_root=OUTPUT_ROOT,
        input_mode=INPUT_MODE,
        background_mode=BACKGROUND_MODE,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_dataset_flow.py::test_roi_preprocessing_can_generate_raw_center_crop_branch -v`

Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add 02_roi_preprocessing.ipynb tests/test_notebook_dataset_flow.py
git commit -m "feat: add raw center crop preprocessing branch"
```

### Task 3: Upgrade `04_train_8fold_mobilenetv3small.ipynb` For Explicit Augmentation, Input Mode, Fine-Tune Mode, And Severe Errors

**Files:**
- Modify: `04_train_8fold_mobilenetv3small.ipynb`
- Modify: `tests/test_notebook_8fold_training.py`
- Test: `tests/test_notebook_8fold_training.py`

**Interfaces:**
- Consumes:
  - `INPUT_MODE: str`
  - `AUGMENTATION_PRESET: str`
  - `resolve_fine_tune_fraction(value: object) -> float`
  - `SEVERE_ERROR_LABEL_PAIRS: list[tuple[str, str]]`
- Produces:
  - `build_training_augmentation(preset: str = AUGMENTATION_PRESET) -> tf.keras.Sequential`
  - `compute_severe_error_rate(y_true: pd.Series, y_pred: pd.Series) -> float`
  - `summarize_fold_metrics(prediction_df: pd.DataFrame) -> dict[str, float]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_8fold_training.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from tests.notebook_test_utils import execute_notebook


def test_train_8fold_mobilenetv3small_writes_procedure_metrics(tmp_path: Path) -> None:
    generated_splits_root = tmp_path / "generated_splits"
    processed_manifest_path = generated_splits_root / "processed_manifest.csv"
    processed_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for sample_number in range(1, 9):
        for label, color in {
            "fresh": (210, 120, 120),
            "not fresh": (170, 140, 110),
            "spoiled": (110, 150, 110),
        }.items():
            image_dir = tmp_path / "data" / "raw_center_crop_224" / f"sample {sample_number}" / label
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
                    "input_mode": "raw_center_crop_224",
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
            "INPUT_MODE": "raw_center_crop_224",
            "GENERATED_SPLITS_ROOT": str(generated_splits_root),
            "TRAINING_OUTPUTS_ROOT": str(training_outputs_root),
            "RUN_SEEDS": [42],
            "SELECT_FOLDS": ["fold1"],
            "MODEL_WEIGHTS": None,
            "BATCH_SIZE": 4,
            "EPOCHS_HEAD": 1,
            "EPOCHS_FINE": 0,
            "FINE_TUNE_FRACTION": 0.0,
            "USE_TRAINING_AUGMENTATION": True,
        },
        cwd=Path.cwd(),
    )

    output_root = training_outputs_root / "mobilenetv3small_8fold_processed_roi_cnn_only"
    metrics_df = pd.read_csv(output_root / "processed_roi8_cnn_only_seed_metrics.csv")

    assert metrics_df.loc[0, "input_mode"] == "raw_center_crop_224"
    assert float(metrics_df.loc[0, "fine_tune_fraction"]) == 0.0
    assert "severe_error_rate" in metrics_df.columns
    assert "macro_f1" in metrics_df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_8fold_training.py::test_train_8fold_mobilenetv3small_writes_procedure_metrics -v`

Expected: `FAIL` because the seed metrics CSV does not yet save `input_mode`, `fine_tune_fraction`, or `severe_error_rate`.

- [ ] **Step 3: Write minimal implementation**

```python
# 04_train_8fold_mobilenetv3small.ipynb
def build_training_augmentation(preset: str = AUGMENTATION_PRESET) -> tf.keras.Sequential:
    if preset != 'conservative_v1':
        raise ValueError(f'Unsupported augmentation preset: {preset}')
    return tf.keras.Sequential(
        [
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.04),
            layers.RandomZoom(height_factor=(-0.08, 0.08), width_factor=(-0.08, 0.08)),
            layers.RandomTranslation(height_factor=0.04, width_factor=0.04),
            layers.RandomBrightness(factor=0.08),
            layers.RandomContrast(0.10),
        ],
        name='training_augmentation',
    )


TRAINING_AUGMENTATION = build_training_augmentation()


class ImageOnlySequence(tf.keras.utils.Sequence):
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        ...
        batch = np.stack(images, axis=0).astype(np.float32)
        if self.use_augmentation:
            batch = TRAINING_AUGMENTATION(batch, training=True).numpy()
        batch = preprocess_input(batch)
        return batch, tf.keras.utils.to_categorical(labels, num_classes=len(LABEL_ORDER))


def compute_severe_error_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    total = len(y_true)
    if total == 0:
        return 0.0
    severe = 0
    for true_label, pred_label in zip(y_true.astype(str), y_pred.astype(str), strict=True):
        if (true_label, pred_label) in SEVERE_ERROR_LABEL_PAIRS:
            severe += 1
    return float(severe / total)


def summarize_fold_metrics(prediction_df: pd.DataFrame) -> dict[str, float]:
    y_true = prediction_df['true_label'].astype(str)
    y_pred = prediction_df['predicted_label'].astype(str)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average='macro',
        zero_division=0,
    )
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'severe_error_rate': compute_severe_error_rate(y_true, y_pred),
    }


FINE_TUNE_FRACTION = resolve_fine_tune_fraction(override('FINE_TUNE_FRACTION', FINE_TUNE_FRACTION))


summary = {
    'fold': fold_name,
    'seed': seed,
    'input_mode': INPUT_MODE,
    'fine_tune_fraction': float(FINE_TUNE_FRACTION),
    'accuracy': metrics['accuracy'],
    'macro_precision': metrics['macro_precision'],
    'macro_recall': metrics['macro_recall'],
    'macro_f1': metrics['macro_f1'],
    'severe_error_rate': metrics['severe_error_rate'],
    ...
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_8fold_training.py::test_train_8fold_mobilenetv3small_writes_procedure_metrics -v`

Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add 04_train_8fold_mobilenetv3small.ipynb tests/test_notebook_8fold_training.py
git commit -m "feat: enrich 8-fold MobileNetV3Small procedure metrics"
```

### Task 4: Extend `05_regenerate_metrics_and_reports.ipynb` With Worst-Fold, Severe-Error, And Procedure Summaries

**Files:**
- Modify: `05_regenerate_metrics_and_reports.ipynb`
- Modify: `tests/test_notebook_reports.py`
- Test: `tests/test_notebook_reports.py`

**Interfaces:**
- Consumes:
  - `compute_severe_error_rate(y_true: pd.Series, y_pred: pd.Series) -> float`
  - `SEVERE_ERROR_LABEL_PAIRS: list[tuple[str, str]]`
- Produces:
  - `processed_roi8_cnn_only_fold_summary.csv`
  - `processed_roi8_cnn_only_worst_fold_summary.csv`
  - `processed_roi8_cnn_only_severe_error_summary.csv`
  - `processed_roi8_cnn_only_procedure_summary.csv`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_reports.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.notebook_test_utils import execute_notebook


def test_regenerate_reports_writes_procedure_summary_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "training_outputs" / "mobilenetv3small_8fold_processed_roi_cnn_only"
    predictions_root = output_root / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    prediction_path = predictions_root / "processed_roi8_cnn_only_fold1_seed42_test_predictions.csv"
    pd.DataFrame(
        [
            {"true_label": "fresh", "predicted_label": "spoiled"},
            {"true_label": "not fresh", "predicted_label": "not fresh"},
            {"true_label": "spoiled", "predicted_label": "fresh"},
        ]
    ).to_csv(prediction_path, index=False)

    metrics_path = output_root / "processed_roi8_cnn_only_seed_metrics.csv"
    pd.DataFrame(
        [
            {
                "fold": "fold1",
                "seed": 42,
                "predictions_path": str(prediction_path),
                "input_mode": "processed_hsv_lab_threshold_roi_224",
                "fine_tune_fraction": 0.25,
            }
        ]
    ).to_csv(metrics_path, index=False)

    execute_notebook(
        Path("05_regenerate_metrics_and_reports.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "EIGHTFOLD_OUTPUT_ROOT": str(output_root),
            "SEED_METRICS_PATH": str(metrics_path),
        },
        cwd=Path.cwd(),
    )

    assert (output_root / "processed_roi8_cnn_only_worst_fold_summary.csv").exists()
    assert (output_root / "processed_roi8_cnn_only_severe_error_summary.csv").exists()
    assert (output_root / "processed_roi8_cnn_only_procedure_summary.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_reports.py::test_regenerate_reports_writes_procedure_summary_outputs -v`

Expected: `FAIL` because the report notebook does not yet emit the new summary CSVs.

- [ ] **Step 3: Write minimal implementation**

```python
# 05_regenerate_metrics_and_reports.ipynb
def summarize_prediction_frame(prediction_df: pd.DataFrame) -> dict[str, object]:
    ...
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'severe_error_rate': compute_severe_error_rate(y_true, y_pred),
        'confusion_matrix': confusion,
    }


def regenerate_official_reports(seed_metrics_path: Path, output_root: Path) -> dict[str, Path]:
    ...
    worst_fold_df = (
        fold_summary_df.sort_values(['macro_f1', 'severe_error_rate', 'fold', 'seed'])
        .head(1)
        .reset_index(drop=True)
    )
    severe_error_df = fold_summary_df[['fold', 'seed', 'severe_error_rate']].copy()
    procedure_summary_df = pd.DataFrame(
        [
            {
                'input_mode': str(metrics_df['input_mode'].iloc[0]) if 'input_mode' in metrics_df.columns else INPUT_MODE,
                'fine_tune_fraction': float(metrics_df['fine_tune_fraction'].iloc[0]) if 'fine_tune_fraction' in metrics_df.columns else FINE_TUNE_FRACTION,
                'macro_f1_mean': float(fold_summary_df['macro_f1'].mean()),
                'macro_f1_std': float(fold_summary_df['macro_f1'].std(ddof=0)),
                'worst_fold_macro_f1': float(worst_fold_df.loc[0, 'macro_f1']),
                'severe_error_rate_mean': float(fold_summary_df['severe_error_rate'].mean()),
            }
        ]
    )

    worst_fold_path = output_root / 'processed_roi8_cnn_only_worst_fold_summary.csv'
    severe_error_path = output_root / 'processed_roi8_cnn_only_severe_error_summary.csv'
    procedure_summary_path = output_root / 'processed_roi8_cnn_only_procedure_summary.csv'

    worst_fold_df.to_csv(worst_fold_path, index=False)
    severe_error_df.to_csv(severe_error_path, index=False)
    procedure_summary_df.to_csv(procedure_summary_path, index=False)

    return {
        ...,
        'worst_fold_summary': worst_fold_path,
        'severe_error_summary': severe_error_path,
        'procedure_summary': procedure_summary_path,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_reports.py::test_regenerate_reports_writes_procedure_summary_outputs -v`

Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add 05_regenerate_metrics_and_reports.ipynb tests/test_notebook_reports.py
git commit -m "feat: add worst-fold and severe-error notebook reports"
```

### Task 5: Make `06` Through `08` Deployment-Aware Of Sample-Held-Out Validation And Input-Mode Metadata

**Files:**
- Modify: `06_train_final_deployment_model.ipynb`
- Modify: `07_export_onnx.ipynb`
- Modify: `08_inference_smoke_test.ipynb`
- Modify: `tests/test_notebook_final_deployment.py`
- Modify: `tests/test_notebook_onnx_workflow.py`
- Test: `tests/test_notebook_final_deployment.py`
- Test: `tests/test_notebook_onnx_workflow.py`

**Interfaces:**
- Consumes:
  - `build_sample_heldout_validation_split(...) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `INPUT_MODE: str`
  - `resolve_fine_tune_fraction(value: object) -> float`
- Produces:
  - `deployment_metadata.json` with `input_mode` and `fine_tune_fraction`
  - ONNX metadata JSON with `input_mode`
  - smoke-test summary JSON with the exported model path and output shapes

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_notebook_final_deployment.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from tests.notebook_test_utils import execute_notebook


def test_final_deployment_uses_sample_heldout_validation(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed_hsv_lab_threshold_roi_224"
    rows: list[dict[str, object]] = []
    for sample_idx in range(1, 9):
        for label, color in {
            "fresh": (210, 120, 120),
            "not fresh": (170, 140, 110),
            "spoiled": (110, 150, 110),
        }.items():
            image_dir = processed_root / f"sample {sample_idx}" / label
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"sample_{sample_idx}_{label.replace(' ', '_')}.jpg"
            Image.new("RGB", (224, 224), color=color).save(image_path)
            rows.append(
                {
                    "image_file_name": image_path.name,
                    "label": label,
                    "sample_number": str(sample_idx),
                    "sample_id": f"sample_{sample_idx}",
                    "local_image_path": str(image_path),
                    "input_mode": "processed_hsv_lab_threshold_roi_224",
                }
            )

    processed_manifest_path = tmp_path / "generated_splits" / "processed_manifest.csv"
    processed_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed_manifest_path, index=False)

    output_root = tmp_path / "training_outputs" / "mobilenetv3small_8samples_final_deployment_cnn_only"
    execute_notebook(
        Path("06_train_final_deployment_model.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "SKIP_GPU_CHECK": True,
            "PROCESSED_MANIFEST_PATH": str(processed_manifest_path),
            "FINAL_DEPLOYMENT_OUTPUT_ROOT": str(output_root),
            "MODEL_WEIGHTS": None,
            "BATCH_SIZE": 4,
            "EPOCHS_HEAD": 1,
            "EPOCHS_FINE": 0,
            "FINAL_VAL_SIZE": 0.25,
        },
        cwd=Path.cwd(),
    )

    prediction_df = pd.read_csv(output_root / "final_validation_predictions.csv")
    metadata = json.loads((output_root / "deployment_metadata.json").read_text(encoding="utf-8"))

    assert prediction_df["sample_id"].nunique() >= 1
    assert metadata["input_mode"] == "processed_hsv_lab_threshold_roi_224"
    assert metadata["fine_tune_fraction"] == 0.25
```

```python
# tests/test_notebook_onnx_workflow.py
from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from tests.notebook_test_utils import execute_notebook


def test_notebook_onnx_workflow_preserves_input_mode_metadata(tmp_path: Path) -> None:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(224, 224, 3), name="image_input"),
            tf.keras.layers.Rescaling(scale=1.0 / 255.0),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model_path = tmp_path / "final_model.keras"
    model.save(model_path, include_optimizer=False)

    metadata_path = tmp_path / "deployment_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "meatlens_final_8samples_cnn_only_mobilenetv3small",
                "input_mode": "raw_center_crop_224",
                "labels": ["fresh", "not fresh", "spoiled"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    onnx_path = tmp_path / "model.onnx"
    onnx_metadata_path = tmp_path / "model.metadata.json"

    execute_notebook(
        Path("07_export_onnx.ipynb"),
        overrides={
            "NOTEBOOK_TEST_MODE": True,
            "MODEL_PATH": str(model_path),
            "BASE_METADATA_PATH": str(metadata_path),
            "ONNX_PATH": str(onnx_path),
            "ONNX_METADATA_PATH": str(onnx_metadata_path),
        },
        cwd=Path.cwd(),
    )

    onnx_metadata = json.loads(onnx_metadata_path.read_text(encoding="utf-8"))
    assert onnx_metadata["input_mode"] == "raw_center_crop_224"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_final_deployment.py::test_final_deployment_uses_sample_heldout_validation tests\test_notebook_onnx_workflow.py::test_notebook_onnx_workflow_preserves_input_mode_metadata -v`

Expected: `FAIL` because final deployment still uses random image-level validation and the ONNX metadata path does not yet guarantee `input_mode`.

- [ ] **Step 3: Write minimal implementation**

```python
# 06_train_final_deployment_model.ipynb
def split_final_deployment_df(
    processed_df: pd.DataFrame,
    random_state: int,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_sample_heldout_validation_split(
        processed_df,
        val_size=val_size,
        random_state=random_state,
    )


metadata = {
    'model_name': 'meatlens_final_8samples_cnn_only_mobilenetv3small',
    'backbone': 'MobileNetV3Small',
    'model_input_mode': 'cnn_only',
    'image_crop_mode': INPUT_MODE,
    'input_mode': INPUT_MODE,
    'fine_tune_fraction': float(FINE_TUNE_FRACTION),
    ...
}
```

```python
# 07_export_onnx.ipynb
BASE_METADATA_PATH = Path(str(override('BASE_METADATA_PATH', TRAINING_OUTPUTS_ROOT / 'mobilenetv3small_8samples_final_deployment_cnn_only' / 'deployment_metadata.json')))
BASE_METADATA = json.loads(BASE_METADATA_PATH.read_text(encoding='utf-8'))


def build_onnx_metadata(model_path: Path, base_metadata: dict[str, object]) -> dict[str, object]:
    return {
        **base_metadata,
        'onnx_model_path': str(model_path),
        'input_mode': base_metadata.get('input_mode', INPUT_MODE),
        'labels': base_metadata.get('labels', LABEL_ORDER),
        'input_shape': list(INPUT_SHAPE),
    }
```

```python
# 08_inference_smoke_test.ipynb
summary = smoke_test_onnx(ONNX_PATH, batch)
summary['onnx_path'] = str(ONNX_PATH)
summary['expected_input_shape'] = list(INPUT_SHAPE)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_final_deployment.py::test_final_deployment_uses_sample_heldout_validation tests\test_notebook_onnx_workflow.py::test_notebook_onnx_workflow_preserves_input_mode_metadata -v`

Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add 06_train_final_deployment_model.ipynb 07_export_onnx.ipynb 08_inference_smoke_test.ipynb tests/test_notebook_final_deployment.py tests/test_notebook_onnx_workflow.py
git commit -m "feat: make deployment and onnx metadata procedure-aware"
```

### Task 6: Update Docs And Run The Focused Verification Sweep

**Files:**
- Modify: `docs/training-pipeline-usage.md`
- Modify: `tests/test_notebook_docs.py`
- Test: `tests/test_notebook_docs.py`
- Test: `tests/test_notebook_suite_structure.py`
- Test: `tests/test_notebook_dataset_flow.py`
- Test: `tests/test_notebook_8fold_training.py`
- Test: `tests/test_notebook_reports.py`
- Test: `tests/test_notebook_final_deployment.py`
- Test: `tests/test_notebook_onnx_workflow.py`

**Interfaces:**
- Consumes:
  - new experiment knobs and notebook outputs from Tasks 1 through 5
- Produces:
  - notebook usage docs that accurately describe the improved procedure workflow

- [ ] **Step 1: Write the failing doc test**

```python
# tests/test_notebook_docs.py
from __future__ import annotations

from pathlib import Path


def test_training_pipeline_usage_describes_procedure_improvements() -> None:
    content = Path("docs/training-pipeline-usage.md").read_text(encoding="utf-8")

    assert "raw_center_crop_224" in content
    assert "severe-error rate" in content
    assert "worst-fold macro-F1" in content
    assert "sample-aware validation" in content
```

- [ ] **Step 2: Run the doc test to verify it fails**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_docs.py::test_training_pipeline_usage_describes_procedure_improvements -v`

Expected: `FAIL` because the current doc does not yet describe the new experiment surface.

- [ ] **Step 3: Write minimal documentation**

```markdown
# docs/training-pipeline-usage.md
## Procedure Controls

The `MobileNetV3Small` experiment notebooks now expose:

- `INPUT_MODE`
  - `processed_hsv_lab_threshold_roi_224`
  - `raw_center_crop_224`
- `AUGMENTATION_PRESET`
  - `conservative_v1`
- `FINE_TUNE_FRACTION`
  - `0.0`
  - `0.25`
  - `1.0`

## Reporting

In addition to `accuracy`, macro `precision`, macro `recall`, macro `F1`, and the confusion matrix, the report notebook now writes:

- severe-error rate
- worst-fold macro-F1
- mean and standard deviation across runs

## Final Deployment Validation

`06_train_final_deployment_model.ipynb` now uses sample-aware validation instead of random image-level validation.

## Raw Center Crop Note

`raw_center_crop_224` requires raw image sources in the audited manifest flow. If only the canonical processed ROI dataset is available locally, keep `INPUT_MODE=processed_hsv_lab_threshold_roi_224` until raw sources are provided.
```

- [ ] **Step 4: Run the full focused verification sweep**

Run: `& 'C:\Users\Adriaan M. Dimate\anaconda3\envs\meatlens-pork-training\python.exe' -m pytest tests\test_notebook_docs.py tests\test_notebook_suite_structure.py tests\test_notebook_dataset_flow.py tests\test_notebook_8fold_training.py tests\test_notebook_reports.py tests\test_notebook_final_deployment.py tests\test_notebook_onnx_workflow.py -v`

Expected:
- all targeted tests `PASS`
- only known notebook-format warnings are acceptable
- no new test should depend on `docs/windows_instructions.md` or `tmp/`

- [ ] **Step 5: Commit**

```bash
git add docs/training-pipeline-usage.md tests/test_notebook_docs.py
git commit -m "docs: describe MobileNetV3Small procedure improvements"
```

## Self-Review

### Spec coverage

- Shared experiment knobs for input mode, augmentation preset, and fine-tune fraction: Task 1
- Less-processed `raw_center_crop_224` branch: Task 2
- Explicit conservative augmentation: Task 3
- Fine-tune settings `0.0`, `0.25`, `1.0`: Tasks 1 and 3
- Preserve strict sample-held-out 8-fold evaluation: no change to split logic, reaffirmed in Task 3
- Severe-error reporting and worst-fold reporting: Tasks 3 and 4
- Sample-aware deployment validation: Task 5
- ONNX metadata continuity: Task 5
- Notebook usage docs: Task 6

### Placeholder scan

- No `TODO`, `TBD`, or “implement later” markers remain.
- Each task names exact files.
- Each test step includes an exact command and expected result.
- Each code step includes the intended function names and output files.

### Type consistency

- `INPUT_MODE`, `AUGMENTATION_PRESET`, and `SEVERE_ERROR_LABEL_PAIRS` originate in Task 1 and are reused consistently later.
- `resolve_input_root(...)` is produced in Task 1 and consumed in Task 2.
- `build_sample_heldout_validation_split(...)` is produced in Task 1 and consumed in Task 5.
- `compute_severe_error_rate(...)` is produced in Task 3 and consumed in Task 4.
- ONNX metadata reads `input_mode` from deployment metadata in Task 5 and preserves it consistently across `07` and `08`.

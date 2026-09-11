# MeatLens Official Run Checklist

Date: Friday, July 31, 2026

This is the exact notebook-first run plan for the training-1-compatible Roboflow pass.

## Goal

Run the full MeatLens training and export flow using the Roboflow dataset and the training-1-compatible end-to-end image-training methodology, while preserving the recorded cached Roboflow baseline.

## Environment

Use the real Windows GPU environment on the laptop with the `RTX 4050`.

```powershell
conda activate meatlens-tf210-gpu
```

## Default Strategy

Leave the training strategy on:

```python
TRAINING_STRATEGY = 'training1_compatible_end_to_end'
```

The preserved comparison baseline is explicitly:

```python
TRAINING_STRATEGY = 'roboflow_cached_baseline_v1'
```

It writes to the original cached-baseline namespace and must not overwrite it.

## Notebook Order

Run these in order:

1. `00_shared_setup.ipynb`
2. `01_manifest_and_dataset_audit.ipynb`
3. `02_roi_preprocessing.ipynb` only if new raw data or a new manifest arrived
4. `03_build_cross_rotation_splits.ipynb`
5. `04_train_8fold_mobilenetv3small.ipynb`
6. `05_regenerate_metrics_and_reports.ipynb`
7. `06_train_final_deployment_model.ipynb`
8. `07_export_onnx.ipynb`
9. `08_inference_smoke_test.ipynb`

## Recommended Overrides

These are the exact overrides I would use for the official run.

### Notebook 03

Normally no special override is needed if the local processed manifest is already the one you want.

If you want to be explicit:

```python
NOTEBOOK_OVERRIDES = {
    'NOTEBOOK_TEST_MODE': False,
    'DATASET_SOURCE': 'roboflow',
    'GENERATED_SPLITS_ROOT': ROOT / 'generated_splits' / 'roboflow',
}
```

### Notebook 04

Use this exact configuration for the official 8-fold run:

```python
NOTEBOOK_OVERRIDES = {
    'NOTEBOOK_TEST_MODE': False,
    'SKIP_GPU_CHECK': False,
    'INPUT_MODE': 'processed_hsv_lab_threshold_roi_224',
    'AUGMENTATION_PRESET': 'geometry_only_v1',
    'DATASET_SOURCE': 'roboflow',
    'TRAINING_STRATEGY': 'training1_compatible_end_to_end',
    'RUN_SEEDS': [42, 123, 2026],
    'SELECT_FOLDS': [f'fold{i}' for i in range(1, 9)],
    'MODEL_WEIGHTS': 'imagenet',
    'BATCH_SIZE': 32,
    'DETERMINISTIC_OPS': False,
    'EPOCHS_HEAD': 4,
    'EPOCHS_FINE': 8,
    'HEAD_LR': 5e-4,
    'FINE_TUNE_LR': 1e-5,
    'FINE_TUNE_FRACTION': 0.25,
    'USE_TRAINING_AUGMENTATION': True,
    'EIGHTFOLD_OUTPUT_ROOT': TRAINING_OUTPUTS_ROOT / 'mobilenetv3small_8fold_processed_roi_cnn_only_training1_compatible_end_to_end',
}
```

Notes:

- `FINE_TUNE_FRACTION` is kept for compatibility with the notebook contract.
- under `cached_embeddings_sgd_v1`, the actual learning path is frozen feature extraction plus cached embedding classification.
- the notebook still writes the same predictions, confusion matrices, histories, and metrics CSVs you already expect.

### Notebook 05

Usually no override is needed.

If you want to be explicit:

```python
NOTEBOOK_OVERRIDES = {
    'TRAINING_OUTPUTS_ROOT': ROOT / 'training_outputs',
}
```

### Notebook 06

Use this exact configuration for the final deployment model:

```python
NOTEBOOK_OVERRIDES = {
    'NOTEBOOK_TEST_MODE': False,
    'SKIP_GPU_CHECK': False,
    'INPUT_MODE': 'processed_hsv_lab_threshold_roi_224',
    'AUGMENTATION_PRESET': 'geometry_only_v1',
    'DATASET_SOURCE': 'roboflow',
    'TRAINING_STRATEGY': 'training1_compatible_end_to_end',
    'MODEL_WEIGHTS': 'imagenet',
    'FINAL_TRAINING_SEED': 42,
    'FINAL_VAL_SIZE': 0.15,
    'BATCH_SIZE': 32,
    'DETERMINISTIC_OPS': False,
    'EPOCHS_HEAD': 4,
    'EPOCHS_FINE': 8,
    'HEAD_LR': 5e-4,
    'FINE_TUNE_LR': 1e-5,
    'FINE_TUNE_FRACTION': 0.25,
    'USE_TRAINING_AUGMENTATION': True,
    'FINAL_DEPLOYMENT_OUTPUT_ROOT': TRAINING_OUTPUTS_ROOT / 'mobilenetv3small_8samples_final_deployment_cnn_only_training1_compatible_end_to_end',
}
```

### Notebook 07

Usually no override is needed unless you want a custom ONNX output path.

### Notebook 08

Usually no override is needed unless you want to point the smoke test at a different exported ONNX file.

## What Changed From Earlier Runs

- `04_train_8fold_mobilenetv3small.ipynb` now defaults to `DATASET_SOURCE='roboflow'` and `TRAINING_STRATEGY='training1_compatible_end_to_end'`.
- `06_train_final_deployment_model.ipynb` uses fold 1's generated Roboflow train/validation files and the same official strategy.
- the existing cached Roboflow baseline remains under `training_outputs_committable/roboflow/mobilenetv3small_8fold_processed_roi_cnn_only/`; its 92.7% accuracy record is unchanged.
- the output contract is still the same at the user-facing level: metrics CSVs, predictions CSVs, confusion matrix CSV/PNG, Keras model, ONNX export, and smoke-test outputs.
- you may now notice small internal staging folders like `_p` and `_e` under the training output root during notebook execution. Those are expected.

## Expected Main Outputs

From notebook `04`:

- `training_outputs/roboflow/mobilenetv3small_8fold_processed_roi_cnn_only_training1_compatible_end_to_end/processed_roi8_cnn_only_seed_metrics.csv`
- per-fold prediction CSVs
- per-fold confusion-matrix CSVs
- per-fold confusion-matrix PNGs

From notebook `06`:

- `training_outputs/roboflow/mobilenetv3small_8samples_final_deployment_cnn_only_training1_compatible_end_to_end/models/meatlens_final_8samples_cnn_only_mobilenetv3small.keras`
- `training_outputs/roboflow/mobilenetv3small_8samples_final_deployment_cnn_only_training1_compatible_end_to_end/final_validation_predictions.csv`
- `training_outputs/roboflow/mobilenetv3small_8samples_final_deployment_cnn_only_training1_compatible_end_to_end/final_training_history.csv`
- `training_outputs/roboflow/mobilenetv3small_8samples_final_deployment_cnn_only_training1_compatible_end_to_end/deployment_metadata.json`

From notebooks `07` and `08`:

- ONNX model
- ONNX smoke-test result

## Practical Advice For Tomorrow

- Run `00` first and confirm the default processed dataset is detected correctly.
- If there is no new raw data, skip `02`.
- Let `04` finish completely before opening `05`.
- Check the regenerated confusion matrices after `05`, especially the `not fresh` row.
- Use the official 8-fold outputs before deciding whether to keep seed `42` for deployment or to change the final-training seed.

## If You Need A Quick Sanity Run Instead Of The Full Official Run

Use this reduced override only for debugging, not for the official record:

```python
NOTEBOOK_OVERRIDES = {
    'NOTEBOOK_TEST_MODE': False,
    'SKIP_GPU_CHECK': False,
    'INPUT_MODE': 'processed_hsv_lab_threshold_roi_224',
    'AUGMENTATION_PRESET': 'geometry_only_v1',
    'TRAINING_STRATEGY': 'cached_embeddings_sgd_v1',
    'RUN_SEEDS': [42],
    'SELECT_FOLDS': ['fold1'],
    'MODEL_WEIGHTS': 'imagenet',
    'BATCH_SIZE': 32,
    'EPOCHS_HEAD': 4,
    'EPOCHS_FINE': 8,
    'HEAD_LR': 1e-4,
    'FINE_TUNE_LR': 1e-5,
    'FINE_TUNE_FRACTION': 0.25,
    'USE_TRAINING_AUGMENTATION': True,
}
```

That shortcut is only for a fast checkpoint run before committing to the full multi-seed multi-fold job.

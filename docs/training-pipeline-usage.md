# MeatLens MobileNetV3-Small Notebook Training Pipeline

## Environment

For real RTX 4050 training on native Windows, use the dedicated GPU conda environment:

```bash
conda env create -f environment.windows-gpu.yml
conda activate meatlens-tf210-gpu
```

This environment intentionally pins:

- `tensorflow==2.10.1`
- `numpy<2`
- `cudatoolkit=11.2`
- `cudnn=8.1.0`

Those pins keep native Windows TensorFlow GPU support working with the laptop `RTX 4050`.

If you also want the lighter repo/dev environment for non-training tasks, keep using:

```bash
conda env create -f environment.yml
conda activate meatlens-pork-training
```

## Dataset Default

The notebook suite is built around the dataset already present in the repo:

- processed dataset root: `data/processed_hsv_lab_threshold_roi_224`
- canonical manifest: `data/processed_hsv_lab_threshold_roi_224/processing_summary.csv`
- target labels: `fresh`, `not fresh`, `spoiled`

The stored `E:\...` paths inside `processing_summary.csv` are treated as old source references. The notebooks rebuild local paths from the local dataset structure instead of trusting those old absolute paths.

`00_shared_setup.ipynb` now reports whether that default processed dataset is ready, how many sample folders it sees, and how many manifest rows are available before you move into auditing or training.

Raw or Excel manifests are still supported in the early notebooks, but the current default workflow starts from the processed summary that is already in `data/`.

## GPU Requirement

Full training runs must use the laptop `RTX 4050`. The training notebooks check for an NVIDIA GPU name containing `RTX 4050` and also require TensorFlow to report a visible GPU device before running the real training flow.

Test-mode notebook runs can skip that guard, but the real 8-fold training and final deployment training are expected to run on the `RTX 4050`.

## Notebook Order

Run the notebooks in this order:

1. `00_shared_setup.ipynb`
2. `01_manifest_and_dataset_audit.ipynb`
3. `02_roi_preprocessing.ipynb`
4. `03_build_cross_rotation_splits.ipynb`
5. `04_train_8fold_mobilenetv3small.ipynb`
6. `05_regenerate_metrics_and_reports.ipynb`
7. `06_train_final_deployment_model.ipynb`
8. `07_export_onnx.ipynb`
9. `08_inference_smoke_test.ipynb`

Official evaluation metrics come from the 8-fold cross-rotation workflow.

## Recommended Workflow

### 1. Audit the current processed dataset

Start with:

- `00_shared_setup.ipynb`
- `01_manifest_and_dataset_audit.ipynb`

This confirms label normalization, sample IDs, and local image-path reconstruction from `processing_summary.csv`.

### 2. Refresh preprocessing only if needed

Run `02_roi_preprocessing.ipynb` when:

- new raw pork images arrive
- a new CSV or Excel manifest arrives
- you want to rebuild the ROI-processed dataset

If you are using the already-preprocessed dataset inside `data/processed_hsv_lab_threshold_roi_224`, this notebook is mainly for validation or refresh work.

### 3. Build the official folds

Run `03_build_cross_rotation_splits.ipynb`.

This creates:

- `fold1_train.csv` through `fold8_test.csv`
- `cross_rotation_summary.csv`
- `cross_rotation_leakage_check.csv`
- `all_sampled_images.csv`

### 4. Run official MobileNetV3-small training

Run `04_train_8fold_mobilenetv3small.ipynb`.

Default training contract:

- backbone: `MobileNetV3Small`
- mode: `cnn_only`
- input: `224x224x3`
- seeds: `42`, `123`, `2026`
- metrics: `accuracy`, macro `precision`, macro `recall`, macro `F1`
- confusion matrix: CSV and PNG

Procedure controls now exposed through the shared setup and training notebooks:

- `INPUT_MODE`
  - `processed_hsv_lab_threshold_roi_224`
  - `raw_center_crop_224`
- `AUGMENTATION_PRESET`
  - `conservative_v1`
- `FINE_TUNE_FRACTION`
  - `0.0`
  - `0.25`
  - `1.0`

Key output root:

```text
training_outputs/mobilenetv3small_8fold_processed_roi_cnn_only/
```

### 5. Regenerate official reports

Run `05_regenerate_metrics_and_reports.ipynb` after training finishes.

This rebuilds aggregate reports from saved prediction CSVs without rerunning the whole training loop.

In addition to the required core metrics, the report notebook now writes:

- severe-error rate
- worst-fold macro-F1
- mean and standard deviation across runs

### 6. Train the final deployment model

Run `06_train_final_deployment_model.ipynb` after the official 8-fold evaluation is done.

This notebook now uses sample-aware validation instead of random image-level validation.

Key output root:

```text
training_outputs/mobilenetv3small_8samples_final_deployment_cnn_only/
```

Important outputs include:

- `models/meatlens_final_8samples_cnn_only_mobilenetv3small.keras`
- `final_validation_predictions.csv`
- `final_training_history.csv`
- `deployment_metadata.json`

### 7. Export ONNX

Run `07_export_onnx.ipynb`.

This exports the final deployment model directly to ONNX.

### 8. Smoke-test the ONNX artifact

Run `08_inference_smoke_test.ipynb`.

This verifies the exported ONNX model with ONNX Runtime before handoff.

## Expected Outputs

Across the notebook suite, the main deliverables are:

- per-fold metrics CSVs
- official `accuracy`
- official macro `precision`
- official macro `recall`
- official macro `F1`
- confusion matrix CSVs
- confusion matrix PNGs
- per-image prediction CSVs
- final deployment Keras model
- deployment metadata JSON
- ONNX model
- ONNX smoke-test summary

## Raw Center Crop Note

`raw_center_crop_224` requires raw image sources in the audited manifest and preprocessing flow. If only the canonical processed ROI dataset is available locally, keep `INPUT_MODE=processed_hsv_lab_threshold_roi_224` until raw source images are provided.

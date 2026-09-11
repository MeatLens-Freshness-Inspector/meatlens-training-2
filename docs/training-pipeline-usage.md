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

The notebook suite explicitly uses the Roboflow dataset already present in the repo:

- Roboflow export root: `roboflow dataset/`
- processed dataset root: `data/roboflow_processed_hsv_lab_threshold_roi_224`
- generated folds: `generated_splits/roboflow/fold{i}_{train,val,test}.csv`
- target labels: `fresh`, `not fresh`, `spoiled`

The stored `E:\...` paths inside `processing_summary.csv` are treated as old source references. The notebooks rebuild local paths from the local dataset structure instead of trusting those old absolute paths.

`00_shared_setup.ipynb` now reports whether that default processed dataset is ready, how many sample folders it sees, and how many manifest rows are available before you move into auditing or training.

Raw or Excel manifests remain supported in the early audit/preprocessing notebooks, but the official training-2 workflow starts from the Roboflow export and its processed summary.

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
- frozen classification head for 4 epochs maximum, followed by top-25% backbone fine-tuning for 8 epochs maximum
- head learning rate `5e-4`; fine-tuning learning rate `1e-5`
- class weights and geometry-only augmentation applied only to training images
- checkpointing, early stopping, and learning-rate reduction monitored by validation macro-F1

Procedure controls now exposed through the shared setup and training notebooks:

- `INPUT_MODE`
  - `processed_hsv_lab_threshold_roi_224`
  - `raw_center_crop_224`
- `AUGMENTATION_PRESET`
  - `geometry_only_v1`
  - `conservative_v1`
- `TRAINING_STRATEGY`
  - default `training1_compatible_end_to_end`
  - `roboflow_cached_baseline_v1`
  - legacy aliases remain accepted by the package CLI
- `HEAD_LR`
  - default `5e-4`
- `FINE_TUNE_FRACTION`
  - `0.0`
  - `0.25`
  - `1.0`
- `CACHE_MODE`
  - default `memory`; caches decoded/resized images before random augmentation
  - `none` disables decoded-image caching when host RAM is constrained
- `DETERMINISTIC_OPS`
  - default `False` for the pinned native-Windows TensorFlow 2.10 GPU environment because its deterministic `UnsortedSegmentSum` gradient kernel is unavailable
  - Python, NumPy, TensorFlow seeds, split random states, fold assignments, and manifests remain fixed
  - set `True` only on a runtime that supports the required deterministic GPU kernels; otherwise training fails before completing an epoch
- `TRAIN_VERBOSE`
  - default `2` for one concise Keras line per epoch
- `EPOCHS_HEAD` and `EPOCHS_FINE`
  - official reduced-budget defaults are 4 and 8 maximum epochs, respectively
  - all 8 folds and 3 seeds are retained; this changes the training budget, not the evaluation coverage
- `f1_macro` label-shape handling
  - the official metric accepts one-hot labels and sparse labels, including the `(batch, 1)` shape emitted by Keras during evaluation
  - runs produced before this fix must be rerun before their checkpoint-selection metrics are used as the official result
- Each fold writes timing records to `logs/<fold>_seed<seed>_performance.jsonl`.
  These records include epoch duration, train/validation batch time, images per
  second, and validation metrics. GPU utilization and mixed-precision benefit
  still require measurement on the actual RTX 4050 environment.

For the package CLI, the equivalent training controls are:

```text
python -m meatlens_pork_pipeline.cli train ... --cache-mode memory --deterministic-ops --verbose 2
python -m meatlens_pork_pipeline.cli train ... --cache-mode none --no-deterministic-ops --verbose 0
```

Strategy and accuracy provenance:

- `training1_compatible_end_to_end` is the official training-2 path and follows the training-1 CNN methodology.
- `roboflow_cached_baseline_v1` maps to the existing cached-embedding implementation and writes to the original baseline namespace.
- The recorded Roboflow fold 4 / seed 123 baseline remains exactly `0.927038626609442` (92.7%) under `training_outputs_committable/roboflow/mobilenetv3small_8fold_processed_roi_cnn_only/`.
- The training-1-compatible rerun has a separate output namespace and does not overwrite or relabel that baseline. Its accuracy must be reported from its own saved predictions after the RTX 4050 run.

The previously completed reduced-budget campaign was generated before the
sparse-label macro-F1 correction. Its test predictions remain useful as
diagnostic evidence, but its validation macro-F1 values and best-checkpoint
selection are not methodologically valid. Preserve that output for audit and
rerun the same folds and seeds with the corrected code in a new output root.

Key output root:

```text
training_outputs/roboflow/mobilenetv3small_8fold_processed_roi_cnn_only_training1_compatible_end_to_end/
```

### 5. Regenerate official reports

Run `05_regenerate_metrics_and_reports.ipynb` after training finishes.

This rebuilds aggregate reports from saved prediction CSVs without rerunning the whole training loop.

The report notebook selects the same strategy-specific root as notebook 04. It regenerates reports from saved prediction CSVs without rerunning training.

In addition to the required core metrics, the report notebook now writes:

- severe-error rate
- worst-fold macro-F1
- mean and standard deviation across runs

### 6. Train the final deployment model

Run `06_train_final_deployment_model.ipynb` after the official 8-fold evaluation is done.

This notebook now uses sample-aware validation instead of random image-level validation.

Key output root:

```text
training_outputs/roboflow/mobilenetv3small_8samples_final_deployment_cnn_only_training1_compatible_end_to_end/
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

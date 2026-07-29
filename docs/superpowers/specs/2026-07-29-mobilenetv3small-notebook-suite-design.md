# MobileNetV3-Small Notebook Suite Design

## Goal

Replace the current script/package-first training flow with a notebook-first training suite in the root workspace for pork freshness classification. The suite should mirror the partner notebook workflow as closely as practical while adapting it to the new project root, the already-present processed dataset under `data/processed_hsv_lab_threshold_roi_224`, any future raw manifest flow, and the requirement to end with an ONNX artifact.

Target classes are exactly:

- `fresh`
- `not fresh`
- `spoiled`

## Scope

This design covers:

- codebase architecture for the notebook suite
- workflow architecture across notebooks
- model architecture and training defaults
- reliability, validation, and export expectations

This design does not yet implement:

- the final notebook files
- final retraining results

## Existing Context

The partner workflow shown in [new6_mobilenetv3small_8fold_processed_roi_cnn_only.ipynb](C:/Users/Adriaan%20M.%20Dimate/Downloads/new6_mobilenetv3small_8fold_processed_roi_cnn_only.ipynb) establishes the baseline design to mirror:

- `MobileNetV3Small`
- `cnn_only`
- `8-fold cross-rotation` as the official evaluation
- processed ROI images at `224x224`
- class-weighted training
- augmentation during training
- two-stage training: frozen head, then fine-tuning
- `val_f1_macro` as the main callback monitor
- separate final deployment training after official evaluation
- separate artifact export for deployment

The current root workspace at [meatlens-training-2](C:/Users/Adriaan%20M.%20Dimate/Desktop/development/school/meatlens-training-2) already contains reference pipeline code and tests, but the new primary interface should be notebooks rather than the current Python-entrypoint flow.

The current `data/` folder already changes the implementation assumptions in a useful way:

- canonical dataset root is `data/processed_hsv_lab_threshold_roi_224`
- canonical manifest is `data/processed_hsv_lab_threshold_roi_224/processing_summary.csv`
- there are `8` sample folders, each arranged by label
- the manifest already contains `sample_number`, `sample_id`, and `label`
- the stored `E:\...` source paths are not portable, so local image paths should be rebuilt from local columns instead of trusted directly
- the dataset includes mixed `.jpg` and `.png` images
- sample counts are not perfectly uniform, especially for the `fresh` class in samples `3`, `4`, and `7`

## Design Choice

The chosen direction is a notebook-only suite with multiple focused notebooks.

To reduce notebook duplication without falling back to Python helper modules as the main control surface, the suite may use one shared setup notebook loaded by other notebooks with `%run`. The notebooks remain the primary entrypoint and the primary artifact the team uses.

## Codebase Architecture

The root workspace should be organized around a notebook suite:

- `00_shared_setup.ipynb`
  - shared imports
  - constants
  - label mappings
  - path definitions
  - seed definitions
  - reusable helper cells for the other notebooks
- `01_manifest_and_dataset_audit.ipynb`
  - default audit of `processing_summary.csv`
  - optional raw manifest loading
  - label normalization
  - local path rebuilding
  - missing-file checks
  - dataset count and quality audits
- `02_roi_preprocessing.ipynb`
  - optional raw-image to processed-ROI refresh
  - processed-manifest normalization
  - summary CSVs
  - failure logs
  - sample visualization cells
- `03_build_cross_rotation_splits.ipynb`
  - creation of official `8-fold cross-rotation` split CSVs
  - leakage checks
  - per-fold distribution summaries
- `04_train_8fold_mobilenetv3small.ipynb`
  - official fold-and-seed training runs
  - saved models
  - predictions
  - training histories
  - per-run metrics
- `05_regenerate_metrics_and_reports.ipynb`
  - aggregation of official evaluation outputs
  - confusion matrices
  - summary CSVs
  - comparison-ready metrics tables
- `06_train_final_deployment_model.ipynb`
  - trains the final deployment model after official evaluation is complete
- `07_export_onnx.ipynb`
  - exports the final deployment model to ONNX
  - writes metadata JSON
  - validates artifact creation
- `08_inference_smoke_test.ipynb`
  - loads the ONNX artifact with ONNX Runtime
  - runs a small sanity-check batch

Supporting project folders should stay explicit and stable:

- `docs/`
- `generated_splits/`
- `data/processed_hsv_lab_threshold_roi_224/`
- `data/processed_hsv_lab_threshold_roi_224/processing_summary.csv`
- `training_outputs/`

The current Python package and scripts may remain temporarily as reference material during migration, but the notebook suite becomes the preferred operational interface.

## Workflow Architecture

The notebook flow should be linear and explicit:

1. `00_shared_setup.ipynb`
   - defines the project contract for all downstream notebooks
2. `01_manifest_and_dataset_audit.ipynb`
   - audits the existing processed dataset summary by default
   - validates labels, sample IDs, and rebuilt local file resolution
3. `02_roi_preprocessing.ipynb`
   - optionally refreshes or rebuilds the canonical ROI-preprocessed dataset from raw data when needed
4. `03_build_cross_rotation_splits.ipynb`
   - generates the authoritative 8-fold split files
5. `04_train_8fold_mobilenetv3small.ipynb`
   - runs the official experiments across folds and seeds
6. `05_regenerate_metrics_and_reports.ipynb`
   - rebuilds the official aggregate metrics and confusion-matrix reports from saved predictions
7. `06_train_final_deployment_model.ipynb`
   - trains a final shippable model using the validated full dataset
8. `07_export_onnx.ipynb`
   - exports the deployment model to ONNX and writes deployment metadata
9. `08_inference_smoke_test.ipynb`
   - confirms the ONNX artifact works before handoff

The key workflow rule is:

- official performance claims come from the 8-fold evaluation notebooks
- deployment notebooks produce the shippable artifact but do not replace the official evaluation

## Model Architecture

The model design should stay close to the partner workflow for comparability:

- backbone: `MobileNetV3Small`
- pretrained initialization: ImageNet
- mode: `cnn_only`
- input shape: `224x224x3`
- classes: `fresh`, `not fresh`, `spoiled`
- no handcrafted RGB, HSV, LAB, or GLCM feature branch

Training stages:

- stage 1: freeze the backbone and train the classification head
- stage 2: unfreeze the top part of the backbone and fine-tune

Default training settings to mirror the partner setup:

- seeds: `42`, `123`, `2026`
- batch size: `32`
- head epochs: `8`
- fine-tune epochs: `20`
- head learning rate: `5e-4`
- fine-tune learning rate: `1e-5`
- fine-tune fraction: top `25%` of backbone layers
- augmentation: enabled on training data only
- callback monitor: `val_f1_macro`

The final deployment notebook should train a final model after the official 8-fold evaluation is complete. That deployment model becomes the source model for ONNX export.

## Data Contract

The first notebook should be designed around the dataset that already exists in the repo, while still allowing a raw-manifest path later:

- default input manifest: `data/processed_hsv_lab_threshold_roi_224/processing_summary.csv`
- canonical columns used by the official workflow:
  - `image_file_name`
  - `label`
  - `sample_number`
  - `sample_id`
- local processed image paths should be rebuilt from:
  - dataset root
  - `sample_number`
  - `label`
  - `image_file_name`
- optional alternate manifest format support may still include `.csv`, `.xlsx`, or `.xls`

Labels must normalize to:

- `fresh`
- `not fresh`
- `spoiled`

The current manifest should be treated as the authoritative default. If a later incoming raw dataset differs, only the early ingestion and preprocessing notebooks should need adaptation while the rest of the notebook suite remains stable.

## Preprocessing Contract

The training input contract should remain:

- center-square crop
- resize to `224x224`
- HSV/LAB-threshold ROI extraction
- neutralized background
- final RGB output
- stored under `data/processed_hsv_lab_threshold_roi_224`

This preserves compatibility with the partner's processed-ROI workflow and keeps the model input mode consistent for later ONNX deployment.

The notebook suite must also tolerate the current processed dataset realities:

- mixed `.jpg` and `.png` files
- imperfectly balanced per-sample label counts
- non-portable original `E:\...` paths inside the CSV

## Metrics And Outputs

The notebook suite must produce:

- `accuracy`
- macro `precision`
- macro `recall`
- macro `F1`
- confusion matrix CSV
- confusion matrix PNG
- per-image predictions CSV
- fold-level summary CSVs
- deployment metadata JSON
- ONNX model file

Secondary transition-aware metrics may also be included if the team wants continuity with the partner workflow:

- top-2 accuracy
- adjacent accuracy
- severe error rate
- mean absolute ordinal error

These are supplementary. The required primary metrics remain accuracy, precision, recall, F1, and confusion matrix outputs.

## Reliability And Validation

Each notebook should fail early with explicit checks rather than silently continuing.

Validation expectations:

- manifest notebook stops on missing required columns
- manifest notebook stops on unsupported labels
- manifest notebook stops on unresolved image paths
- preprocessing notebook writes failure logs and sample visualizations
- split notebook checks leakage and class distribution before saving
- training notebook can skip already completed fold-and-seed runs if artifacts exist
- failed runs are logged separately so reruns can continue
- report notebook rebuilds summaries from saved predictions rather than manual values
- export notebook validates ONNX creation
- smoke-test notebook verifies the exported ONNX artifact runs in ONNX Runtime

This keeps the notebook workflow reproducible and debuggable even when long-running training is interrupted.

## Migration Intent

The current root Python pipeline should be treated as a transitional reference implementation. The next phase should refactor the preferred user workflow into notebooks while preserving the validated model contract:

- MobileNetV3Small
- processed ROI input
- official 8-fold evaluation
- final deployment training
- ONNX export

## Success Criteria

The redesign is successful when:

- the root workspace exposes a clear multi-notebook training suite
- users can run the pipeline notebook-by-notebook without relying on a single script entrypoint
- official evaluation is clearly separated from deployment training
- the workflow ends in a validated ONNX artifact
- the notebook suite is structurally aligned with the partner's training flow

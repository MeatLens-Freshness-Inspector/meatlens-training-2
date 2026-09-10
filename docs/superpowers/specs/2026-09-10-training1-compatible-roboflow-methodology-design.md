# Training-1-Compatible Methodology for the Roboflow Training-2 Pipeline

## Goal

Make `meatlens-training-2` provide an exact `training-1`-compatible end-to-end MobileNetV3Small training path while keeping the Roboflow dataset, its deterministic eight-partition rotation, and the existing 92.7% cached-embedding baseline intact.

## Constraints and evidence

- The dataset source for `training-2` remains the local Roboflow export.
- Roboflow has no physical `sample_id` grouping, so its current deterministic stratified eight-partition rotation remains the evaluation protocol.
- The existing baseline is `fold4 / seed123`, with accuracy `0.927038626609442`, recorded in `training_outputs_committable/roboflow/mobilenetv3small_8fold_processed_roi_cnn_only/_e/fold4_s123/test_metrics.json`.
- `training-1` uses MobileNetV3Small CNN-only training, processed HSV/LAB ROI input at `224x224`, class weighting, training augmentation, a frozen-head phase, top-25% fine-tuning, `HEAD_LR=5e-4`, `FINE_TUNE_LR=1e-5`, and validation macro-F1 monitoring.
- Existing baseline artifacts must not be overwritten or relabeled as end-to-end results.

## Design

### Dataset and folds

Keep `DATASET_SOURCE='roboflow'` as the explicit official source for `training-2`. Preserve the current Roboflow manifest, processed images, and generated `fold1` through `fold8` train/validation/test CSVs. Preserve Roboflow's native split only as provenance metadata. Do not substitute the sample-held-out current-dataset protocol because that would change the dataset methodology and invalidate the baseline comparison.

### Training strategies

Expose two explicit strategies:

- `training1_compatible_end_to_end`: the official compatibility path. It trains the image model end-to-end in two phases: frozen backbone/head training followed by top-25% backbone fine-tuning. It uses the `training-1` architecture, augmentation behavior, class weights, learning rates, epoch counts, and `val_f1_macro` checkpoint/early-stopping monitor.
- `roboflow_cached_baseline_v1`: the existing cached-embedding SGD path. It remains available for reproducing the recorded 92.7% baseline and is never silently treated as the end-to-end method.

The compatibility path writes to a separate output namespace from the cached baseline. Existing committable baseline files remain unchanged.

### Model and preprocessing contract

The compatibility path uses:

- MobileNetV3Small, ImageNet weights, CNN-only mode;
- processed HSV/LAB threshold ROI images, neutral gray background, RGB, `224x224x3`;
- the `training-1` classification head and two-stage training schedule;
- augmentation only for training batches, with validation and test inputs unaugmented;
- categorical labels in the order `fresh`, `not fresh`, `spoiled`;
- balanced class weights;
- deterministic seeds `42`, `123`, and `2026`.

### Reporting and artifact provenance

Official reports must identify the strategy, dataset source, fold, seed, model settings, and metric source. The existing `0.927038626609442` accuracy remains attributed only to the cached Roboflow baseline. New end-to-end results are reported separately and are not required to equal the cached baseline unless a fresh run demonstrates that independently.

## Verification

Add or update tests to verify:

1. Roboflow remains the selected dataset source and its eight generated folds remain disjoint.
2. The compatibility strategy has the `training-1` architecture and training defaults.
3. Augmentation is applied only to training data.
4. The end-to-end path uses `val_f1_macro` for checkpointing and early stopping, performs the two phases, and fine-tunes the top 25%.
5. Cached-baseline artifacts and the recorded 92.7% metric remain unchanged.
6. Notebook defaults and documentation distinguish the two strategies and their metric provenance.

The complete local verification gate is the repository's pytest suite plus `git diff --check`. Full GPU training is an external run; it must not be represented as locally verified unless executed on the required RTX 4050 environment.

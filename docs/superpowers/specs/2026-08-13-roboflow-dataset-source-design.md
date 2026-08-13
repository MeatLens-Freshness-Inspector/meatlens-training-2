# Roboflow Dataset Source Design

## Goal

Allow the notebook pipeline to use either the existing MeatLens dataset or the local Roboflow export, with `FRESH` mapped to `fresh`, `HALF-FRESH` mapped to `not fresh`, and `SPOILED` mapped to `spoiled`.

## Design

`DATASET_SOURCE` defaults to `current` and is selectable through the existing notebook override mechanism. The `roboflow` option reads `roboflow dataset/{train,valid,test}/_classes.csv`, resolves each image path, converts the one-hot class columns into the project labels, and writes a normalized manifest containing the native `roboflow_split`.

Notebook `01` will select the source manifest. Notebook `02` will preprocess Roboflow images while preserving their native split. Notebook `03` will emit a single `fold1` whose train/validation/test files are Roboflow train/valid/test, while retaining the current dataset’s eight-fold cross-rotation behavior. Notebook `04` will default to `fold1` for Roboflow and all eight folds for the current dataset. Notebook `06` will use the native Roboflow train/valid files for final deployment validation.

## Safety

- Existing behavior remains the default.
- Roboflow rows must have exactly one active status label.
- Missing image files and unsupported dataset sources fail with clear errors.
- The Roboflow dataset remains user-provided local data and is not copied into source control.

## Testing

Add tests for label conversion, source selection, native split generation, and notebook contracts. Run the targeted suite and the full available suite.

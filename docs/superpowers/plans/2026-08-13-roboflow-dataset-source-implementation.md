# Roboflow Dataset Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans (inline execution requested). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable current/Roboflow dataset source while preserving Roboflow’s native train/valid/test split.

**Architecture:** Add reusable Roboflow manifest conversion helpers in `meatlens_pork_pipeline`, expose source paths and defaults through `00_shared_setup.ipynb`, and branch notebooks `01`, `02`, `03`, `04`, and `06` at their existing data boundaries. Current-dataset behavior remains the default.

**Tech Stack:** Python 3.10, pandas, pathlib, Jupyter notebooks, pytest.

## Global Constraints

- `DATASET_SOURCE` defaults to `current`.
- `FRESH` maps to `fresh`; `HALF-FRESH` maps to `not fresh`; `SPOILED` maps to `spoiled`.
- Preserve Roboflow’s native `train/valid/test` split.
- Do not add the user’s Roboflow images to git.
- Existing current-dataset tests and behavior must remain valid.

---

### Task 1: Add tested Roboflow manifest conversion helpers

**Files:**
- Create: `meatlens_pork_pipeline/roboflow.py`
- Test: `tests/unit/manifest/test_roboflow.py`

- [x] Write a failing test for one-hot status conversion, split manifest construction, ambiguous labels, and missing image failure.
- [x] Run the new tests and confirm the helper API is missing.
- [x] Implement `ROBOFLOW_STATUS_COLUMNS`, `ROBOFLOW_LABEL_MAP`, `normalize_roboflow_row(row, split, image_root)`, and `build_roboflow_manifest(dataset_root)`.
- [x] Run the tests and confirm they pass.

### Task 2: Add dataset-source configuration and source selection

**Files:**
- Modify: `00_shared_setup.ipynb`
- Modify: `01_manifest_and_dataset_audit.ipynb`
- Test: `tests/integration/runtime/test_notebook_suite_structure.py`

- [x] Add failing source-contract tests for `DATASET_SOURCE`, `ROBOFLOW_DATASET_ROOT`, and the Roboflow manifest builder call.
- [x] Add defaults and source validation to shared setup.
- [x] Make notebook `01` choose the current canonical manifest or generate a Roboflow manifest, then audit it.
- [x] Run source-contract tests and current manifest tests.

### Task 3: Preserve native Roboflow split through preprocessing and fold generation

**Files:**
- Modify: `02_roi_preprocessing.ipynb`
- Modify: `03_build_cross_rotation_splits.ipynb`
- Test: `tests/integration/data_pipeline/test_notebook_split_flow.py`

- [x] Add failing tests proving the Roboflow split column survives preprocessing and produces `fold1_train.csv`, `fold1_val.csv`, and `fold1_test.csv` from native splits.
- [x] Preserve `roboflow_split` in preprocessing output.
- [x] Add a native-split branch in notebook `03`; leave current eight-fold validation unchanged.
- [x] Run data-pipeline tests.

### Task 4: Select the correct downstream fold and final validation path

**Files:**
- Modify: `04_train_8fold_mobilenetv3small.ipynb`
- Modify: `06_train_final_deployment_model.ipynb`
- Test: `tests/integration/training_pipeline/test_notebook_8fold_training.py`
- Test: `tests/integration/training_pipeline/test_notebook_final_deployment.py`

- [x] Add notebook contract tests for Roboflow default fold selection and native final validation.
- [x] Default `SELECT_FOLDS` to `['fold1']` for Roboflow and keep eight folds for current data.
- [x] Make final deployment use native Roboflow train/valid files when selected.
- [x] Run training/final-deployment integration tests.

### Task 5: Verify and merge

- [x] Run targeted tests with a short Windows pytest temp directory.
- [x] Run the available unit, runtime, data-pipeline, and training integration suites.
- [x] Inspect diff and confirm no Roboflow images are staged.
- [ ] Commit, merge fast-forward into `master`, and remove the feature worktree.

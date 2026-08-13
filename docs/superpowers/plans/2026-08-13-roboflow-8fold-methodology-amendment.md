# Roboflow 8-Fold Methodology Amendment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans (inline execution requested). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Roboflow dataset follow the thesis’s existing deterministic 8-fold methodology.

**Architecture:** Pool the imported Roboflow manifest, assign stratified partitions with a fixed seed, and emit the same `fold1`–`fold8` CSV contract used by the current dataset. Preserve `roboflow_split` only as provenance; downstream training and final deployment consume generated folds.

**Tech Stack:** Python 3.10, pandas, scikit-learn, Jupyter notebooks, pytest.

## Global Constraints

- The thesis methodology requires eight folds.
- Each fold has disjoint train, validation, and test rows.
- Roboflow’s original split remains available as metadata.
- `DATASET_SOURCE='current'` behavior remains unchanged.
- Do not add Roboflow images to git.

### Task 1: Make the split regression test require eight Roboflow folds

**Files:** `tests/integration/data_pipeline/test_notebook_split_flow.py`, `tests/integration/runtime/test_notebook_suite_structure.py`

- [x] Replace the native three-file expectation with a balanced 24-row Roboflow fixture and require `fold1` through `fold8`.
- [x] Require every fold’s train/validation/test image names to be disjoint and require the summary to contain eight rows.
- [x] Require notebook 04’s Roboflow default fold list to contain all eight folds.
- [x] Run the focused tests and observe failure because the current implementation only emits fold1.

### Task 2: Generate deterministic stratified eight-fold splits

**Files:** `03_build_cross_rotation_splits.ipynb`

- [x] Add a Roboflow builder using `StratifiedKFold(n_splits=8, shuffle=True, random_state=2026)` over all rows.
- [x] Emit `fold{i}_train.csv`, `fold{i}_val.csv`, and `fold{i}_test.csv` for every `i`.
- [x] Emit leakage checks based on row image identity, not empty sample IDs.
- [x] Run data-pipeline tests and confirm they pass.

### Task 3: Align downstream defaults with eight-fold output

**Files:** `04_train_8fold_mobilenetv3small.ipynb`, `06_train_final_deployment_model.ipynb`

- [x] Default Roboflow training to all eight folds.
- [x] Make final deployment consume generated `fold1_train.csv` and `fold1_val.csv` for Roboflow.
- [x] Run training and final-deployment regression tests.

### Task 4: Verify and merge

- [x] Run the complete test suite.
- [x] Inspect staged paths and confirm no Roboflow images are included.
- [ ] Commit, fast-forward merge into `master`, restore user notebook edits, and remove the worktree.

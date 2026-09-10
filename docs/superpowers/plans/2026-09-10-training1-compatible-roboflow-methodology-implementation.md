# Training-1-Compatible Roboflow Methodology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Roboflow-backed `meatlens-training-2` notebook workflow reproduce the `training-1` end-to-end MobileNetV3Small methodology while preserving the existing cached Roboflow baseline and its recorded 92.7% accuracy.

**Architecture:** Keep the current Roboflow manifest and deterministic eight-partition rotation as the sole dataset protocol for `training-2`. Add one explicit `training1_compatible_end_to_end` strategy that owns the exact two-stage image-training path, and retain the existing cached-embedding implementation under `roboflow_cached_baseline_v1`; each strategy writes to a separate output namespace and records its provenance.

**Tech Stack:** Python 3.10, TensorFlow/Keras, MobileNetV3Small, pandas, scikit-learn, Pillow, Jupyter notebooks, pytest, ONNX Runtime.

## Global Constraints

- The dataset source for `training-2` remains `DATASET_SOURCE='roboflow'`.
- The current Roboflow export, processed images, and generated eight-fold CSVs remain unchanged.
- Roboflow native `train`, `valid`, and `test` labels remain provenance only; generated stratified eight-partition folds define evaluation.
- The official compatibility strategy is `training1_compatible_end_to_end`.
- The cached baseline strategy is `roboflow_cached_baseline_v1` and maps to the existing cached-embedding implementation.
- The recorded baseline accuracy remains exactly `0.927038626609442` for Roboflow fold 4 / seed 123.
- Backbone is `MobileNetV3Small` with ImageNet weights and CNN-only mode.
- Input is processed HSV/LAB-threshold ROI, neutral gray background, RGB, `224x224x3`.
- Labels are exactly `fresh`, `not fresh`, and `spoiled` in that order.
- Training uses class weights, seeds `42`, `123`, and `2026`, and training-only augmentation.
- End-to-end training uses a frozen-head phase followed by top-25% backbone fine-tuning.
- End-to-end defaults are batch size `32`, head epochs `8`, fine-tune epochs `20`, head learning rate `5e-4`, and fine-tune learning rate `1e-5`.
- End-to-end checkpointing, early stopping, and learning-rate reduction monitor validation macro-F1.
- Existing cached baseline artifacts are never overwritten or relabeled as end-to-end outputs.
- No Roboflow image files are added to source control.
- Full GPU training is only reported as executed when run on the required RTX 4050 environment.

## File Map

- `meatlens_pork_pipeline/config.py`: named training strategies and shared default values.
- `meatlens_pork_pipeline/modeling.py`: the training-1-compatible MLP classification head.
- `meatlens_pork_pipeline/training.py`: shared image sequence, augmentation-aware two-stage trainer, strategy normalization, and F1 callback.
- `meatlens_pork_pipeline/cli.py`: CLI strategy names and training defaults.
- `00_shared_setup.ipynb`: Roboflow/default strategy and training contract exposed to notebooks.
- `04_train_8fold_mobilenetv3small.ipynb`: official fold/seed training and strategy-specific output roots.
- `05_regenerate_metrics_and_reports.ipynb`: strategy-aware report regeneration.
- `06_train_final_deployment_model.ipynb`: final deployment training from generated Roboflow fold 1 files.
- `07_export_onnx.ipynb` and `08_inference_smoke_test.ipynb`: export and smoke-test the selected official deployment artifact.
- `docs/training-pipeline-usage.md` and `training_outputs_committable/README.md`: metric and artifact provenance.
- `tests/unit/training/test_training1_compatibility.py`: unit contract for exact training settings and behavior.
- `tests/integration/training_pipeline/test_strategy_provenance.py`: notebook strategy wiring and baseline preservation.
- `tests/integration/runtime/test_notebook_suite_structure.py`: notebook source contract.

---

### Task 1: Lock the training strategies and the existing 92.7% baseline with tests

**Files:**
- Create: `tests/unit/training/test_training1_compatibility.py`
- Create: `tests/integration/training_pipeline/test_strategy_provenance.py`
- Modify: `tests/unit/cli/test_cli_training_strategy.py`
- Modify: `tests/integration/runtime/test_notebook_suite_structure.py`

**Interfaces:**
- Consumes: current Roboflow baseline files under `training_outputs_committable/roboflow/`.
- Produces: failing tests that define `TRAINING_STRATEGIES`, the exact default values, baseline hashes, and notebook strategy names required by later tasks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/training/test_training1_compatibility.py
from meatlens_pork_pipeline.config import (
    END_TO_END_DEFAULTS,
    ROBOFLOW_CACHED_BASELINE_STRATEGY,
    TRAINING1_COMPATIBLE_STRATEGY,
    TRAINING_STRATEGIES,
)


def test_named_training_strategies_have_one_official_and_one_baseline() -> None:
    assert TRAINING1_COMPATIBLE_STRATEGY == "training1_compatible_end_to_end"
    assert ROBOFLOW_CACHED_BASELINE_STRATEGY == "roboflow_cached_baseline_v1"
    assert set(TRAINING_STRATEGIES) == {
        "training1_compatible_end_to_end",
        "roboflow_cached_baseline_v1",
    }


def test_training1_defaults_match_training1() -> None:
    assert END_TO_END_DEFAULTS == {
        "batch_size": 32,
        "epochs_head": 8,
        "epochs_fine": 20,
        "head_lr": 5e-4,
        "fine_tune_lr": 1e-5,
        "fine_tune_fraction": 0.25,
        "augmentation": True,
        "monitor": "val_f1_macro",
    }
```

```python
# tests/integration/training_pipeline/test_strategy_provenance.py
import hashlib
import json
from pathlib import Path


BASELINE_ROOT = Path("training_outputs_committable/roboflow/mobilenetv3small_8fold_processed_roi_cnn_only")
EXPECTED_METRICS_SHA256 = "E5B2108565733E9947A6ABA8BCFFB12CB6272065A009E4D4858D95D11C998697"
EXPECTED_MODEL_SHA256 = "A51C7C899BA0E65BC209714095881FA7781BAA77717FA50984427DBFCA138131"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_recorded_roboflow_baseline_remains_92_7_percent() -> None:
    metrics_path = BASELINE_ROOT / "_e/fold4_s123/test_metrics.json"
    model_path = BASELINE_ROOT / "models/processed_roi8_cnn_only_fold4_seed123.keras"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics["accuracy"] == 0.927038626609442
    assert _sha256(metrics_path) == EXPECTED_METRICS_SHA256
    assert _sha256(model_path) == EXPECTED_MODEL_SHA256


def test_official_roboflow_outputs_use_a_separate_strategy_namespace() -> None:
    notebook_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(Path("04_train_8fold_mobilenetv3small.ipynb").read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "training1_compatible_end_to_end" in notebook_source
    assert "roboflow_cached_baseline_v1" in notebook_source
    assert "training1_compatible_end_to_end" in notebook_source.split("TRAINING_OUTPUTS_ROOT", 1)[-1]
```

The CLI test must parse both new names and assert that the old `cached_embeddings_sgd_v1` name remains accepted as a backward-compatible alias.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
conda run -n meatlens-pork-training python -m pytest tests/unit/training/test_training1_compatibility.py tests/integration/training_pipeline/test_strategy_provenance.py tests/unit/cli/test_cli_training_strategy.py -v
```

Expected: collection or assertion failures because the new strategy constants, defaults, and notebook namespaces do not yet exist.

- [ ] **Step 3: Commit the failing contract tests**

```powershell
git add tests/unit/training/test_training1_compatibility.py tests/integration/training_pipeline/test_strategy_provenance.py tests/unit/cli/test_cli_training_strategy.py tests/integration/runtime/test_notebook_suite_structure.py
git commit -m "test: define training1-compatible Roboflow contract"
```

### Task 2: Implement the exact training-1-compatible end-to-end strategy

**Files:**
- Modify: `meatlens_pork_pipeline/config.py`
- Modify: `meatlens_pork_pipeline/modeling.py`
- Modify: `meatlens_pork_pipeline/training.py`
- Modify: `meatlens_pork_pipeline/cli.py`
- Test: `tests/unit/training/test_training1_compatibility.py`
- Test: `tests/unit/cli/test_cli_training_strategy.py`

**Interfaces:**
- Consumes: train and validation CSVs with `processed_image_path`, `local_image_path`, or `processed_output_file`, labels in `LABEL_ORDER`, and the selected strategy name.
- Produces: `train_model(..., training_strategy="training1_compatible_end_to_end") -> TrainingArtifacts` using the exact two-stage image path; `train_model(..., training_strategy="roboflow_cached_baseline_v1")` continues to use cached embeddings.

- [ ] **Step 1: Add the shared strategy constants**

Add these definitions to `meatlens_pork_pipeline/config.py` without removing existing path dataclasses:

```python
TRAINING1_COMPATIBLE_STRATEGY = "training1_compatible_end_to_end"
ROBOFLOW_CACHED_BASELINE_STRATEGY = "roboflow_cached_baseline_v1"
TRAINING_STRATEGIES = (
    TRAINING1_COMPATIBLE_STRATEGY,
    ROBOFLOW_CACHED_BASELINE_STRATEGY,
)
END_TO_END_DEFAULTS = {
    "batch_size": 32,
    "epochs_head": 8,
    "epochs_fine": 20,
    "head_lr": 5e-4,
    "fine_tune_lr": 1e-5,
    "fine_tune_fraction": 0.25,
    "augmentation": True,
    "monitor": "val_f1_macro",
}
```

- [ ] **Step 2: Add the training-1 classification head variant**

Add a `training1_mlp_v1` branch to `build_classification_head` in `meatlens_pork_pipeline/modeling.py`:

```python
if head_variant == "training1_mlp_v1":
    features = layers.Dropout(0.30, name="image_dropout")(features)
    features = layers.Dense(128, activation="relu", name="dense_128")(features)
    features = layers.Dropout(0.30, name="dense_dropout")(features)
    return layers.Dense(num_classes, activation="softmax", name="classification_head")(features)
```

Keep `linear_v1` and `mlp_v1` unchanged so the cached baseline and existing package tests retain their current behavior.

- [ ] **Step 3: Make the image sequence apply augmentation only for training**

Update `CsvImageSequence` in `training.py` with `use_augmentation` and `augmentation_preset` parameters. Build the augmentation pipeline once and apply it before the existing MobileNet rescaling:

```python
class CsvImageSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        shuffle: bool = False,
        use_augmentation: bool = False,
        augmentation_preset: str = "geometry_only_v1",
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.use_augmentation = bool(use_augmentation)
        self.augmentation = build_training_augmentation(augmentation_preset)
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]
        images = np.stack(
            [load_image_array(row, target_size=INPUT_SIZE) for row in batch_df.to_dict(orient="records")],
            axis=0,
        ).astype(np.float32)
        if self.use_augmentation:
            images = self.augmentation(images, training=True).numpy()
        labels = [LABEL_ORDER.index(str(row["label"])) for row in batch_df.to_dict(orient="records")]
        return images, tf.keras.utils.to_categorical(labels, num_classes=len(LABEL_ORDER))
```

The implementation must keep validation and test sequences at `use_augmentation=False`.

- [ ] **Step 4: Add validation macro-F1 monitoring and the two-stage trainer**

Implement a callback that predicts the unaugmented validation sequence and writes `logs["val_f1_macro"]`. Add a strategy branch before the existing cached branches in `train_model`:

```python
if training_strategy == ROBOFLOW_CACHED_BASELINE_STRATEGY:
    training_strategy = "cached_embeddings_sgd_v1"

if training_strategy == TRAINING1_COMPATIBLE_STRATEGY:
    train_sequence = CsvImageSequence(
        train_df,
        batch_size=END_TO_END_DEFAULTS["batch_size"],
        shuffle=True,
        use_augmentation=True,
        augmentation_preset="geometry_only_v1",
    )
    val_sequence = CsvImageSequence(
        val_df,
        batch_size=END_TO_END_DEFAULTS["batch_size"],
        shuffle=False,
        use_augmentation=False,
        augmentation_preset="geometry_only_v1",
    )
    model = build_mobilenetv3small_model(
        weights=weights,
        learning_rate=END_TO_END_DEFAULTS["head_lr"],
        head_variant="training1_mlp_v1",
    )
    f1_callback = ValidationMacroF1Callback(val_sequence)
    fit_callbacks = [
        f1_callback,
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_f1_macro",
            mode="max",
            save_best_only=True,
        ),
        callbacks.EarlyStopping(
            monitor="val_f1_macro",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_f1_macro",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        ),
    ]
    class_weights = compute_class_weights(train_df["label"])
    head_history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=8,
        class_weight=class_weights,
        callbacks=fit_callbacks,
        verbose=2,
    )
    backbone = _find_backbone(model)
    backbone.trainable = True
    fine_tune_at = max(int(len(backbone.layers) * 0.75), 1)
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=build_classification_loss(label_smoothing=0.0),
        metrics=["accuracy"],
    )
    fine_history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=20,
        class_weight=class_weights,
        callbacks=fit_callbacks,
        verbose=2,
    )
    history_df = _history_to_frame([("head", head_history), ("fine_tune", fine_history)])
    history_df.to_csv(history_csv_path, index=False)
    best_model = tf.keras.models.load_model(checkpoint_path, compile=False)
    best_model.save(model_h5_path, include_optimizer=False)
    return TrainingArtifacts(
        model_h5_path=model_h5_path,
        checkpoint_path=checkpoint_path,
        history_csv_path=history_csv_path,
        train_count=len(train_df),
        val_count=len(val_df),
        class_weights=class_weights,
        training_strategy=TRAINING1_COMPATIBLE_STRATEGY,
    )
```

The actual implementation must use the existing project imports and helper names; the snippet defines the required behavior and phase boundaries. The cached branch must remain byte-for-byte behavior-compatible at the model/metric interface and must continue accepting `cached_embeddings_sgd_v1` as an alias.

- [ ] **Step 5: Update CLI defaults and choices**

Change both CLI strategy argument declarations to:

```python
parser.add_argument(
    "--training-strategy",
    default="training1_compatible_end_to_end",
    choices=[
        "training1_compatible_end_to_end",
        "roboflow_cached_baseline_v1",
        "cached_embeddings_sgd_v1",
        "cached_embeddings_v1",
        "end_to_end",
    ],
)
```

Set CLI `--epochs-fine` defaults to `20`, `--head-lr` to `5e-4`, and `--fine-tune-lr` to `1e-5` where those options exist.

- [ ] **Step 6: Run focused tests and verify they pass**

Run:

```powershell
conda run -n meatlens-pork-training python -m pytest tests/unit/training/test_training1_compatibility.py tests/unit/cli/test_cli_training_strategy.py tests/unit/training/test_training.py -v
```

Expected: all focused strategy, model, and existing training tests pass.

- [ ] **Step 7: Commit the implementation**

```powershell
git add meatlens_pork_pipeline/config.py meatlens_pork_pipeline/modeling.py meatlens_pork_pipeline/training.py meatlens_pork_pipeline/cli.py tests/unit/training/test_training1_compatibility.py tests/unit/cli/test_cli_training_strategy.py
git commit -m "feat: add training1-compatible Roboflow trainer"
```

### Task 3: Wire the notebook suite to Roboflow and separate strategy outputs

**Files:**
- Modify: `00_shared_setup.ipynb`
- Modify: `04_train_8fold_mobilenetv3small.ipynb`
- Modify: `05_regenerate_metrics_and_reports.ipynb`
- Modify: `06_train_final_deployment_model.ipynb`
- Modify: `07_export_onnx.ipynb`
- Modify: `08_inference_smoke_test.ipynb`
- Test: `tests/integration/training_pipeline/test_strategy_provenance.py`
- Test: `tests/integration/runtime/test_notebook_suite_structure.py`

**Interfaces:**
- Consumes: `generated_splits/roboflow/fold{i}_{train,val,test}.csv`, `DATASET_SOURCE='roboflow'`, and the named training strategy.
- Produces: official end-to-end outputs under a strategy-specific Roboflow root; cached outputs remain under the existing baseline root; deployment/export notebooks consume the official end-to-end root by default.

- [ ] **Step 1: Make the shared notebook contract explicit**

Change the shared setup defaults to:

```python
DATASET_SOURCE = str(override("DATASET_SOURCE", "roboflow")).strip().lower()
TRAINING_STRATEGY = str(
    override("TRAINING_STRATEGY", "training1_compatible_end_to_end")
)
HEAD_LR = float(override("HEAD_LR", 5e-4))
FINE_TUNE_LR = float(override("FINE_TUNE_LR", 1e-5))
EPOCHS_HEAD = int(override("EPOCHS_HEAD", 8))
EPOCHS_FINE = int(override("EPOCHS_FINE", 20))
```

Keep `RUN_SEEDS = [42, 123, 2026]`, `BATCH_SIZE = 32`, `FINE_TUNE_FRACTION = 0.25`, `INPUT_MODE = 'processed_hsv_lab_threshold_roi_224'`, and all existing GPU checks.

- [ ] **Step 2: Separate the 04 output roots and strategy dispatch**

Use these deterministic defaults in notebook 04:

```python
BASE_ROBOFLOW_OUTPUT_ROOT = ROOT / "training_outputs" / "roboflow"
if TRAINING_STRATEGY == "training1_compatible_end_to_end":
    EIGHTFOLD_OUTPUT_ROOT = BASE_ROBOFLOW_OUTPUT_ROOT / "mobilenetv3small_8fold_processed_roi_cnn_only_training1_compatible_end_to_end"
elif TRAINING_STRATEGY == "roboflow_cached_baseline_v1":
    EIGHTFOLD_OUTPUT_ROOT = BASE_ROBOFLOW_OUTPUT_ROOT / "mobilenetv3small_8fold_processed_roi_cnn_only"
else:
    raise ValueError(
        "TRAINING_STRATEGY must be 'training1_compatible_end_to_end' "
        "or 'roboflow_cached_baseline_v1' for the official Roboflow notebook run."
    )
```

Route the fold trainer through `pipeline_train_model` using the selected strategy, translate `roboflow_cached_baseline_v1` to the package alias only at the package boundary, and include `dataset_source`, `training_strategy`, `head_lr`, `fine_tune_lr`, `epochs_head`, `epochs_fine`, and `fine_tune_fraction` in every run’s metrics row and metadata.

The default Roboflow fold selection remains:

```python
default_select_folds = [f"fold{i}" for i in range(1, 9)]
```

- [ ] **Step 3: Make final deployment use generated Roboflow fold 1**

Keep the existing branch in notebook 06 that reads `fold1_train.csv` and `fold1_val.csv` when `DATASET_SOURCE == 'roboflow'`. Change its training call to the selected official strategy and set its default output root to:

```python
TRAINING_OUTPUTS_ROOT / "mobilenetv3small_8samples_final_deployment_cnn_only_training1_compatible_end_to_end"
```

The deployment metadata must contain:

```python
{
    "dataset_source": "roboflow",
    "training_strategy": "training1_compatible_end_to_end",
    "split_source": "generated_splits/roboflow/fold1_train.csv + fold1_val.csv",
    "official_metric_source": "8-fold Roboflow cross-rotation report",
}
```

- [ ] **Step 4: Point export and smoke test at the official deployment namespace**

Update notebook 07 and notebook 08 defaults to the new end-to-end deployment model path. Retain override support so the cached baseline can still be exported or smoke-tested explicitly without changing its files.

- [ ] **Step 5: Run notebook contract and integration tests**

Run:

```powershell
conda run -n meatlens-pork-training python -m pytest tests/integration/data_pipeline tests/integration/training_pipeline tests/integration/runtime/test_notebook_suite_structure.py -v
```

Expected: all Roboflow eight-fold, notebook structure, deployment split, report, and strategy provenance tests pass.

- [ ] **Step 6: Commit notebook wiring**

Before staging, inspect `git status --short` and preserve the user’s unrelated notebook metadata edit in `04_train_8fold_mobilenetv3small.ipynb`. Stage only intentional methodology changes:

```powershell
git add 00_shared_setup.ipynb 04_train_8fold_mobilenetv3small.ipynb 05_regenerate_metrics_and_reports.ipynb 06_train_final_deployment_model.ipynb 07_export_onnx.ipynb 08_inference_smoke_test.ipynb tests/integration/training_pipeline/test_strategy_provenance.py tests/integration/runtime/test_notebook_suite_structure.py
git commit -m "feat: wire Roboflow notebooks to explicit training strategies"
```

### Task 4: Update documentation and artifact provenance

**Files:**
- Modify: `docs/training-pipeline-usage.md`
- Modify: `docs/2026-07-31-official-run-checklist.md`
- Modify: `training_outputs_committable/README.md`
- Test: `tests/integration/training_pipeline/test_notebook_docs.py`

**Interfaces:**
- Consumes: strategy names, output roots, fold contract, and baseline metric provenance from Tasks 1–3.
- Produces: documentation that Chapter III can cite without conflating cached baseline accuracy with the new end-to-end procedure.

- [ ] **Step 1: Add a provenance test**

```python
def test_training_docs_distinguish_official_and_cached_roboflow_paths() -> None:
    text = Path("docs/training-pipeline-usage.md").read_text(encoding="utf-8")
    assert "training1_compatible_end_to_end" in text
    assert "roboflow_cached_baseline_v1" in text
    assert "0.927038626609442" in text
    assert "cached Roboflow baseline" in text
    assert "official evaluation" in text.lower()
```

- [ ] **Step 2: Document the official run order and exact overrides**

Document that the official run uses Roboflow, all eight generated folds, seeds `[42, 123, 2026]`, processed ROI input, geometry-only augmentation, class weights, `HEAD_LR=5e-4`, `FINE_TUNE_LR=1e-5`, two phases, and validation macro-F1 monitoring. Document the cached path separately as the source of the existing 92.7% fold-4/seed-123 baseline.

- [ ] **Step 3: Run the documentation test**

Run:

```powershell
conda run -n meatlens-pork-training python -m pytest tests/integration/training_pipeline/test_notebook_docs.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit documentation**

```powershell
git add docs/training-pipeline-usage.md docs/2026-07-31-official-run-checklist.md training_outputs_committable/README.md tests/integration/training_pipeline/test_notebook_docs.py
git commit -m "docs: record Roboflow strategy and metric provenance"
```

### Task 5: Run the complete verification gate

**Files:**
- Test: entire repository test suite and working-tree diff.

**Interfaces:**
- Consumes: all implementation, notebook, documentation, and test changes from Tasks 1–4.
- Produces: fresh local verification evidence and a clear statement of whether RTX 4050 training was executed.

- [ ] **Step 1: Run the complete pytest suite**

```powershell
conda run -n meatlens-pork-training python -m pytest -q
```

Expected: exit code `0` with zero failures and zero errors.

- [ ] **Step 2: Validate Python syntax and notebook JSON**

```powershell
conda run -n meatlens-pork-training python -m compileall -q meatlens_pork_pipeline tests
conda run -n meatlens-pork-training python -c "import json; from pathlib import Path; [json.loads(p.read_text(encoding='utf-8')) for p in Path('.').glob('*.ipynb')]; print('notebook JSON: PASS')"
```

Expected: both commands exit `0`, and the second prints `notebook JSON: PASS`.

- [ ] **Step 3: Run the repository diff check**

```powershell
git diff --check
```

Expected: no output and exit code `0`.

- [ ] **Step 4: Verify baseline files and staged scope**

```powershell
git status --short
git diff --name-only HEAD~4..HEAD
```

Confirm the baseline metric still equals `0.927038626609442`, both recorded baseline hashes still match Task 1, and no Roboflow image file is staged.

- [ ] **Step 5: Run the official GPU training only when the environment is available**

On the RTX 4050 machine:

```powershell
conda activate meatlens-tf210-gpu
jupyter nbconvert --to notebook --execute 04_train_8fold_mobilenetv3small.ipynb --output 04_train_8fold_mobilenetv3small.executed.ipynb
```

The run is valid only if the notebook reports the RTX 4050, completes all `8 x 3 = 24` fold/seed runs, writes the separate end-to-end strategy root, and the report notebook regenerates metrics from predictions. If this GPU run is not performed, report the local test/syntax state and mark GPU training as unverified.

## Self-Review Checklist

- The Roboflow dataset and generated eight-fold protocol are preserved: Task 3.
- The exact training-1 end-to-end model, augmentation, class weighting, two phases, learning rates, top-25% fine-tune, and F1 monitor are implemented: Task 2.
- The cached 92.7% baseline remains intact and separately named: Tasks 1 and 3.
- Notebook defaults, deployment, export, smoke testing, and reporting are aligned: Task 3.
- Documentation distinguishes baseline accuracy from new end-to-end results: Task 4.
- Full tests, syntax, JSON, diff, and optional GPU execution are verified: Task 5.

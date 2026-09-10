# MeatLens Training Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the end-to-end training hot path with a tested TensorFlow data/input execution path that removes per-batch Python/PIL work and duplicate validation inference while preserving the documented MeatLens experiment.

**Architecture:** `meatlens_pork_pipeline/dataset_pipeline.py` will resolve manifest paths once, build finite `tf.data.Dataset` objects that decode/resize/cache deterministic images before prefetch, and expose stable integer labels. The end-to-end model will own the existing random augmentation layers so training receives fresh augmentation while validation/export run deterministically. A TensorFlow confusion-matrix metric will calculate macro-F1 during Kerasâ€™ single validation traversal, and a callback will write bounded JSONL timing records.

**Tech Stack:** Python 3.10, TensorFlow 2.16.1/Keras, NumPy, Pandas, scikit-learn, pytest, Jupyter notebooks.

## Global Constraints

- Keep `MobileNetV3Small`, 224Ã—224 RGB input, ImageNet initialization, split manifests, label order, class weights, headâ†’fine-tune phases, and `geometry_only_v1` augmentation semantics unchanged by default.
- Keep deterministic Python/NumPy/TensorFlow seeds fixed; `deterministic_ops=True` remains the default and disabling it is an explicit benchmark setting.
- Macro-F1 must match `sklearn.metrics.f1_score(..., average="macro", zero_division=0)` for the same predictions.
- Do not weaken or skip existing tests; run targeted tests after each change and the complete local pytest gate before completion.
- Do not launch the full 24-run campaign as a validation step; CPU/synthetic tests and one-fold notebook checks are the local gate.

## File Map

- Create `meatlens_pork_pipeline/dataset_pipeline.py`: path/label extraction and TensorFlow decode, resize, cache, batch, shuffle, and prefetch construction.
- Modify `meatlens_pork_pipeline/modeling.py`: add a serializable confusion-matrix macro-F1 metric and keep its public model-building contracts intact.
- Modify `meatlens_pork_pipeline/training.py`: use the dataset builder for the training1-compatible strategy, move augmentation into the model graph, make determinism/cache/verbosity/timing configurable, and remove the prediction callbackâ€™s second validation traversal. Keep the legacy `CsvImageSequence` for compatibility.
- Modify `meatlens_pork_pipeline/config.py`: add explicit performance defaults and accepted cache/determinism values.
- Modify `meatlens_pork_pipeline/cli.py`: expose performance options without changing default behavior.
- Modify `04_train_8fold_mobilenetv3small.ipynb`: route the official fold runner through the package options and keep notebook output concise.
- Modify `docs/MEATLENS_TRAINING_PERFORMANCE_OPTIMIZATION_PLAN.md`: record the implementation decisions, evidence limits, and completed verification gates.
- Create or modify `tests/unit/data_pipeline/test_dataset_pipeline.py`, `tests/unit/modeling/test_macro_f1.py`, and `tests/unit/training/test_training.py`: prove data/label parity, metric parity, configuration propagation, and timing output.

### Task 1: Add failing data-pipeline contract tests

**Files:**
- Create: `tests/unit/data_pipeline/test_dataset_pipeline.py`
- Test target: `meatlens_pork_pipeline.dataset_pipeline`

**Interfaces:**
- The tests will define the required `build_image_dataset(df, batch_size, shuffle, cache_mode, seed)` and `dataframe_to_paths_and_labels(df)` contracts for later implementation.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
from PIL import Image

from meatlens_pork_pipeline.dataset_pipeline import build_image_dataset, dataframe_to_paths_and_labels


def test_dataframe_to_paths_and_labels_preserves_row_order_and_label_indices(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (224, 224), (255, 0, 0)).save(first)
    Image.new("RGB", (224, 224), (0, 0, 255)).save(second)
    frame = pd.DataFrame([
        {"local_image_path": str(first), "label": "spoiled"},
        {"local_image_path": str(second), "label": "fresh"},
    ])

    paths, labels = dataframe_to_paths_and_labels(frame)

    assert paths == [str(first), str(second)]
    np.testing.assert_array_equal(labels, [2, 0])


def test_build_image_dataset_returns_static_float_images_and_integer_labels(tmp_path):
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (16, 12), (10, 20, 30)).save(image)
    frame = pd.DataFrame([{"local_image_path": str(image), "label": "fresh"}])

    dataset = build_image_dataset(frame, batch_size=1, shuffle=False, cache_mode="none")
    images, labels = next(iter(dataset))

    assert images.shape == (1, 224, 224, 3)
    assert images.dtype == np.float32
    assert labels.shape == (1,)
    assert labels.dtype == np.int32
    assert int(labels[0]) == 0


def test_build_image_dataset_uses_cache_before_batching_when_requested(tmp_path):
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (224, 224), (10, 20, 30)).save(image)
    frame = pd.DataFrame([{"local_image_path": str(image), "label": "fresh"}])

    dataset = build_image_dataset(frame, batch_size=1, shuffle=False, cache_mode="memory")

    assert "CacheDataset" in type(dataset._input_dataset).__name__
```

- [ ] **Step 2: Run the tests to verify they fail for the missing module**

Run: `pytest tests/unit/data_pipeline/test_dataset_pipeline.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'meatlens_pork_pipeline.dataset_pipeline'`.

- [ ] **Step 3: Implement the minimal data pipeline**

Create `dataframe_to_paths_and_labels` using `resolve_image_path` and `LABEL_ORDER.index`. Implement `_decode_and_resize(path, label)` with `tf.io.read_file`, `tf.io.decode_image(..., channels=3, expand_animations=False)`, `tf.image.resize(..., method="bilinear")` only when needed, `tf.ensure_shape(..., (224, 224, 3))`, and `tf.cast(..., tf.float32)`. Build `Dataset.from_tensor_slices`, map with `num_parallel_calls=tf.data.AUTOTUNE`, apply `cache()` only for `memory` (and `cache(path)` for a `Path`), then shuffle before batch and prefetch with `tf.data.AUTOTUNE`. Reject unsupported cache modes and non-positive batch sizes.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/data_pipeline/test_dataset_pipeline.py -q`

Expected: 3 passed.

- [ ] **Step 5: Commit the coherent data-pipeline change**

Run: `git add meatlens_pork_pipeline/dataset_pipeline.py tests/unit/data_pipeline/test_dataset_pipeline.py && git commit -m "feat(data): add cached tf.data image pipeline"`

### Task 2: Add failing macro-F1 parity tests and metric implementation

**Files:**
- Create: `tests/unit/modeling/test_macro_f1.py`
- Modify: `meatlens_pork_pipeline/modeling.py`

**Interfaces:**
- Produce `MacroF1Metric(num_classes: int = 3, name: str = "f1_macro")`, a Keras metric whose `result()` is numerically equivalent to the sklearn macro-F1 reference.

- [ ] **Step 1: Write the failing parity tests**

```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from meatlens_pork_pipeline.modeling import MacroF1Metric


def test_macro_f1_metric_matches_sklearn_for_missing_and_present_classes():
    y_true = np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)
    y_pred = np.array([0, 1, 1, 2, 0, 2], dtype=np.int32)
    metric = MacroF1Metric(num_classes=3)
    metric.update_state(tf.one_hot(y_true, 3), tf.one_hot(y_pred, 3))

    assert float(metric.result()) == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )


def test_macro_f1_metric_resets_between_validation_passes():
    metric = MacroF1Metric(num_classes=3)
    metric.update_state(tf.one_hot([0, 1, 2], 3), tf.one_hot([0, 1, 2], 3))
    metric.reset_state()

    assert float(metric.result()) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/modeling/test_macro_f1.py -q`

Expected: import failure because `MacroF1Metric` does not exist.

- [ ] **Step 3: Implement the metric**

Add a registered `tf.keras.metrics.Metric` with an `int64` `[num_classes, num_classes]` confusion matrix. Convert one-hot or sparse labels with `argmax` when rank is greater than one, convert predictions with `argmax`, update via `tf.math.confusion_matrix`, and compute per-class precision/recall/F1 with `divide_no_nan`; average all class F1 values so absent classes contribute zero exactly as sklearn does. Implement `get_config` and `reset_state`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/modeling/test_macro_f1.py -q`

Expected: 2 passed.

### Task 3: Add failing training-configuration and timing tests

**Files:**
- Modify: `tests/unit/training/test_training.py`
- Create: `tests/unit/training/test_performance_instrumentation.py`
- Modify: `meatlens_pork_pipeline/config.py`, `meatlens_pork_pipeline/training.py`

**Interfaces:**
- Add `PERFORMANCE_DEFAULTS = {"cache_mode": "memory", "deterministic_ops": True, "verbose": 2}`.
- Add `PerformanceTimingCallback(output_path, train_sample_count, val_sample_count)` that emits one JSON object per epoch with `epoch_total_seconds`, `train_batch_seconds`, `validation_batch_seconds`, and `train_images_per_second`.
- Change `_set_training_seed(seed, deterministic_ops=True)` without changing the default.

- [ ] **Step 1: Write failing tests**

Test that `_set_training_seed(42, deterministic_ops=False)` does not call `enable_op_determinism`, test that the timing callback writes valid JSONL with the required numeric fields after one synthetic epoch, and test that `build_mobilenetv3small_model(weights=None, augmentation=True)` contains the existing four random augmentation layers while `augmentation=False` does not.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/training/test_training.py tests/unit/training/test_performance_instrumentation.py -q`

Expected: failures for the missing callback/configuration and unsupported model argument.

- [ ] **Step 3: Implement the smallest passing changes**

Add the defaults, timing callback hooks (`on_epoch_begin`, batch begin/end hooks, `on_epoch_end`), and deterministic flag. Update `build_mobilenetv3small_model` with `augmentation: bool = False`; when enabled, insert `build_training_augmentation("geometry_only_v1")` before rescaling. Add `MacroF1Metric` to `metrics` only for the new dataset path, preserving the existing public model defaults.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/training/test_training.py tests/unit/training/test_performance_instrumentation.py -q`

Expected: all targeted tests pass.

### Task 4: Replace the training1-compatible hot path

**Files:**
- Modify: `meatlens_pork_pipeline/training.py`
- Modify: `tests/unit/training/test_training.py`

**Interfaces:**
- `train_model(..., cache_mode="memory", deterministic_ops=True, verbose=2, performance_log_path=None)` remains backward-compatible for all existing callers.

- [ ] **Step 1: Add a failing integration-level assertion**

Add a test that monkeypatches `build_image_dataset` and asserts the training1-compatible branch passes the train/validation DataFrames, requested batch size, `shuffle=True` only for train, and the configured cache mode. Also assert the returned history contains `val_f1_macro` from Keras metrics rather than a callback-created second prediction traversal.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/training/test_training.py -q`

Expected: the branch still constructs `CsvImageSequence` and the monkeypatched dataset builder is never called.

- [ ] **Step 3: Implement the training path**

For `TRAINING1_COMPATIBLE_STRATEGY`, build train and validation datasets with the new builder, use `augmentation=True` only in the model for training, compile with `MacroF1Metric`, and call `model.fit` with datasets. Keep the existing class-weight mapping and phase boundaries. Recompile after fine-tuning with the same metric and callbacks monitoring `val_f1_macro`; remove `ValidationMacroF1Callback` from this path. Pass `verbose`, `deterministic_ops`, and performance log path through both phases. Keep `CsvImageSequence` and its callback available for compatibility callers/tests that use them directly.

- [ ] **Step 4: Run targeted tests and the notebookâ€™s CPU integration tests**

Run: `pytest tests/unit/training tests/integration/training_pipeline/test_notebook_8fold_training.py -q`

Expected: targeted training tests and the one-fold notebook tests pass without TensorFlow warning floods.

### Task 5: Expose explicit performance controls in CLI and notebook

**Files:**
- Modify: `meatlens_pork_pipeline/cli.py`
- Modify: `04_train_8fold_mobilenetv3small.ipynb`
- Modify: `tests/unit/cli/test_cli_training_strategy.py`

**Interfaces:**
- CLI train options: `--cache-mode {none,memory}`, `--deterministic-ops/--no-deterministic-ops`, `--verbose {0,1,2}`.
- Notebook overrides: `CACHE_MODE`, `DETERMINISTIC_OPS`, `TRAIN_VERBOSE`, and `PERFORMANCE_LOG_PATH` with defaults matching `PERFORMANCE_DEFAULTS`.

- [ ] **Step 1: Write failing parser/propagation tests**

Parse `train ... --cache-mode none --no-deterministic-ops --verbose 0` and assert the values; monkeypatch `train_model` in the CLI and assert the same values are forwarded.

- [ ] **Step 2: Run the tests to verify failure**

Run: `pytest tests/unit/cli/test_cli_training_strategy.py -q`

Expected: argparse rejects the new options or the forwarded kwargs are absent.

- [ ] **Step 3: Implement CLI/notebook wiring**

Add the options with reproducibility-preserving defaults, pass them only to `train_model`, and update the notebookâ€™s `pipeline_train_model` call and concise progress output. Do not hard-code a GPU-specific performance claim in notebook output.

- [ ] **Step 4: Run parser and notebook structure tests**

Run: `pytest tests/unit/cli tests/integration/runtime/test_notebook_suite_structure.py -q`

Expected: all pass.

### Task 6: Strengthen documentation and run the complete verification gates

**Files:**
- Modify: `docs/MEATLENS_TRAINING_PERFORMANCE_OPTIMIZATION_PLAN.md`
- Modify: `docs/training-pipeline-usage.md` if the new CLI flags are documented there

- [ ] **Step 1: Update the optimization plan**

Record that the implementation uses a model-owned augmentation graph, a single Keras validation traversal via `MacroF1Metric`, in-process decoded-image caching by default, opt-in non-deterministic benchmarking, JSONL timing fields, and an explicit limitation: RTX 4050 throughput, GPU utilization, and mixed-precision benefit require a real GPU benchmark and cannot be claimed from CPU CI.

- [ ] **Step 2: Run static notebook validation and targeted full gates**

Run:

```text
pytest tests/unit -q
pytest tests/integration/runtime tests/integration/data_pipeline tests/integration/training_pipeline -q
python -m compileall meatlens_pork_pipeline
```

Expected: all commands exit 0. If the environment lacks TensorFlow/GPU support, report the exact failed lane and keep CPU-compatible tests green.

- [ ] **Step 3: Inspect the final diff**

Run: `git diff --check; git status --short; git diff --stat`

Confirm that only optimization code/tests/docs are changed and the pre-existing notebook, plan, and log changes are preserved.



# VS Code Notebook Progress Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make long-running `02` and `05` notebook cells visibly active in VS Code without changing model, preprocessing, or reporting results.

**Architecture:** Improve the existing plain progress fallback with bounded time-based heartbeats, make TensorFlow loading lazy in the shared setup notebook, and add concise phase markers to report regeneration. Training and export notebooks retain their explicit TensorFlow imports.

**Tech Stack:** Python 3.10, Jupyter notebooks, TensorFlow, pandas, matplotlib, pytest.

## Global Constraints

- Preserve all model, dataset, and report calculations.
- Keep progress output bounded so the notebook renderer is not overloaded.
- Do not add a required dependency or change VS Code settings.
- Write each behavior test before its implementation.

---

### Task 1: Add bounded heartbeat output to the plain progress fallback

**Files:**
- Modify: `meatlens_pork_pipeline/notebook_progress.py:13-40`
- Test: `tests/integration/runtime/test_notebook_progress.py`

**Interfaces:**
- Consumes: Existing `_PlainProgressBar(total, description, unit)` behavior.
- Produces: The same progress-bar API, with an optional clock and emission interval used only to make time-based behavior testable.

- [ ] **Step 1: Write the failing test**

Add a test that constructs `_PlainProgressBar` with a fake clock, advances an item without reaching the 10% checkpoint, and asserts that a heartbeat is emitted after the configured interval. Also assert that an immediate update does not emit a second heartbeat.

```python
def test_plain_progress_emits_time_heartbeat_without_checkpoint(monkeypatch, capsys) -> None:
    current_time = [0.0]
    progress = notebook_progress._PlainProgressBar(
        total=100,
        description="images",
        unit="image",
        emit_interval_seconds=5.0,
        clock=lambda: current_time[0],
    )
    capsys.readouterr()

    progress.update()
    assert capsys.readouterr().out == ""

    current_time[0] = 5.0
    progress.update()
    output = capsys.readouterr().out
    assert "[RUNNING] images [2/100 image]" in output

    progress.update()
    assert capsys.readouterr().out == ""
```

Import the module in the test so the private fallback can be tested without depending on whether `tqdm` is installed.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/integration/runtime/test_notebook_progress.py::test_plain_progress_emits_time_heartbeat_without_checkpoint -q`

Expected: FAIL because `_PlainProgressBar` does not accept `emit_interval_seconds` or `clock`.

- [ ] **Step 3: Write the minimal implementation**

Add `emit_interval_seconds: float = 5.0` and `clock: Callable[[], float] = perf_counter` to `_PlainProgressBar.__init__`. Store `_last_emitted_at`, and update `_should_emit()` to return true when either the existing item checkpoint is reached or `clock() - _last_emitted_at >= emit_interval_seconds`. Update `_emit()` to refresh `_last_emitted_at`.

Keep the existing checkpoint rules and output format unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/integration/runtime/test_notebook_progress.py::test_plain_progress_emits_time_heartbeat_without_checkpoint -q`

Expected: PASS.

- [ ] **Step 5: Run the existing progress tests**

Run: `pytest tests/integration/runtime/test_notebook_progress.py -q`

Expected: PASS.

### Task 2: Make TensorFlow loading lazy in shared setup

**Files:**
- Modify: `00_shared_setup.ipynb` shared setup code cell
- Test: `tests/integration/runtime/test_notebook_suite_structure.py`

**Interfaces:**
- Consumes: Existing shared `tf` global and `enforce_training_gpu()` behavior.
- Produces: `tf = None` until `get_tensorflow()` or GPU enforcement requires TensorFlow; training callers still receive the TensorFlow module.

- [ ] **Step 1: Write the failing contract test**

Add a test that joins the shared setup code and asserts it defines `get_tensorflow`, initializes `tf` to `None`, and imports TensorFlow inside the lazy loader rather than at module bootstrap.

```python
def test_shared_setup_loads_tensorflow_lazily() -> None:
    notebook = json.loads(Path("00_shared_setup.ipynb").read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "tf = None" in code_source
    assert "def get_tensorflow() -> object:" in code_source
    assert "import tensorflow as tensorflow_module" in code_source
    assert "tf = tensorflow_module" in code_source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/integration/runtime/test_notebook_suite_structure.py::test_shared_setup_loads_tensorflow_lazily -q`

Expected: FAIL because shared setup currently imports TensorFlow immediately and has no `get_tensorflow()` function.

- [ ] **Step 3: Write the minimal notebook implementation**

Replace the eager shared setup import with:

```python
tf = None


def get_tensorflow() -> object:
    global tf
    if tf is None:
        import tensorflow as tensorflow_module

        tf = tensorflow_module
    return tf
```

Update `enforce_training_gpu()` to call `get_tensorflow()` after its test/skip checks and use the returned module for `list_physical_devices`. Keep `set_global_seed()` compatible with the existing explicit TensorFlow imports in training notebooks.

- [ ] **Step 4: Run the contract test to verify it passes**

Run: `pytest tests/integration/runtime/test_notebook_suite_structure.py::test_shared_setup_loads_tensorflow_lazily -q`

Expected: PASS.

- [ ] **Step 5: Run notebook runtime contract tests**

Run: `pytest tests/integration/runtime -q`

Expected: PASS, or an environment-specific TensorFlow collection failure must be reported separately.

### Task 3: Add phase markers to report regeneration

**Files:**
- Modify: `05_regenerate_metrics_and_reports.ipynb` reporting helper and execution cells
- Test: `tests/integration/training_pipeline/test_notebook_reports.py`

**Interfaces:**
- Consumes: Existing `load_prediction_csvs_for_metrics()` and `regenerate_official_reports()` return values.
- Produces: The same report files and data, plus bounded `[REPORT]` phase messages.

- [ ] **Step 1: Write the failing contract test**

Add a source-contract test asserting that the report notebook includes phase messages for loading predictions, summarizing folds, and writing artifacts.

```python
def test_report_notebook_exposes_progress_phase_markers() -> None:
    notebook = json.loads(Path("05_regenerate_metrics_and_reports.ipynb").read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "[REPORT] Loading prediction CSVs" in code_source
    assert "[REPORT] Summarizing folds" in code_source
    assert "[REPORT] Writing report artifacts" in code_source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/integration/training_pipeline/test_notebook_reports.py::test_report_notebook_exposes_progress_phase_markers -q`

Expected: FAIL because the report notebook currently has no phase markers.

- [ ] **Step 3: Write the minimal notebook implementation**

Add one print before and one after each major phase in `regenerate_official_reports()`:

```python
print('[REPORT] Loading prediction CSVs')
metrics_df, all_predictions_df = load_prediction_csvs_for_metrics(seed_metrics_path)
print(f'[REPORT] Loaded {len(all_predictions_df)} predictions')
print('[REPORT] Summarizing folds')
...
print('[REPORT] Writing report artifacts')
...
print('[REPORT] Report artifacts written')
```

Do not alter the DataFrame transformations or output paths.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/integration/training_pipeline/test_notebook_reports.py -q`

Expected: PASS.

### Task 4: Final verification

**Files:**
- Verify: `meatlens_pork_pipeline/notebook_progress.py`, `00_shared_setup.ipynb`, `05_regenerate_metrics_and_reports.ipynb`, and the added tests

- [ ] **Step 1: Run targeted tests**

Run: `pytest tests/integration/runtime/test_notebook_progress.py tests/integration/runtime/test_notebook_suite_structure.py tests/integration/training_pipeline/test_notebook_reports.py -q`

Expected: PASS.

- [ ] **Step 2: Inspect the final diff**

Run: `git diff --check; git diff --stat; git status --short`

Expected: no whitespace errors; only the planned files plus the user’s pre-existing `02_roi_preprocessing.ipynb` execution-count edit.

- [ ] **Step 3: Run the full available test suite**

Run: `pytest -q`

Expected: all available tests pass, or any failure is reported with its exact environment/dependency cause.

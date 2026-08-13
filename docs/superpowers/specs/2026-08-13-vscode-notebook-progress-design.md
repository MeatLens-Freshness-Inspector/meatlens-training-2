# VS Code Notebook Progress and Startup Design

## Problem

Notebook execution is visible in VS Code, but cells in `02_roi_preprocessing.ipynb` and `05_regenerate_metrics_and_reports.ipynb` can appear frozen while the kernel remains busy. The shared setup eagerly imports TensorFlow for notebooks that do not need it, and the fallback progress helper emits only coarse checkpoints.

## Goals

- Avoid importing and initializing TensorFlow in non-training notebooks.
- Give long-running notebook loops a low-volume heartbeat that remains visible in VS Code.
- Add phase-level progress around report regeneration.
- Preserve all model, dataset, and report calculations.
- Keep output bounded so progress reporting does not create a new renderer problem.

## Design

### Lazy TensorFlow loading

The shared setup will expose TensorFlow as a lazy dependency. Non-training notebooks can execute shared setup without importing TensorFlow. Training and export notebooks will continue to import TensorFlow explicitly, and GPU checks will load it only when invoked.

### Progress heartbeat

The plain progress fallback will emit when either a meaningful item checkpoint is reached or a small time interval has elapsed. It will retain final completion output and will not emit once per item. The existing `tqdm` path remains available when installed.

### Report phases

The report notebook will print concise phase markers before and after loading prediction files, summarizing folds, and writing report artifacts. This makes a slow operation distinguishable from a failed one without changing its result.

## Testing

- Add unit coverage for time-based progress emission without sleeping in tests.
- Add notebook contract coverage for lazy TensorFlow setup and report phase markers.
- Run the targeted progress/report tests, then the full test suite available in the environment.

## Non-goals

- No changes to model architecture, training hyperparameters, preprocessing algorithms, or report formulas.
- No forced dependency installation or VS Code settings changes.

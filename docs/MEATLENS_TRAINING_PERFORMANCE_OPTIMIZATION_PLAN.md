# MeatLens Training-2 Performance Optimization Plan

> **Scope:** `04_train_8fold_mobilenetv3small.ipynb` and the supporting `meatlens_pork_pipeline` training stack  
> **Primary objective:** Dramatically reduce MobileNetV3Small training runtime on the RTX 4050 without materially changing the experiment or degrading model quality.

## 1. Problem Statement

The current training implementation produces strong model results, but the execution pipeline is computationally inefficient.

Observed baseline from the interrupted 8-fold training run:

- GPU detected: NVIDIA GeForce RTX 4050 Laptop GPU
- Typical epoch time: approximately **500â€“600 seconds**
- Typical step time: approximately **6â€“7 seconds**
- One completed run consumed approximately **10,173.5 seconds (~2 h 49 min)**
- Strong observed validation results included:
  - Validation accuracy: approximately **94â€“96%**
  - Macro-F1: approximately **93â€“95%**
- The logs contained tens of thousands of repeated TensorFlow warnings involving:
  - `RngReadAndSkip`
  - `Bitcast`
  - `StatelessRandomUniformV2`
  - `ImageProjectiveTransformV3`
- TensorFlow also reported repeated `tf.function` retracing.

The primary issue is therefore **training throughput, not model convergence or model quality**.

---

## 2. Optimization Principle

**Do not nerf the experiment first. Fix the engine first.**

The following experimental properties must initially remain unchanged:

| Experimental property | Requirement |
|---|---|
| Architecture | MobileNetV3Small |
| Input resolution | 224Ã—224 |
| Cross-validation design | Preserve existing design |
| Train/validation splits | Preserve |
| ImageNet initialization | Preserve |
| Training strategy | Head training â†’ fine-tuning |
| Class weighting | Preserve |
| Augmentation semantics | Preserve |
| Primary selection metric | Macro-F1 |
| Early stopping | Preserve |
| LR reduction | Preserve |
| Dataset preprocessing | Preserve image semantics |

Historical baseline training configuration:

- Batch size: 32
- Head epochs: 8
- Fine-tuning epochs: 20
- Head learning rate: `5e-4`
- Fine-tuning learning rate: `1e-5`
- Fine-tuning fraction: 25%
- Augmentation: enabled
- Monitor: `val_f1_macro`

The official run schedule is amended to a reduced-budget protocol after the
performance review:

- Head epochs: 4 maximum
- Fine-tuning epochs: 8 maximum
- Total scheduled epochs per fold-seed run: 12 maximum
- Fold coverage: 8 folds, unchanged
- Seed coverage: 42, 123, and 2026, unchanged

The 8+20 schedule remains the historical baseline for comparison. The reduced
schedule is a declared methodology change and must be reported in the thesis;
results from the two schedules must not be pooled or presented as equivalent.
The validation macro-F1 checkpoint, early stopping, learning-rate reduction,
class weighting, augmentation semantics, split manifests, and deterministic
seeds remain unchanged.

Optimization work must change **how efficiently the experiment executes**, not silently change what experiment is being performed.

---

## 3. Phase 0 â€” Freeze the Baseline

Before modifying the training engine:

1. Preserve the current logs.
2. Preserve existing checkpoints and fold artifacts.
3. Record the current Git commit.
4. Record the TensorFlow, CUDA, cuDNN, Python, and GPU environment.
5. Preserve the current fold/split manifests.
6. Establish baseline timing and ML-quality metrics.

### Baseline performance envelope

```text
Epoch runtime:
~500â€“600 seconds

Strong observed validation performance:
Accuracy: ~94â€“96%
Macro-F1: ~93â€“95%
```

Exact metric equality is **not** required because stochastic augmentation and GPU execution may cause small variations.

The optimized implementation must nevertheless demonstrate comparable convergence and validation quality.

---

## 4. Phase 1 â€” Instrument Before Optimizing

Before replacing components, measure where epoch time is spent.

Add lightweight timing instrumentation around:

```text
image/data loading
augmentation
training batches
Keras validation
macro-F1 calculation
checkpoint saving
total epoch time
```

Where practical, also record:

```text
GPU utilization
GPU memory utilization
GPU power
CPU utilization
images/second
```

### Initial benchmark

Do **not** execute the full cross-validation campaign.

Run:

```text
1 fold
1 head-training epoch
```

Example desired profiler output:

```text
epoch_total          540.2s
training_batches     361.7s
keras_validation      78.4s
macro_f1_validation   79.1s
checkpoint              2.3s
other                  18.7s
```

The purpose is to optimize measured bottlenecks rather than assumptions.

---

## 5. Phase 2 â€” Replace `CsvImageSequence`

### Priority: P0

The current hot path crosses several execution boundaries per batch:

```text
Pandas
  â†“
Python dictionaries/rows
  â†“
PIL
  â†“
NumPy
  â†“
TensorFlow
  â†“
augmentation
  â†“
.numpy()
  â†“
TensorFlow
  â†“
GPU
```

This introduces serial Python work, filesystem work, memory copies, and synchronization.

### Target architecture

Replace the hot-path `CsvImageSequence` implementation with a native `tf.data` pipeline:

```text
DataFrame initialization
        â†“
paths + integer labels
        â†“
tf.data.Dataset.from_tensor_slices()
        â†“
tf.io.read_file()
        â†“
tf.io.decode_jpeg()
        â†“
shape enforcement / resize if required
        â†“
cache
        â†“
shuffle
        â†“
batch
        â†“
augmentation
        â†“
prefetch(tf.data.AUTOTUNE)
        â†“
GPU
```

Use parallel mapping where appropriate:

```python
dataset.map(
    decode_function,
    num_parallel_calls=tf.data.AUTOTUNE,
)
```

Use:

```python
dataset.prefetch(tf.data.AUTOTUNE)
```

### Caching rule

Cache **decoded/preprocessed deterministic images before random augmentation** where memory/storage constraints permit.

This preserves fresh augmentation between epochs while avoiding repeated image decoding.

### Proposed module ownership

Create:

```text
meatlens_pork_pipeline/
â”œâ”€â”€ dataset_pipeline.py
â”œâ”€â”€ training.py
â”œâ”€â”€ augmentation.py
â”œâ”€â”€ modeling.py
â””â”€â”€ ...
```

Responsibilities:

- `dataset_pipeline.py`: image decoding, labels, caching, batching, prefetching
- `augmentation.py`: augmentation definition
- `training.py`: training orchestration
- `modeling.py`: architecture construction

No Pandas/PIL processing should remain in the performance-critical per-batch training loop.

---

## 6. Phase 3 â€” Refactor Augmentation Execution

The existing augmentation semantics are reasonable and should initially remain unchanged:

```text
RandomFlip
RandomRotation
RandomZoom
RandomTranslation
```

The problem is primarily **where and how they execute**.

The previous run generated extremely large numbers of TensorFlow fallback warnings involving random-number and image-transform operations.

### Target

Execute augmentation inside the TensorFlow execution graph rather than through Python batch orchestration.

Preferred architecture:

```text
Input
  â†“
augmentation model
  â†“
MobileNetV3Small
  â†“
classifier
```

Alternative:

```python
dataset.map(
    augment,
    num_parallel_calls=tf.data.AUTOTUNE,
)
```

### Invariant

Optimize:

```text
WHERE augmentation executes
HOW augmentation executes
```

Do **not** initially alter:

```text
WHAT transformations augmentation performs
```

Add tests verifying that output shape, dtype, label association, and allowed transformations remain valid.

---

## 7. Phase 4 â€” Eliminate Duplicate Validation

The existing architecture performs validation through `model.fit(... validation_data=...)` and subsequently performs another validation traversal to calculate macro-F1.

Conceptually:

```text
Training
   â†“
Validation pass #1
   â”œâ”€â”€ val_loss
   â””â”€â”€ val_accuracy

Validation pass #2
   â””â”€â”€ val_f1_macro
```

This duplicates inference and may duplicate image loading.

### Target

Generate validation predictions once per required validation cycle and derive all necessary metrics from the same predictions:

```text
predictions
   â”œâ”€â”€ accuracy
   â”œâ”€â”€ macro-F1
   â”œâ”€â”€ per-class F1
   â”œâ”€â”€ precision
   â”œâ”€â”€ recall
   â””â”€â”€ confusion matrix
```

### Correctness requirement

The optimized macro-F1 calculation must be mathematically equivalent to:

```python
sklearn.metrics.f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)
```

Create explicit unit tests using known `y_true` and `y_pred` arrays.

**No approximate replacement of the thesis metric is acceptable.**

---

## 8. Phase 5 â€” Make Deterministic GPU Operations Configurable

Current training enables deterministic TensorFlow operations.

Retain fixed:

```text
Python seed
NumPy seed
TensorFlow seed
cross-validation splits
dataset manifests
```

But separate these from strict deterministic GPU kernels.

Introduce configuration similar to:

```python
deterministic_ops = True
```

For the pinned native-Windows TensorFlow 2.10 GPU environment, exact
deterministic GPU kernels are not available for the `UnsortedSegmentSum`
gradient used by sparse categorical cross-entropy. Therefore the official
runtime setting is explicitly:

```python
deterministic_ops = False
```

This does not disable experiment control. Python, NumPy, TensorFlow seeds,
split random states, fold assignments, manifests, validation selection, and
test isolation remain fixed. A runtime that supports the required deterministic
kernels may opt in to `True` and record that choice.

Suggested modes on a compatible runtime:

```text
FINAL / REPRODUCIBILITY RUN (compatible runtime)
deterministic_ops = True

DEVELOPMENT / PERFORMANCE BENCHMARK
deterministic_ops = False
```

### Benchmark requirement

Measure both modes.

Do not describe the pinned Windows GPU run as bitwise deterministic when its
required deterministic kernel is unavailable; report the runtime limitation
and preserve the seeded, fixed-manifest experiment controls instead.

Decision rule:

- negligible improvement â†’ retain deterministic mode
- major improvement â†’ document and explicitly decide which mode is scientifically appropriate for the final experiment

---

## 9. Phase 6 â€” Add Optional Mixed Precision

After the new data pipeline is stable and verified, benchmark mixed precision on the RTX 4050.

Example:

```python
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")
```

Where appropriate, force the final probability layer to float32:

```python
Dense(
    len(LABEL_ORDER),
    activation="softmax",
    dtype="float32",
)
```

### Acceptance criteria

Mixed precision remains enabled only if:

```text
No NaN/Inf instability
No meaningful loss instability
No material macro-F1 regression
No material validation-accuracy regression
Meaningful throughput improvement
```

If mixed precision adds complexity without a measurable improvement, revert it.

---

## 10. Phase 7 â€” Batch-Size Throughput Sweep

Only tune batch size after the input pipeline and execution graph have been optimized.

Current baseline:

```text
batch_size = 32
```

Benchmark candidates such as:

```text
32
64
96
128
```

Stop increasing when:

- VRAM becomes insufficient,
- training becomes unstable, or
- images/second no longer improves.

Measure:

```text
images/second
seconds/epoch
peak VRAM
GPU utilization
validation behavior
```

Select the **highest-throughput stable configuration**, not automatically the largest batch.

Because batch size can alter optimization dynamics, any final change requires an ML regression check.

---

## 11. Phase 8 â€” Fix Notebook and Logging Overhead

The interrupted training produced approximately **58,000+ lines of logs**.

Repeated TensorFlow warnings should not dominate the notebook UI.

### Notebook output target

Prefer concise progress:

```text
Fold 2/8 | Head 3/8
loss=.6419 acc=.7156
val_loss=.3913 val_acc=.8841 val_f1=.8680
time=93.4s
```

Detailed diagnostic information should be written to files:

```text
logs/
â”œâ”€â”€ training.log
â””â”€â”€ performance.jsonl
```

Warnings should be investigated and fixed where possible rather than merely hidden.

After known harmless warnings are understood, duplicate console output may be suppressed.

The notebook should remain responsive during long-running experiments.

---

## 12. Phase 9 â€” Escalating Benchmark Gates

Never execute the entire 24-run campaign merely to determine whether an optimization worked.

Use escalating gates:

| Gate | Scope | Purpose |
|---|---|---|
| A | Unit/synthetic tests | Correctness |
| B | ~10 batches | Pipeline sanity |
| C | 1 fold Ã— 1 epoch | Throughput |
| D | 1 fold Ã— 3 epochs | Convergence sanity |
| E | 1 complete fold | ML equivalence |
| F | 2 complete folds | Variance sanity |
| G | Full experiment | Final results |

### Performance targets

Current:

```text
~550 seconds/epoch
```

Initial targets:

```text
Minimum acceptable: <180 s/epoch
Good:               <120 s/epoch
Excellent:           <90 s/epoch
Stretch:             <60 s/epoch
```

These are engineering targets, **not guaranteed results**.

If a refactor only produces:

```text
550s â†’ 480s
```

the primary bottleneck has probably not been eliminated.

Profile again instead of launching the complete experiment.

---

## 13. Phase 10 â€” ML Regression Gate

Before authorizing the full experiment, compare the baseline and optimized pipeline using equivalent fold conditions.

### Quality gate

The optimized pipeline should show:

```text
Macro-F1:
No material regression; investigate changes > ~1 percentage point

Validation accuracy:
Same general performance range

Training curves:
Comparable convergence behavior

Per-class recall:
No catastrophic class-specific regression

Confusion matrix:
No new systematic failure pattern
```

### Experimental equivalence gate

Verify:

```text
old split == new split
old preprocessing == new preprocessing
old label mapping == new label mapping
old architecture == new architecture
old augmentation semantics â‰ˆ new augmentation semantics
old class weighting == new class weighting
```

This allows the optimization to be described academically as an implementation-level performance improvement rather than an undocumented methodological change.

Suggested manuscript framing:

> The training implementation was optimized for computational efficiency while retaining the dataset, preprocessing methodology, model architecture, validation protocol, class weighting, augmentation strategy, and experimental hyperparameters.

---

## 14. Phase 11 â€” Full Experiment Authorization

The complete training campaign is authorized only after Gate F passes.

Execution sequence:

```text
optimized training pipeline
        â†“
cross-validation campaign
        â†“
aggregate fold/run metrics
        â†“
mean Â± standard deviation
        â†“
per-class metrics
        â†“
confusion matrices
        â†“
model selection
        â†“
final deployment-model training
```

No full-scale training run should begin while a known major performance regression remains unresolved.

---

## 15. Planned Commit Sequence

Keep changes small, reviewable, testable, and bisectable.

```text
1.  test(training): add baseline pipeline equivalence tests

2.  perf(training): add epoch and pipeline instrumentation

3.  feat(data): introduce tf.data image pipeline

4.  refactor(training): replace CsvImageSequence hot path

5.  refactor(augmentation): move augmentation into TensorFlow graph

6.  test(metrics): verify macro-F1 parity with sklearn

7.  perf(validation): eliminate duplicate validation inference

8.  feat(training): make deterministic ops configurable

9.  perf(training): enable optional mixed precision

10. perf(training): add configurable batch-size benchmarking

11. perf(notebook): suppress duplicate TensorFlow warning spam

12. test(training): add optimized-vs-baseline regression suite

13. docs(training): document performance methodology

14. perf(training): benchmark RTX 4050 optimized pipeline
```

After these commits:

**Stop feature work and benchmark before proceeding.**

---

## 16. Definition of Done

The optimization effort is complete only when all applicable conditions below are satisfied.

### Correctness

- [ ] Dataset samples and labels remain correctly associated.
- [ ] Existing split manifests remain unchanged.
- [ ] Image preprocessing semantics remain unchanged.
- [ ] Augmentation semantics remain equivalent.
- [ ] Macro-F1 matches sklearn reference calculations.
- [ ] Class weighting remains correct.
- [ ] Checkpoint selection remains based on the intended metric.
- [ ] Early stopping behavior remains correct.
- [ ] Fine-tuning unfreezes the intended backbone layers.
- [ ] No NaN/Inf training instability occurs.

### Performance

- [ ] Per-stage timing instrumentation exists.
- [ ] Python/PIL/Pandas work is removed from the hot per-batch path.
- [ ] Input decoding is parallelized.
- [ ] Prefetching is enabled.
- [ ] Appropriate deterministic preprocessing is cached.
- [ ] Repeated `.numpy()` synchronization is removed from the training hot path.
- [ ] Duplicate validation inference is eliminated or justified.
- [ ] TensorFlow retracing warnings are resolved or understood.
- [ ] Augmentation fallback-warning explosion is resolved or understood.
- [ ] GPU utilization is measured.
- [ ] Images/second is measured.
- [ ] Epoch runtime improves substantially over baseline.

### ML Quality

- [ ] One complete optimized fold converges normally.
- [ ] Two-fold regression gate passes.
- [ ] Validation accuracy remains within the expected range.
- [ ] Macro-F1 shows no material unexplained regression.
- [ ] Per-class recall remains acceptable.
- [ ] Confusion matrices show no new systematic failure mode.

### Reproducibility

- [ ] Random seeds are explicitly controlled.
- [ ] Deterministic-operation behavior is documented.
- [ ] Software/hardware environment is recorded.
- [ ] Performance configuration is recorded.
- [ ] Final experiment configuration is version-controlled.

### Operations

- [ ] Notebook remains responsive during training.
- [ ] Console output is concise.
- [ ] Detailed logs are persisted separately.
- [ ] Checkpoints survive interrupted training.
- [ ] Failed/interrupted runs do not silently overwrite valid artifacts.
- [ ] Full experiment can be resumed or restarted predictably.

---

## 17. Benchmark Decision Rule

Every expensive training execution must answer a predefined question.

Examples:

```text
Does tf.data materially improve images/second?
Does graph-based augmentation eliminate retracing?
Does removing duplicate validation reduce epoch time?
Does mixed precision improve RTX 4050 throughput?
Does batch size 64 outperform batch size 32?
Does optimized training preserve macro-F1?
```

Do **not** run multi-hour experiments merely to "see what happens."

Example success:

```text
BEFORE
88 steps
550 sec/epoch
6.25 sec/step

AFTER
88 steps
70 sec/epoch
0.80 sec/step
```

â†’ Continue optimization/validation.

Example insufficient result:

```text
550 sec/epoch
â†“
480 sec/epoch
```

â†’ Profile again. Do not launch the full experiment.

---

## 18. Final Engineering Rule

The current MobileNetV3Small experiment has already demonstrated that it can reach strong validation accuracy and macro-F1.

Therefore:

> **The model is not currently the primary optimization target. The training infrastructure is.**

The next objective is to make the RTX 4050 receive data and execute the training graph efficiently while preserving the scientific validity of the MeatLens experiment.

**Fix the engine. Prove equivalence. Benchmark one fold. Then launch the full campaign.**

## 19. Implementation Decisions and Evidence Boundaries

The approved implementation follows the existing package boundaries instead of duplicating the training engine inside the notebook:

- `meatlens_pork_pipeline.dataset_pipeline` owns path resolution, TensorFlow decoding/resizing, optional decoded-image caching, batching, shuffling, and prefetching.
- The end-to-end Keras model owns the existing `geometry_only_v1` random augmentation layers. They run only when `training=True`, so validation and exported inference remain unaugmented while training receives fresh augmentation after deterministic input caching.
- Macro-F1 is a Keras confusion-matrix metric named `f1_macro`. Keras computes it during its normal validation traversal, removing the callback-driven second prediction pass. Its parity contract is the sklearn reference with `average="macro"` and `zero_division=0`.
- Deterministic GPU operations are disabled by default for the pinned native-Windows TensorFlow 2.10 environment because the required `UnsortedSegmentSum` gradient kernel is unavailable. This does not change the Python, NumPy, TensorFlow, split, or manifest seeds; a compatible runtime may explicitly opt in.
- Timing is emitted as bounded JSONL records with epoch total, train-batch, validation-batch, and images/second fields. GPU utilization, power, and mixed-precision benefit remain benchmark claims until measured on the RTX 4050 environment.
- The legacy `CsvImageSequence` remains available for compatibility, but the official training1-compatible strategy uses the TensorFlow dataset path.

The implementation gate is deliberately split into correctness, one-fold CPU sanity, and real-GPU performance measurement. No full 24-run campaign is authorized by local tests alone, and no CPU result will be presented as an RTX 4050 throughput result.

## 20. Implementation Status

- [x] TensorFlow decode/resize/cache/batch/prefetch pipeline added and covered by unit tests.
- [x] Streaming macro-F1 added with sklearn parity tests.
- [x] Official training1-compatible strategy switched away from `CsvImageSequence`.
- [x] Augmentation moved into the Keras training graph while remaining disabled during validation/inference.
- [x] Deterministic operations, cache mode, and concise verbosity are explicit controls with reproducibility-preserving defaults.
- [x] Per-epoch JSONL timing instrumentation added.
- [x] Notebook progress-cell hygiene and performance-control wiring validated.
- [x] CPU unit/integration gates completed; an RTX 4050 throughput benchmark and mixed-precision decision remain intentionally pending until the GPU environment is run.

## 21. Reduced-Budget Methodology Amendment

The official Training 2 protocol uses 4 head-training epochs and 8
fine-tuning epochs as maximum budgets. This reduces scheduled optimization
work from 28 to 12 epochs per fold-seed run while retaining the complete
eight-fold, three-seed evaluation design. The change is motivated by the
five-hour operational target and is guarded by the existing validation
macro-F1 checkpointing and early stopping callbacks.

The reduced schedule is scientifically defensible only if the completed run
provides traceable evidence: every run records its configured and observed
epoch counts, its best validation macro-F1, its checkpoint path, and its test
predictions. Before interpreting the results, compare the reduced schedule's
aggregate and per-class metrics against the historical 8+20 baseline where
artifacts exist, and report any material difference as a limitation rather
than selecting a favorable schedule after seeing test results.

The official claim is therefore limited to the reduced-budget protocol. The
historical 8+20 results remain a comparator and are not silently overwritten.


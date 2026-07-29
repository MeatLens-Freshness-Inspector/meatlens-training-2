# MobileNetV3-Small Procedure Improvements Design

## Goal

Improve the current notebook-first `MobileNetV3Small` pork-freshness training pipeline so it has a better chance of reaching stronger real-world performance without expanding into multiple backbones or unrelated research branches.

This design locks the model family to `MobileNetV3Small` and focuses on training-procedure improvements plus one controlled input comparison:

- current `processed_hsv_lab_threshold_roi_224`
- one less-processed input branch for comparison

The required deployment outcome remains unchanged:

- final trained Keras model
- ONNX export
- ONNX smoke-test verification

## Why This Pass

The recommendation document in [MeatLens_Training_Procedure_Recommendations_COMPLETE.md](C:/Users/Adriaan%20M.%20Dimate/Desktop/development/school/meatlens-training-2/docs/MeatLens_Training_Procedure_Recommendations_COMPLETE.md) identifies several likely bottlenecks:

- augmentation may not be clearly wired into the actual training path
- processed ROI may be hiding useful signal or introducing artifacts
- model reporting should emphasize worst-fold and severe-error behavior, not only average metrics
- final validation for deployment should respect physical-sample boundaries
- `MobileNetV3Small` should be improved before exploring more architectures

Rather than taking on every recommendation at once, this pass should answer the highest-value questions while keeping the project operational and thesis-friendly.

## Scope

This pass includes:

- keeping only `MobileNetV3Small`
- tightening the training augmentation path
- making the training pipeline support two input modes
- preserving sample-level cross-rotation evaluation
- making final deployment validation sample-aware
- adding stronger evaluation summaries
- exposing clean fine-tuning modes within the existing notebook suite

This pass does not include:

- new backbones such as `EfficientNetB0`, `ResNet50`, or `MobileNetV2`
- ordinal loss
- cost-sensitive loss
- hard-example mining
- confidence calibration
- temporal smoothing
- dual-input raw-plus-ROI architecture

Those items remain valid future experiments, but they are intentionally deferred to avoid overloading this cycle.

## Chosen Design

The chosen design is:

- keep the notebook-first workflow
- keep `MobileNetV3Small` as the only model family
- compare input handling, not backbone families
- improve the reliability of the current training and reporting contract

This is the smallest design that still addresses the strongest recommendations from the new training-procedure notes.

## Existing Context

The repo already contains:

- a notebook-first suite from `00` through `08`
- official `8-fold` sample-held-out cross rotation
- metrics for `accuracy`, macro `precision`, macro `recall`, and macro `F1`
- confusion matrix outputs
- ONNX export and smoke test notebooks
- a Windows-native RTX 4050 TensorFlow environment path

The current defaults are centered around:

- `processed_hsv_lab_threshold_roi_224`
- `MobileNetV3Small`
- `cnn_only`
- frozen-head training followed by partial fine-tuning

The approved improvement pass should build on that structure rather than replacing it.

## Codebase Architecture

The notebook suite remains the main interface:

- `00_shared_setup.ipynb`
- `01_manifest_and_dataset_audit.ipynb`
- `02_roi_preprocessing.ipynb`
- `03_build_cross_rotation_splits.ipynb`
- `04_train_8fold_mobilenetv3small.ipynb`
- `05_regenerate_metrics_and_reports.ipynb`
- `06_train_final_deployment_model.ipynb`
- `07_export_onnx.ipynb`
- `08_inference_smoke_test.ipynb`

This pass extends that architecture rather than introducing a parallel training system.

The main structural additions are:

- shared configuration knobs for input mode, augmentation preset, and fine-tune fraction
- a second processed-manifest path or parallel manifest flow for the less-processed input branch
- extra report outputs for severe errors, worst fold, and variance

## Workflow Architecture

The workflow stays notebook-first and linear:

1. `00_shared_setup.ipynb`
2. `01_manifest_and_dataset_audit.ipynb`
3. `02_roi_preprocessing.ipynb`
4. `03_build_cross_rotation_splits.ipynb`
5. `04_train_8fold_mobilenetv3small.ipynb`
6. `05_regenerate_metrics_and_reports.ipynb`
7. `06_train_final_deployment_model.ipynb`
8. `07_export_onnx.ipynb`
9. `08_inference_smoke_test.ipynb`

The workflow changes in this pass are:

- preprocessing can produce either the current processed ROI branch or a less-processed branch
- 8-fold training can select the branch through configuration
- report regeneration produces more decision-oriented outputs
- final deployment uses sample-aware validation selection rather than a random image split

## Input Strategy

This pass introduces two supported input modes for `MobileNetV3Small`:

1. `processed_hsv_lab_threshold_roi_224`
2. `raw_center_crop_224`

The purpose is not to redesign the model. The purpose is to compare whether the current thresholded ROI preprocessing is helping or hiding important freshness cues.

The less-processed branch should:

- preserve more natural color and texture information
- avoid HSV/LAB threshold masking artifacts
- remain deployment-feasible
- keep the same final image size contract of `224x224`

This branch should be generated and tracked through notebook outputs in the same way as the existing processed ROI branch so comparisons stay fair.

## Training Architecture

The backbone remains:

- `MobileNetV3Small`

The model mode remains:

- `cnn_only`

The output remains:

- `3-class softmax`

The primary architectural variable in this pass is input mode, not model family.

## Augmentation Design

This pass should make augmentation explicit, conservative, and training-only.

Recommended augmentation family:

- horizontal flip
- small rotation
- small zoom
- small translation
- mild contrast adjustment
- mild brightness adjustment

Constraints:

- no aggressive hue shifts
- no aggressive saturation changes
- no validation or test augmentation
- augmentation should happen before final `MobileNetV3` preprocessing

The design goal is to simulate small camera and positioning variation without destroying real freshness evidence.

## Fine-Tuning Design

The current partial fine-tuning approach should become an explicit experimental knob rather than a hidden constant.

This pass should support:

- `0.0` frozen backbone
- `0.25` partial fine-tuning
- `1.0` full fine-tuning

The default should remain `0.25` to preserve continuity with the current pipeline.

When `1.0` is selected, the learning rate should remain very low and the reporting should make it easy to compare stability across folds.

## Validation And Split Design

Official evaluation should continue to use strict sample-held-out cross rotation.

This remains one of the strongest methodological parts of the current repo and should not be weakened.

For deployment training, this pass changes the validation design:

- current behavior uses random image-level stratified validation
- new behavior should hold out entire physical samples for validation

The deployment notebook should therefore split on `sample_id`, not only on image rows, while still preserving label coverage as well as the available data allows.

The validation set in deployment training is for model selection and early stopping only. It should still respect the same physical-sample boundary principle as the official evaluation workflow.

## Reporting Design

The current required metrics stay the same:

- `accuracy`
- macro `precision`
- macro `recall`
- macro `F1`
- confusion matrix

This pass adds decision-oriented summary metrics:

- mean macro `F1` across runs
- standard deviation of macro `F1`
- worst-fold macro `F1`
- severe-error rate

Severe-error rate is defined for this pass as:

- `fresh -> spoiled`
- `spoiled -> fresh`

These summaries should be saved alongside the existing fold and seed outputs so the team can compare procedures rather than cherry-picking a single lucky run.

## Notebook Responsibilities

### `00_shared_setup.ipynb`

Add shared configuration for:

- `INPUT_MODE`
- input-root or manifest-path selection
- augmentation preset
- fine-tune fraction or mode
- severe-error label pairs

This notebook should remain the single source of truth for training defaults.

### `02_roi_preprocessing.ipynb`

Extend preprocessing to support:

- current processed ROI generation
- less-processed `raw_center_crop_224` generation

Outputs should remain explicit and reproducible so downstream notebooks can point to either branch by configuration.

### `03_build_cross_rotation_splits.ipynb`

Keep sample-level rotation behavior unchanged.

If manifest structure is shared across input modes, the split builder should work identically for either branch as long as `sample_id`, `label`, and `local_image_path` are available.

### `04_train_8fold_mobilenetv3small.ipynb`

This notebook should become the main implementation target for:

- explicit augmentation policy
- input-mode switching
- fine-tuning mode switching
- per-run severe-error accounting
- fold-level artifact naming that clearly captures the chosen procedure

### `05_regenerate_metrics_and_reports.ipynb`

Extend aggregation to produce:

- worst-fold summaries
- variance summaries
- severe-error summaries
- procedure-comparison friendly CSV outputs

### `06_train_final_deployment_model.ipynb`

Change the validation split from random image-level stratification to sample-aware selection while preserving the same final deployment artifact flow.

### `07_export_onnx.ipynb`

Keep ONNX export behavior stable.

The exported metadata should reflect the chosen input mode so deployment artifacts stay traceable.

### `08_inference_smoke_test.ipynb`

Keep the smoke test stable and verify that the exported artifact still matches the expected input and output contract.

## Data Contract

To support controlled input comparison, the manifest contract for downstream training should stay stable regardless of input mode:

- `image_file_name`
- `label`
- `sample_number`
- `sample_id`
- `local_image_path`

This allows the same split logic, training logic, and reporting logic to operate across both input branches.

The input-mode comparison should therefore be implemented through path generation and preprocessing outputs, not through incompatible downstream schemas.

## Artifact Contract

This pass should preserve the existing artifact expectations:

- per-fold metrics CSV
- per-image predictions CSV
- confusion matrix CSV
- confusion matrix PNG
- trained Keras model
- deployment metadata JSON
- ONNX model
- ONNX smoke-test summary

It should additionally produce:

- severe-error summary CSV
- worst-fold summary CSV
- procedure summary CSV with mean and standard deviation

## Testing Strategy

This pass should add or update tests for:

- augmentation being applied only to training batches
- input-mode switching between processed ROI and raw center crop
- report regeneration including severe-error and worst-fold outputs
- final deployment split respecting sample boundaries

Notebook execution tests should continue to use the current test harness so the behavior remains verifiable in CI-like local runs.

## Risks And Mitigations

### Risk: Input-mode expansion bloats the workflow

Mitigation:

- keep only one additional branch
- keep the same manifest schema
- keep `MobileNetV3Small` fixed

### Risk: Full fine-tuning increases instability

Mitigation:

- keep `0.25` as the default
- report variance and worst-fold behavior
- use a very low learning rate for full fine-tuning

### Risk: Sample-aware deployment validation may reduce validation size flexibility

Mitigation:

- split on `sample_id`
- document the exact rule
- prefer methodological correctness over superficially balanced random image splits

### Risk: More metrics create more files but not better decisions

Mitigation:

- only add metrics that directly influence procedure selection
- keep severe errors and worst-fold behavior as the main additions

## Success Criteria

This improvement pass is successful when:

- the notebook suite still trains only `MobileNetV3Small`
- augmentation is explicitly and verifiably part of the training path
- the pipeline can compare `processed_hsv_lab_threshold_roi_224` against `raw_center_crop_224`
- final deployment validation respects physical-sample boundaries
- reports include worst-fold and severe-error behavior in addition to the required core metrics
- ONNX export remains intact

## Out Of Scope Follow-Ups

After this pass is stable, the next most defensible follow-ups are:

- cost-sensitive loss
- ordinal loss
- hard-example weighting
- confidence calibration

Those should be treated as separate, controlled experiment passes rather than mixed into this implementation cycle.

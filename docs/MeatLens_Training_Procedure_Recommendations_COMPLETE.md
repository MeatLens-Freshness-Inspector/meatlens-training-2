# MeatLens Training Procedure Recommendations (Complete)

## Executive Summary

The current MeatLens training pipeline is already stronger than a basic image-classification workflow. It uses sample-level cross-rotation folds, multiple seeds, macro-F1 monitoring, class weights, staged fine-tuning, transition-aware evaluation, and performance breakdowns by pork cut, phone group, and capture source.

However, the next gains should come from **improving the training procedure**, not from purchasing and capturing substantially more meat. The main concerns are augmentation that appears enabled but not actually applied, excessive reliance on correlated frames from a small number of physical meat samples, possible information loss from processed ROI inputs, a loss function that does not reflect freshness ordering, and overly optimistic model selection from many runs.

---

## 1. Ensure Training Augmentation Is Actually Applied

The CNN-only library declares:

```python
USE_TRAINING_AUGMENTATION = True
```

However, the `ImageOnlySequence` path appears to load the processed ROI, run MobileNetV3 preprocessing, and stack the images without applying augmentation.

This makes augmentation the most immediate no-new-data improvement.

### Recommended conservative augmentation

```python
training_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),
    layers.RandomZoom(
        height_factor=(-0.08, 0.08),
        width_factor=(-0.08, 0.08),
    ),
    layers.RandomTranslation(
        height_factor=0.04,
        width_factor=0.04,
    ),
    layers.RandomContrast(0.10),
], name="training_augmentation")
```

Apply it only to training batches:

```python
class ImageOnlySequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        image_paths,
        labels=None,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False,
    ):
        self.image_paths = list(image_paths)
        self.labels = None if labels is None else np.asarray(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __getitem__(self, index):
        # Existing loading code...
        batch_images = build_image_array_from_paths(batch_paths)

        if self.augment:
            batch_images = training_augmentation(
                batch_images,
                training=True,
            ).numpy()

        return batch_images, batch_labels
```

The exact placement depends on the image range returned by the loader, but augmentation should generally occur before final MobileNet preprocessing.

### Important constraint

Do not use aggressive hue or saturation augmentation by default. MeatLens is learning deterioration partly through color, so heavy color changes may alter the class evidence rather than simulate realistic camera variation.

---

## 2. Stop Increasing the Number of Frames per Interval

More images from the same physical meat piece increase frame count but not biological diversity.

The effective number of independent experimental units is closer to the number of physical meat samples than the total number of captured frames.

Capturing more near-adjacent frames can:

- increase storage and training time;
- create highly correlated examples;
- encourage memorization of texture, plate, lighting, or segmentation artifacts;
- inflate apparent dataset size without equivalent generalization gains.

Run a controlled ablation:

| Variant | Images per interval |
|---|---:|
| A | 2 |
| B | 4 |
| C | 6 |
| D | 8 |

Use identical folds, seeds, architecture, and hyperparameters. Compare mean macro-F1, worst-fold macro-F1, severe-error rate, and variance.

If four or six frames perform within approximately one standard deviation of eight frames, use the smaller configuration.

---

## 3. Preserve Sample-Level Train, Validation, and Test Separation

The existing sample-held-out split is one of the strongest methodological decisions in the repository.

Do not replace it with a random 80/10/10 image split. Random image splitting can place adjacent images from the same physical meat sample in both training and testing, causing leakage and inflated performance.

Recommended structure:

- **Test:** one completely held-out physical meat sample;
- **Validation:** another completely held-out physical meat sample;
- **Training:** the remaining physical samples.

Validation should also be sample-held-out. Randomly choosing validation images from training meat samples allows early stopping and learning-rate scheduling to adapt to those same physical samples.

---

## 4. Treat Processed ROI as an Experimental Branch, Not an Assumption

The current final CNN-only configuration uses a preprocessed HSV/LAB-thresholded 224×224 ROI.

That may improve background removal, but it may also:

- remove subtle discoloration;
- create threshold-shaped artifacts;
- make segmentation sensitive to lighting;
- teach the classifier preprocessing behavior instead of meat deterioration;
- fail when deployment images produce different masks.

Run the same folds and seeds across:

1. raw center crop;
2. bounding-box crop without color thresholding;
3. masked HSV/LAB ROI;
4. optional raw-plus-ROI dual branch, only if deployment cost remains acceptable.

Evaluate:

- mean sample-held-out macro-F1;
- worst-fold macro-F1;
- fresh-to-spoiled and spoiled-to-fresh error rate;
- fold-to-fold variance;
- performance by phone and capture source;
- model size and inference speed.

Do not choose the processed ROI merely because it has the highest overall accuracy. A model with slightly lower average F1 but much lower severe-error rate and variance may be safer.

---

## 5. Train Freshness as an Ordered Problem

The classes follow a real progression:

```text
fresh → not fresh → spoiled
```

Ordinary categorical cross-entropy treats all mistakes equally. It does not distinguish between:

- fresh predicted as not fresh; and
- fresh predicted as spoiled.

The transition-aware evaluation recognizes the ordering, but the training loss does not.

### Option A: Ordinal cumulative targets

Use two binary outputs:

```text
Output 1: Is the sample at least not fresh?
Output 2: Is the sample spoiled?
```

Targets:

```text
fresh      = [0, 0]
not fresh  = [1, 0]
spoiled    = [1, 1]
```

This explicitly teaches the natural progression.

### Option B: Cost-sensitive categorical loss

Keep the three-class softmax but penalize severe errors more heavily than adjacent errors.

Example principle:

```text
fresh ↔ not fresh: lower penalty
not fresh ↔ spoiled: lower penalty
fresh ↔ spoiled: highest penalty
```

The lower-risk first experiment is cost-sensitive categorical loss because it preserves the existing output format.

---

## 6. Use Hard-Example Mining on Existing Images

The repository already saves per-image predictions, confidence values, metadata, and confusion matrices. Use those outputs instead of buying more meat.

After each full cross-validation cycle, identify:

- confidently wrong predictions;
- fresh predicted as spoiled;
- spoiled predicted as fresh;
- images whose predicted class changes across seeds;
- images misclassified by several models;
- low-confidence images around transition boundaries.

Increase their training weight in the next cycle.

Example:

```python
sample_weight = 1.0

if severe_previous_error:
    sample_weight = 2.5
elif unstable_across_seeds:
    sample_weight = 1.75
elif transition_boundary:
    sample_weight = 1.5
```

Do not physically duplicate hard images into every split. Use sample weights or a weighted sampler while preserving physical-sample separation.

---

## 7. Add Temporal Consistency as a Separate Experiment

Each meat sample represents a time progression. Predictions should generally move:

```text
fresh → not fresh → spoiled
```

Repeated backward movement such as:

```text
spoiled → fresh → spoiled
```

is usually implausible.

For sequential observations, test temporal smoothing as a secondary condition:

```python
smoothed[t] = (
    0.20 * probability[t - 1]
    + 0.60 * probability[t]
    + 0.20 * probability[t + 1]
)
```

Keep independent single-image prediction as the primary result. Temporal smoothing should be reported separately because it is only available when repeated observations of the same meat batch exist.

---

## 8. Change the Final Model Selection Procedure

The experiment matrix contains multiple folds and seeds. Selecting the single run with the highest macro-F1 is optimistic because the winning model may simply be lucky.

### For thesis reporting

Report:

```text
mean ± standard deviation across folds and seeds
```

Also report:

- worst-fold macro-F1;
- severe-error rate;
- confusion matrices;
- per-sample performance;
- per-device or capture-source performance.

### For deployment

After locking the procedure:

1. freeze architecture and hyperparameters;
2. use a predefined seed rather than choosing the best seed afterward;
3. train one final model using all eligible development samples;
4. retain a later field dataset or untouched sample for external validation when possible.

Choose the best **procedure**, not the luckiest checkpoint.

---

## 9. Reduce Architecture Exploration

The repository already contains many related experiments: hybrid, CNN-only, segmented, processed ROI, different MobileNet versions, EfficientNet, conversion, and inference variants.

Adding many more backbones creates research debt and a multiple-comparison problem.

Limit final architecture comparisons to:

- MobileNetV2 as the baseline;
- MobileNetV3-Small as the proposed mobile model;
- one heavier reference model, such as EfficientNetB0 or ResNet50.

Spend the remaining compute budget on experiments that answer meaningful questions:

- augmentation;
- raw versus processed input;
- ordinal versus categorical loss;
- interval frame count;
- confidence calibration;
- hard-example weighting.

---

## 10. Run a Small, Controlled Fine-Tuning Study

The current setup is already reasonable:

- frozen head training;
- unfreezing the top 25%;
- low learning rate;
- early stopping on validation macro-F1;
- class weighting;
- ReduceLROnPlateau.

Compare only three settings:

```text
0%   — frozen backbone baseline
25%  — current partial fine-tuning
100% — full fine-tuning at approximately 1e-6 to 3e-6
```

Because the domain differs substantially from ImageNet, full low-rate fine-tuning may help. However, the small number of independent meat samples also makes overfitting likely. Cross-rotation variance should determine whether it is worthwhile.

Do not prioritize progressive image-size training yet. Fix augmentation and input preprocessing first.

---

# Recommended New Training Procedure

## Phase A — Lock and Audit the Dataset

- Do not purchase additional meat for the next experimental cycle.
- Audit timestamps and freshness labels.
- Remove corrupt images.
- Detect exact and near duplicates.
- Preserve physical-sample train/validation/test separation.
- Compare 4, 6, and 8 interval frames, with 2 as an optional low-data baseline.

## Phase B — Input Ablation

Using identical folds and seeds, compare:

- raw crop;
- unthresholded meat ROI;
- HSV/LAB processed ROI.

## Phase C — Augmentation Ablation

Using the best input method, compare:

- no augmentation;
- geometry and positioning augmentation;
- geometry plus mild brightness/contrast augmentation.

Avoid aggressive color shifts.

## Phase D — Objective Ablation

Compare:

- ordinary categorical cross-entropy;
- cost-sensitive categorical loss;
- ordinal formulation.

## Phase E — Hard-Example Retraining

- identify severe and unstable mistakes;
- apply sample weighting;
- rerun the locked folds and seeds;
- verify that improvements are consistent rather than limited to one fold.

## Phase F — Final Procedure Selection

Rank procedures using:

1. mean macro-F1;
2. severe-error rate;
3. worst-fold macro-F1;
4. standard deviation across folds and seeds;
5. per-device and capture-source robustness;
6. model size;
7. inference speed.

---

# Priority Order

## Highest Priority

1. Wire augmentation into the actual training loader.
2. Verify strict sample-level validation.
3. Compare raw/unthresholded input against processed ROI.
4. Stop increasing redundant interval frames.
5. Report mean, variance, worst fold, and severe errors.

## Medium Priority

6. Add cost-sensitive or ordinal loss.
7. Add hard-example weighting.
8. Test full low-learning-rate fine-tuning.
9. Calibrate confidence scores.

## Lower Priority / Optional

10. Temporal smoothing for repeated observations.
11. Dual-input raw-plus-ROI architecture.
12. Additional heavyweight architectures.

---

# Main Conclusion

MeatLens does not primarily need a much larger frame count right now.

The immediate bottlenecks are:

1. augmentation appears declared but not wired into the CNN-only loader;
2. many images are correlated frames from a small number of physical samples;
3. processed ROI may remove useful color information or introduce artifacts;
4. freshness ordering is evaluated but not directly learned;
5. selecting the best run among many folds and seeds may overstate expected performance.

Fixing these issues is more likely to improve generalization, reduce training cost, and strengthen the thesis methodology than purchasing and photographing a substantially larger amount of meat.

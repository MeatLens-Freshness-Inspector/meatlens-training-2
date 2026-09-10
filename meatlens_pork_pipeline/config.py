from dataclasses import dataclass
from pathlib import Path

LABEL_ORDER = ("fresh", "not fresh", "spoiled")
INPUT_SIZE = (224, 224)
MODEL_INPUT_MODE = "cnn_only"
IMAGE_CROP_MODE = "preprocessed_hsv_lab_threshold_roi_224"
PIPELINE_OUTPUT_ROOT = Path("training_outputs") / "mobilenetv3small_pork_cnn_only_onnx"

TRAINING1_COMPATIBLE_STRATEGY = "training1_compatible_end_to_end"
ROBOFLOW_CACHED_BASELINE_STRATEGY = "roboflow_cached_baseline_v1"
TRAINING_STRATEGIES = (
    TRAINING1_COMPATIBLE_STRATEGY,
    ROBOFLOW_CACHED_BASELINE_STRATEGY,
)
END_TO_END_DEFAULTS = {
    "batch_size": 32,
    "epochs_head": 4,
    "epochs_fine": 8,
    "head_lr": 5e-4,
    "fine_tune_lr": 1e-5,
    "fine_tune_fraction": 0.25,
    "augmentation": True,
    "monitor": "val_f1_macro",
}
PERFORMANCE_DEFAULTS = {
    "cache_mode": "memory",
    "deterministic_ops": True,
    "verbose": 2,
}


@dataclass(frozen=True)
class PipelinePaths:
    output_root: Path
    processed_dataset_root: Path
    run_root: Path
    processed_dir: Path
    splits_dir: Path
    models_dir: Path
    metrics_dir: Path
    predictions_dir: Path
    plots_dir: Path
    logs_dir: Path


def build_run_paths(
    dataset_slug: str,
    seed: int,
    timestamp: str,
    root: Path | None = None,
) -> PipelinePaths:
    output_root = root or PIPELINE_OUTPUT_ROOT
    processed_dataset_root = output_root / "processed_datasets" / dataset_slug
    run_root = output_root / "runs" / f"{timestamp}_seed{seed}"
    return PipelinePaths(
        output_root=output_root,
        processed_dataset_root=processed_dataset_root,
        run_root=run_root,
        processed_dir=run_root / "processed",
        splits_dir=run_root / "splits",
        models_dir=run_root / "models",
        metrics_dir=run_root / "metrics",
        predictions_dir=run_root / "predictions",
        plots_dir=run_root / "plots",
        logs_dir=run_root / "logs",
    )

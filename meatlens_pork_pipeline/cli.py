from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import PERFORMANCE_DEFAULTS, PIPELINE_OUTPUT_ROOT, build_run_paths
from .evaluation import evaluate_model
from .manifest import load_manifest
from .onnx_export import build_onnx_metadata, export_model_to_onnx, write_metadata_json
from .preprocess_dataset import preprocess_manifest_images
from .splits import generate_stratified_splits
from .training import train_model


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=PIPELINE_OUTPUT_ROOT)
    parser.add_argument("--background-mode", default="gray", choices=["gray", "black", "mean"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-fine", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    _add_training_performance_arguments(parser)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meatlens-pork-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--images-root", type=Path, required=True)
    preprocess_parser.add_argument("--manifest", type=Path, required=True)
    preprocess_parser.add_argument("--dataset-slug", required=True)
    preprocess_parser.add_argument("--output-root", type=Path, default=PIPELINE_OUTPUT_ROOT)
    preprocess_parser.add_argument("--background-mode", default="gray", choices=["gray", "black", "mean"])

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--processed-manifest-csv", type=Path, required=True)
    split_parser.add_argument("--output-dir", type=Path, required=True)
    split_parser.add_argument("--seed", type=int, required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-csv", type=Path, required=True)
    train_parser.add_argument("--val-csv", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--epochs-head", type=int, default=4)
    train_parser.add_argument("--epochs-fine", type=int, default=8)
    train_parser.add_argument("--head-lr", type=float, default=5e-4)
    train_parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    _add_training_performance_arguments(train_parser)
    train_parser.add_argument(
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

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--model-h5", type=Path, required=True)
    evaluate_parser.add_argument("--test-csv", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=32)

    export_parser = subparsers.add_parser("export-onnx")
    export_parser.add_argument("--model-h5", type=Path, required=True)
    export_parser.add_argument("--onnx-path", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    _add_common_run_arguments(run_parser)

    return parser


def _add_training_performance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-mode",
        choices=["none", "memory"],
        default=PERFORMANCE_DEFAULTS["cache_mode"],
    )
    parser.add_argument(
        "--deterministic-ops",
        dest="deterministic_ops",
        action="store_true",
        default=PERFORMANCE_DEFAULTS["deterministic_ops"],
    )
    parser.add_argument(
        "--no-deterministic-ops",
        dest="deterministic_ops",
        action="store_false",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=PERFORMANCE_DEFAULTS["verbose"],
    )


def _run_preprocess(args: argparse.Namespace) -> int:
    manifest_df = load_manifest(manifest_path=args.manifest, images_root=args.images_root)
    preprocess_manifest_images(
        manifest_df=manifest_df,
        output_root=args.output_root / "processed_datasets",
        dataset_slug=args.dataset_slug,
        background_mode=args.background_mode,
    )
    return 0


def _run_split(args: argparse.Namespace) -> int:
    processed_df = pd.read_csv(args.processed_manifest_csv)
    generate_stratified_splits(processed_df=processed_df, output_dir=args.output_dir, seed=args.seed)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    train_model(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs_head=args.epochs_head,
        epochs_fine=args.epochs_fine,
        head_lr=args.head_lr,
        fine_tune_lr=args.fine_tune_lr,
        training_strategy=args.training_strategy,
        cache_mode=args.cache_mode,
        deterministic_ops=args.deterministic_ops,
        verbose=args.verbose,
        performance_log_path=args.output_dir / "performance.jsonl",
    )
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    evaluate_model(
        model_h5_path=args.model_h5,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    return 0


def _run_export(args: argparse.Namespace) -> int:
    export_model_to_onnx(model_h5_path=args.model_h5, onnx_path=args.onnx_path)
    return 0


def _default_split_path(split_paths: dict[str, Path], key: str, directory: Path) -> Path:
    value = split_paths.get(key)
    return Path(value) if value is not None else directory / f"{key}.csv"


def _write_metadata_if_available(
    training_artifacts: object,
    evaluation_summary: object,
    model_dir: Path,
    seed: int,
) -> None:
    metrics = getattr(evaluation_summary, "metrics", None)
    class_weights = getattr(training_artifacts, "class_weights", None)
    train_count = getattr(training_artifacts, "train_count", None)
    val_count = getattr(training_artifacts, "val_count", None)
    test_count = getattr(evaluation_summary, "test_count", None)
    training_strategy = getattr(training_artifacts, "training_strategy", None)

    if not isinstance(metrics, dict):
        return
    if not isinstance(class_weights, dict):
        return
    if not isinstance(train_count, int) or not isinstance(val_count, int) or not isinstance(test_count, int):
        return

    metadata = build_onnx_metadata(
        seed=seed,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        class_weights=class_weights,
        metrics=metrics,
        training_strategy=training_strategy if isinstance(training_strategy, str) else None,
    )
    write_metadata_json(
        metadata,
        model_dir / "meatlens_mobilenetv3small_pork_cnn_only_metadata.json",
    )


def _run_full_pipeline(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    paths = build_run_paths(
        dataset_slug=args.dataset_slug,
        seed=args.seed,
        timestamp=timestamp,
        root=output_root,
    )

    for directory in (
        paths.run_root,
        paths.splits_dir,
        paths.models_dir,
        paths.metrics_dir,
        paths.predictions_dir,
        paths.plots_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_df = load_manifest(manifest_path=args.manifest, images_root=args.images_root)
    processed_df, _, _ = preprocess_manifest_images(
        manifest_df=manifest_df,
        output_root=paths.output_root / "processed_datasets",
        dataset_slug=args.dataset_slug,
        background_mode=args.background_mode,
    )
    split_paths = generate_stratified_splits(
        processed_df=processed_df,
        output_dir=paths.splits_dir,
        seed=args.seed,
    )

    train_csv = _default_split_path(split_paths, "train", paths.splits_dir)
    val_csv = _default_split_path(split_paths, "val", paths.splits_dir)
    test_csv = _default_split_path(split_paths, "test", paths.splits_dir)

    training_artifacts = train_model(
        train_csv=train_csv,
        val_csv=val_csv,
        output_dir=paths.models_dir,
        seed=args.seed,
        epochs_head=args.epochs_head,
        epochs_fine=args.epochs_fine,
        head_lr=args.head_lr,
        fine_tune_lr=args.fine_tune_lr,
        training_strategy=args.training_strategy,
        cache_mode=args.cache_mode,
        deterministic_ops=args.deterministic_ops,
        verbose=args.verbose,
        performance_log_path=paths.logs_dir / "performance.jsonl",
    )
    model_h5_path = Path(
        getattr(
            training_artifacts,
            "model_h5_path",
            paths.models_dir / "meatlens_mobilenetv3small_pork_cnn_only.keras",
        )
    )

    evaluation_summary = evaluate_model(
        model_h5_path=model_h5_path,
        test_csv=test_csv,
        output_dir=paths.metrics_dir,
        batch_size=args.batch_size,
    )

    export_model_to_onnx(
        model_h5_path=model_h5_path,
        onnx_path=paths.models_dir / "meatlens_mobilenetv3small_pork_cnn_only.onnx",
    )
    _write_metadata_if_available(
        training_artifacts=training_artifacts,
        evaluation_summary=evaluation_summary,
        model_dir=paths.models_dir,
        seed=args.seed,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "preprocess":
        return _run_preprocess(args)
    if args.command == "split":
        return _run_split(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    if args.command == "export-onnx":
        return _run_export(args)
    if args.command == "run":
        return _run_full_pipeline(args)
    return 1

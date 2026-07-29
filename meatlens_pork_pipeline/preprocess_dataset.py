from pathlib import Path

import pandas as pd
from PIL import Image

from .image_ops import process_image


def preprocess_manifest_images(
    manifest_df: pd.DataFrame,
    output_root: Path,
    dataset_slug: str,
    background_mode: str = "gray",
) -> tuple[pd.DataFrame, Path, Path]:
    dataset_root = output_root / dataset_slug
    images_root = dataset_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []

    for row in manifest_df.to_dict(orient="records"):
        source_path = Path(str(row["resolved_image_path"]))
        label = str(row["label"])
        output_path = images_root / label / source_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            processed_uint8, metadata = process_image(source_path, background_mode=background_mode)
            Image.fromarray(processed_uint8).save(output_path, quality=95)
            summary_rows.append(
                {
                    "image_name": source_path.name,
                    "label": label,
                    "processed_image_path": str(output_path),
                    **metadata,
                }
            )
            output_rows.append({**row, "processed_image_path": str(output_path)})
        except Exception as exc:
            failure_rows.append(
                {
                    "image_name": source_path.name,
                    "label": label,
                    "error": repr(exc),
                }
            )

    processed_df = pd.DataFrame(output_rows)
    summary_path = dataset_root / "preprocessing_summary.csv"
    failures_path = dataset_root / "preprocessing_failures.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
    return processed_df, summary_path, failures_path

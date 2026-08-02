from pathlib import Path

import pandas as pd

from meatlens_pork_pipeline.splits import generate_stratified_splits


def test_generate_stratified_splits_writes_train_val_test_files(tmp_path: Path) -> None:
    rows = []
    for label in ("fresh", "not fresh", "spoiled"):
        for idx in range(20):
            rows.append(
                {
                    "image_name": f"{label}_{idx}.jpg",
                    "label": label,
                    "processed_image_path": str(tmp_path / f"{label}_{idx}.jpg"),
                }
            )
    processed_df = pd.DataFrame(rows)

    split_paths = generate_stratified_splits(processed_df=processed_df, output_dir=tmp_path, seed=42)

    assert set(split_paths) == {"train", "val", "test", "summary"}
    assert Path(split_paths["train"]).exists()
    assert Path(split_paths["val"]).exists()
    assert Path(split_paths["test"]).exists()

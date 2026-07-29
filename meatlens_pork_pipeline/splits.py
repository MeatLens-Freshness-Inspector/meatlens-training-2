import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def generate_stratified_splits(
    processed_df: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> dict[str, Path]:
    train_df, temp_df = train_test_split(
        processed_df,
        test_size=0.30,
        stratify=processed_df["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    summary_path = output_dir / "split_summary.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "train_count": len(train_df),
                "val_count": len(val_df),
                "test_count": len(test_df),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "summary": summary_path,
    }

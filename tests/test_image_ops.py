from pathlib import Path

import pandas as pd
from PIL import Image

from meatlens_pork_pipeline.preprocess_dataset import preprocess_manifest_images


def test_preprocess_manifest_images_writes_224_rgb_outputs(tmp_path: Path) -> None:
    image_path = tmp_path / "fresh_001.jpg"
    Image.new("RGB", (400, 300), (180, 50, 50)).save(image_path)

    manifest_df = pd.DataFrame(
        [
            {
                "image_name": "fresh_001.jpg",
                "label": "fresh",
                "resolved_image_path": str(image_path),
            }
        ]
    )

    processed_df, summary_path, failures_path = preprocess_manifest_images(
        manifest_df=manifest_df,
        output_root=tmp_path / "processed_datasets",
        dataset_slug="pilot",
    )

    processed_path = Path(processed_df.loc[0, "processed_image_path"])
    assert processed_path.exists()
    assert Image.open(processed_path).size == (224, 224)
    assert summary_path.exists()
    assert failures_path.exists()

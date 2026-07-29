from pathlib import Path

import pandas as pd

from meatlens_pork_pipeline.manifest import load_manifest


def test_load_manifest_normalizes_labels_and_resolves_paths(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    images_root.mkdir()
    (images_root / "A.JPG").write_bytes(b"fake")
    manifest_path = tmp_path / "labels.csv"
    pd.DataFrame([{"image_name": "A.JPG", "label": "Fresh"}]).to_csv(manifest_path, index=False)

    manifest_df = load_manifest(manifest_path=manifest_path, images_root=images_root)

    assert manifest_df.loc[0, "label"] == "fresh"
    assert Path(manifest_df.loc[0, "resolved_image_path"]).name == "A.JPG"

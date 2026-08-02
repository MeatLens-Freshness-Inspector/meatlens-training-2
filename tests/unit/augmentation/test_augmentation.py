from __future__ import annotations

from meatlens_pork_pipeline.augmentation import build_training_augmentation


def test_geometry_only_augmentation_excludes_color_jitter_layers() -> None:
    augmentation = build_training_augmentation("geometry_only_v1")
    layer_names = [layer.name for layer in augmentation.layers]

    assert "random_brightness" not in layer_names
    assert "random_contrast" not in layer_names


def test_conservative_augmentation_keeps_color_jitter_layers() -> None:
    augmentation = build_training_augmentation("conservative_v1")
    layer_names = [layer.name for layer in augmentation.layers]

    assert "random_brightness" in layer_names
    assert "random_contrast" in layer_names

from meatlens_pork_pipeline.onnx_export import build_onnx_metadata


def test_build_onnx_metadata_contains_shipped_contract_fields() -> None:
    metadata = build_onnx_metadata(
        seed=42,
        train_count=70,
        val_count=15,
        test_count=15,
        class_weights={0: 1.0, 1: 1.1, 2: 0.9},
        metrics={
            "accuracy": 0.95,
            "macro_precision": 0.95,
            "macro_recall": 0.94,
            "macro_f1": 0.945,
        },
    )

    assert metadata["backbone"] == "MobileNetV3Small"
    assert metadata["model_input_mode"] == "cnn_only"
    assert metadata["image_crop_mode"] == "preprocessed_hsv_lab_threshold_roi_224"
    assert metadata["label_order"] == ["fresh", "not fresh", "spoiled"]
    assert metadata["seed"] == 42

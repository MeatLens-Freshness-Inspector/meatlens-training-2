from meatlens_pork_pipeline.training import build_mobilenetv3small_model, compute_class_weights
import pytest


def test_build_mobilenetv3small_model_has_three_class_softmax() -> None:
    model = build_mobilenetv3small_model(weights=None)
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 3)
    assert model.layers[-1].activation.__name__ == "softmax"


def test_build_mobilenetv3small_model_uses_stable_training_defaults() -> None:
    model = build_mobilenetv3small_model(weights=None)

    assert "dense_128" not in {layer.name for layer in model.layers}
    assert model.optimizer.learning_rate.numpy() == pytest.approx(1e-4)
    assert model.loss.label_smoothing == pytest.approx(0.0)


def test_compute_class_weights_returns_all_three_indices() -> None:
    class_weights = compute_class_weights(["fresh", "fresh", "not fresh", "spoiled"])
    assert set(class_weights) == {0, 1, 2}

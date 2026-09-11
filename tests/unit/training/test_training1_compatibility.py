from meatlens_pork_pipeline.config import (
    END_TO_END_DEFAULTS,
    PERFORMANCE_DEFAULTS,
    ROBOFLOW_CACHED_BASELINE_STRATEGY,
    TRAINING1_COMPATIBLE_STRATEGY,
    TRAINING_STRATEGIES,
)


def test_named_training_strategies_have_one_official_and_one_baseline() -> None:
    assert TRAINING1_COMPATIBLE_STRATEGY == "training1_compatible_end_to_end"
    assert ROBOFLOW_CACHED_BASELINE_STRATEGY == "roboflow_cached_baseline_v1"
    assert set(TRAINING_STRATEGIES) == {
        "training1_compatible_end_to_end",
        "roboflow_cached_baseline_v1",
    }


def test_training1_defaults_match_training1() -> None:
    assert END_TO_END_DEFAULTS == {
        "batch_size": 32,
        "epochs_head": 4,
        "epochs_fine": 8,
        "head_lr": 5e-4,
        "fine_tune_lr": 1e-5,
        "fine_tune_fraction": 0.25,
        "augmentation": True,
        "monitor": "val_f1_macro",
    }


def test_performance_defaults_avoid_unsupported_tf210_windows_gpu_determinism() -> None:
    assert PERFORMANCE_DEFAULTS["deterministic_ops"] is False

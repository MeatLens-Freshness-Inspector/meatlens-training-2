from __future__ import annotations

import json

import tensorflow as tf

import meatlens_pork_pipeline.training as training


def test_set_training_seed_can_skip_deterministic_gpu_kernels(monkeypatch):
    called = False

    def enable_determinism():
        nonlocal called
        called = True

    monkeypatch.setattr(tf.config.experimental, "enable_op_determinism", enable_determinism)

    training._set_training_seed(42, deterministic_ops=False)

    assert called is False


def test_performance_timing_callback_writes_jsonl_epoch_record(tmp_path):
    output_path = tmp_path / "performance.jsonl"
    callback = training.PerformanceTimingCallback(
        output_path=output_path,
        train_sample_count=8,
        val_sample_count=4,
    )
    callback.set_params({"steps": 2, "validation_steps": 1})

    callback.on_epoch_begin(0)
    for batch in range(2):
        callback.on_train_batch_begin(batch)
        callback.on_train_batch_end(batch)
    callback.on_test_begin()
    callback.on_test_batch_begin(0)
    callback.on_test_batch_end(0)
    callback.on_test_end()
    callback.on_epoch_end(0, {"val_f1_macro": 0.75})

    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["epoch"] == 1
    assert record["epoch_total_seconds"] >= 0.0
    assert record["train_batch_seconds"] >= 0.0
    assert record["validation_batch_seconds"] >= 0.0
    assert record["train_images_per_second"] >= 0.0
    assert record["val_f1_macro"] == 0.75

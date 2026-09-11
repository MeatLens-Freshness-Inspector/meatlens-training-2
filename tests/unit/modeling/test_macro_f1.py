from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import f1_score

from meatlens_pork_pipeline.modeling import MacroF1Metric


def test_macro_f1_metric_matches_sklearn_for_missing_and_present_classes():
    y_true = np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)
    y_pred = np.array([0, 1, 1, 2, 0, 2], dtype=np.int32)
    metric = MacroF1Metric(num_classes=3)
    metric.update_state(tf.one_hot(y_true, 3), tf.one_hot(y_pred, 3))

    assert float(metric.result()) == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )


def test_macro_f1_metric_resets_between_validation_passes():
    metric = MacroF1Metric(num_classes=3)
    metric.update_state(tf.one_hot([0, 1, 2], 3), tf.one_hot([0, 1, 2], 3))
    metric.reset_state()

    assert float(metric.result()) == 0.0


def test_macro_f1_metric_handles_sparse_labels_with_singleton_column_shape():
    y_true = np.array([[0], [1], [2], [2]], dtype=np.int32)
    y_pred = np.array([0, 2, 2, 1], dtype=np.int32)
    metric = MacroF1Metric(num_classes=3)
    metric.update_state(tf.convert_to_tensor(y_true), tf.one_hot(y_pred, 3))

    assert float(metric.result()) == pytest.approx(
        f1_score(y_true.reshape(-1), y_pred, average="macro", zero_division=0)
    )

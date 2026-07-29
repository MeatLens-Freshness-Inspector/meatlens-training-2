import pandas as pd

from meatlens_pork_pipeline.evaluation import summarize_predictions


def test_summarize_predictions_returns_metrics_and_confusion_matrix() -> None:
    prediction_df = pd.DataFrame(
        [
            {"true_label": "fresh", "predicted_label": "fresh"},
            {"true_label": "fresh", "predicted_label": "not fresh"},
            {"true_label": "not fresh", "predicted_label": "not fresh"},
            {"true_label": "spoiled", "predicted_label": "spoiled"},
        ]
    )

    summary = summarize_predictions(prediction_df)

    assert "accuracy" in summary
    assert "macro_precision" in summary
    assert "macro_recall" in summary
    assert "macro_f1" in summary
    assert summary["confusion_matrix"].shape == (3, 3)

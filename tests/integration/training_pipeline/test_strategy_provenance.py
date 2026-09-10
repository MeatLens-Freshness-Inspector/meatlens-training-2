import hashlib
import json
from pathlib import Path


BASELINE_ROOT = Path(
    "training_outputs_committable/roboflow/"
    "mobilenetv3small_8fold_processed_roi_cnn_only"
)
EXPECTED_METRICS_SHA256 = "E5B2108565733E9947A6ABA8BCFFB12CB6272065A009E4D4858D95D11C998697"
EXPECTED_MODEL_SHA256 = "A51C7C899BA0E65BC209714095881FA7781BAA77717FA50984427DBFCA138131"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_recorded_roboflow_baseline_remains_92_7_percent() -> None:
    metrics_path = BASELINE_ROOT / "_e/fold4_s123/test_metrics.json"
    model_path = BASELINE_ROOT / "models/processed_roi8_cnn_only_fold4_seed123.keras"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics["accuracy"] == 0.927038626609442
    assert _sha256(metrics_path) == EXPECTED_METRICS_SHA256
    assert _sha256(model_path) == EXPECTED_MODEL_SHA256


def test_official_roboflow_outputs_use_a_separate_strategy_namespace() -> None:
    notebook_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(
            Path("04_train_8fold_mobilenetv3small.ipynb").read_text(encoding="utf-8")
        )["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "training1_compatible_end_to_end" in notebook_source
    assert "roboflow_cached_baseline_v1" in notebook_source
    assert "training1_compatible_end_to_end" in notebook_source.split(
        "TRAINING_OUTPUTS_ROOT", 1
    )[-1]

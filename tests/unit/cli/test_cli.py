from pathlib import Path

from meatlens_pork_pipeline import cli


def test_run_command_executes_full_pipeline(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch.setattr(cli, "load_manifest", lambda manifest_path, images_root: calls.append("manifest") or [])
    monkeypatch.setattr(
        cli,
        "preprocess_manifest_images",
        lambda **kwargs: calls.append("preprocess") or ([], tmp_path / "summary.csv", tmp_path / "failures.csv"),
    )
    monkeypatch.setattr(cli, "generate_stratified_splits", lambda **kwargs: calls.append("split") or {})
    monkeypatch.setattr(cli, "train_model", lambda **kwargs: calls.append("train") or object())
    monkeypatch.setattr(cli, "evaluate_model", lambda **kwargs: calls.append("evaluate") or object())
    monkeypatch.setattr(cli, "export_model_to_onnx", lambda **kwargs: calls.append("onnx") or (tmp_path / "model.onnx"))

    exit_code = cli.main(
        [
            "run",
            "--images-root",
            str(tmp_path / "images"),
            "--manifest",
            str(tmp_path / "labels.csv"),
            "--dataset-slug",
            "pilot",
            "--seed",
            "42",
        ]
    )

    assert exit_code == 0
    assert calls == ["manifest", "preprocess", "split", "train", "evaluate", "onnx"]

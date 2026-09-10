from meatlens_pork_pipeline import cli


def test_train_parser_accepts_cached_embedding_strategy(tmp_path) -> None:
    parser = cli._build_parser()

    args = parser.parse_args(
        [
            "train",
            "--train-csv",
            str(tmp_path / "train.csv"),
            "--val-csv",
            str(tmp_path / "val.csv"),
            "--output-dir",
            str(tmp_path / "models"),
            "--seed",
            "42",
            "--training-strategy",
            "cached_embeddings_sgd_v1",
        ]
    )

    assert args.training_strategy == "cached_embeddings_sgd_v1"


def test_train_parser_accepts_named_training_strategies(tmp_path) -> None:
    parser = cli._build_parser()

    for strategy in ("training1_compatible_end_to_end", "roboflow_cached_baseline_v1"):
        args = parser.parse_args(
            [
                "train",
                "--train-csv",
                str(tmp_path / "train.csv"),
                "--val-csv",
                str(tmp_path / "val.csv"),
                "--output-dir",
                str(tmp_path / "models"),
                "--seed",
                "42",
                "--training-strategy",
                strategy,
            ]
        )

        assert args.training_strategy == strategy


def test_train_parser_accepts_performance_controls(tmp_path) -> None:
    parser = cli._build_parser()

    args = parser.parse_args(
        [
            "train",
            "--train-csv",
            str(tmp_path / "train.csv"),
            "--val-csv",
            str(tmp_path / "val.csv"),
            "--output-dir",
            str(tmp_path / "models"),
            "--seed",
            "42",
            "--cache-mode",
            "none",
            "--no-deterministic-ops",
            "--verbose",
            "0",
        ]
    )

    assert args.cache_mode == "none"
    assert args.deterministic_ops is False
    assert args.verbose == 0


def test_train_command_forwards_performance_controls(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_train_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "train_model", fake_train_model)

    assert cli.main(
        [
            "train",
            "--train-csv",
            str(tmp_path / "train.csv"),
            "--val-csv",
            str(tmp_path / "val.csv"),
            "--output-dir",
            str(tmp_path / "models"),
            "--seed",
            "42",
            "--cache-mode",
            "none",
            "--no-deterministic-ops",
            "--verbose",
            "0",
        ]
    ) == 0
    assert captured["cache_mode"] == "none"
    assert captured["deterministic_ops"] is False
    assert captured["verbose"] == 0

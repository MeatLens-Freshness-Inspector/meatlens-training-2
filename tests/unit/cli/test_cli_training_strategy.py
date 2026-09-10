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

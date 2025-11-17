"""CLI contract for scripts.build_index."""

from __future__ import annotations

import pytest

from scripts.build_index import build_parser, _resolve_sample_size


def test_build_index_parser_contract() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    args = parser.parse_args(
        [
            "--dataset-name",
            "sqid",
            "--models",
            "google/siglip-base-patch16-224",
            "openai/clip-vit-large-patch14",
            "--alphas",
            "0.2",
            "0.8",
            "--environment",
            "prod",
        ]
    )
    assert args.dataset_name == "sqid"
    assert args.models == [
        "google/siglip-base-patch16-224",
        "openai/clip-vit-large-patch14",
    ]
    assert args.alphas == [0.2, 0.8]
    assert args.environment == "prod"


def test_resolve_sample_size_prefers_cli_override() -> None:
    assert _resolve_sample_size(environment="prod", cli_sample_size=10, default_dev_sample_size=5) == 10


def test_resolve_sample_size_defaults_only_in_dev() -> None:
    assert _resolve_sample_size(environment="dev", cli_sample_size=None, default_dev_sample_size=25) == 25
    assert _resolve_sample_size(
        environment="prod",
        cli_sample_size=None,
        default_dev_sample_size=25,
    ) is None

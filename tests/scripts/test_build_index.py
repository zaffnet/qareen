"""CLI contract for scripts.build_index."""

from __future__ import annotations

import pytest

from scripts.build_index import build_parser


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
            "--environment",
            "prod",
        ]
    )
    assert args.dataset_name == "sqid"
    assert args.models == [
        "google/siglip-base-patch16-224",
        "openai/clip-vit-large-patch14",
    ]
    assert args.environment == "prod"

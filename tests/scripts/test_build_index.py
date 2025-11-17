"""CLI contract for scripts.build_index."""

from __future__ import annotations

import pytest

from scripts.build_index import _validate_alpha_values, build_parser


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


def test_validate_alpha_values_enforces_bounds_and_uniqueness() -> None:
    assert _validate_alpha_values([0.3, 0.3, 0.6]) == [0.3, 0.6]

    with pytest.raises(ValueError):
        _validate_alpha_values([1.2])

"""CLI contract for scripts.build_index."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_index import build_parser, main


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


def test_main_uses_settings_defaults_and_wires_indexer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class DummyLoader:
        def __init__(self, dataset_name: str, *, split: str | None = None) -> None:
            calls["loader_args"] = (dataset_name, split)

        def load(self) -> list[SimpleNamespace]:
            return [SimpleNamespace()]

        def get_dataset_name(self) -> str:
            return "sqid"

    class DummyIndexer:
        def __init__(self, *, environment: str, chroma_db_dir: Path, dev_sample_size: int | None) -> None:
            calls["indexer_init"] = {
                "environment": environment,
                "chroma_db_dir": chroma_db_dir,
                "dev_sample_size": dev_sample_size,
            }

        def get_collection_name(self, *, dataset_name: str, environment: str, model_id: str, alpha: float) -> str:
            calls.setdefault("collection_args", []).append(
                {
                    "dataset_name": dataset_name,
                    "environment": environment,
                    "model_id": model_id,
                    "alpha": alpha,
                }
            )
            return "collection"

        def index(self, items: list[SimpleNamespace], *, model_id: str, alpha: float) -> None:  # pragma: no cover - stub
            calls.setdefault("index_calls", []).append({"model_id": model_id, "alpha": alpha, "count": len(items)})

    fake_settings = SimpleNamespace(
        default_embedding_models=["model-a", "model-b"],
        default_alpha_values=[0.25],
        environment="staging",
        dev_sample_size=11,
        chroma_db_dir=Path("/tmp/chroma"),
    )

    monkeypatch.setattr("scripts.build_index.Settings", lambda: fake_settings)
    monkeypatch.setattr("scripts.build_index.HuggingFaceDatasetLoader", DummyLoader)
    monkeypatch.setattr("scripts.build_index.ChromaIndexer", DummyIndexer)

    assert main(["--dataset-name", "sqid"]) == 0
    assert calls["loader_args"] == ("sqid", "train")
    assert calls["indexer_init"] == {
        "environment": "staging",
        "chroma_db_dir": Path("/tmp/chroma"),
        "dev_sample_size": 11,
    }
    assert calls["index_calls"] == [
        {"model_id": "model-a", "alpha": 0.25, "count": 1},
        {"model_id": "model-b", "alpha": 0.25, "count": 1},
    ]

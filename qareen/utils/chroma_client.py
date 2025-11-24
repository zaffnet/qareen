from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings

if TYPE_CHECKING:
    from pathlib import Path


def create_chroma_client(db_path: Path) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(db_path), settings=ChromaSettings(anonymized_telemetry=False)
    )


def close_chroma_client(client: chromadb.PersistentClient | None) -> None:
    if client is not None:
        with contextlib.suppress(Exception):
            client.clear_system_cache()

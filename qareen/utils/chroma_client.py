from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings

if TYPE_CHECKING:
    from pathlib import Path


def create_chroma_client(db_path: Path) -> chromadb.PersistentClient:
    """
    Create a Chromadb persistent client configured to use the given filesystem path.
    
    Parameters:
        db_path (Path): Filesystem path to the Chromadb persistent database (directory or file).
    
    Returns:
        chromadb.PersistentClient: A PersistentClient connected to the given path with anonymized telemetry disabled.
    """
    return chromadb.PersistentClient(
        path=str(db_path), settings=ChromaSettings(anonymized_telemetry=False)
    )


def close_chroma_client(client: chromadb.PersistentClient | None) -> None:
    """
    Attempt to clear the Chromadb client's system cache if a client is provided.
    
    If `client` is not None, calls its `clear_system_cache()` method and ignores `AttributeError`
    or `RuntimeError` raised by that call. Does nothing when `client` is None.
    """
    if client is not None:
        with contextlib.suppress(AttributeError, RuntimeError):
            client.clear_system_cache()
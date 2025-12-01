from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def create_chroma_client(db_path: Path) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(db_path), settings=ChromaSettings(anonymized_telemetry=False)
    )


def close_chroma_client(client: chromadb.PersistentClient | None) -> None:
    if client is not None:
        try:
            client.clear_system_cache()
        except AttributeError as e:
            logger.debug("ChromaDB client missing clear_system_cache: %s", e)
        except RuntimeError as e:
            logger.warning("RuntimeError during ChromaDB cleanup: %s", e)

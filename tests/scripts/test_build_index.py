"""CLI tests for scripts.build_index."""

from __future__ import annotations

import typer

import scripts.build_index


def test_build_index_imports() -> None:
    """Test that build_index module imports successfully."""
    assert hasattr(scripts.build_index, "app")
    assert hasattr(scripts.build_index, "main")


def test_build_index_app_is_typer() -> None:
    """Test that app is a Typer instance."""
    assert isinstance(scripts.build_index.app, typer.Typer)

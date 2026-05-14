"""Prompt resource files loaded at runtime."""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent


def load_system_prompt() -> str:
    """Load the Solver system prompt from system.md."""
    return (_HERE / "system.md").read_text()

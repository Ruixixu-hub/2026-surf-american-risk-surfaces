"""Resolve frozen original inputs without duplicating them in the extension tree."""

from __future__ import annotations

from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = EXTENSION_ROOT.parent


def frozen_input(relative: str | Path) -> Path:
    """Prefer an extension-local input, then the original repository copy."""

    relative_path = Path(relative)
    local = EXTENSION_ROOT / relative_path
    if local.exists():
        return local
    original = REPOSITORY_ROOT / relative_path
    return original if original.exists() else local


def populated_input_directory(relative: str | Path, pattern: str) -> Path:
    """Choose a local or original input directory containing ``pattern``."""

    relative_path = Path(relative)
    local = EXTENSION_ROOT / relative_path
    if local.exists() and any(local.glob(pattern)):
        return local
    original = REPOSITORY_ROOT / relative_path
    if original.exists() and any(original.glob(pattern)):
        return original
    return local


def portable_path(path: str | Path) -> str:
    """Return a stable label for extension-local or original-repository data."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(EXTENSION_ROOT.resolve()))
    except ValueError:
        pass
    try:
        return f"repository:{resolved.relative_to(REPOSITORY_ROOT.resolve())}"
    except ValueError:
        return str(resolved)

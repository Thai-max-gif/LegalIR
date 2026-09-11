from __future__ import annotations

from pathlib import Path
from typing import Iterable


def artifact_path(root: str | Path, filename: str, category: str | None = None) -> Path | None:
    """Resolve either the documented nested release or the observed flat Kaggle release."""
    root = Path(root)
    candidates = ([root / category / filename] if category else []) + [root / filename]
    return next((path for path in candidates if path.is_file()), None)


def artifact_candidates(root: str | Path, filename: str, category: str | None = None) -> list[Path]:
    """Return every supported location in preference order, whether it exists or not."""
    root = Path(root)
    candidates = [root / filename]
    if category:
        candidates.insert(0, root / category / filename)
    return candidates


def artifact_description(root: str | Path, filename: str, category: str | None = None) -> str:
    return " or ".join(str(path) for path in artifact_candidates(root, filename, category))


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def required_table(root: str | Path, filename: str) -> Path:
    path = artifact_path(root, filename, "canonical")
    if not path:
        raise FileNotFoundError(
            f"Cannot find {filename}; checked {artifact_description(root, filename, 'canonical')}"
        )
    return path

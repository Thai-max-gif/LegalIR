from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(bundle: str | Path) -> Path:
    root = Path(bundle)
    out = root / "checksums.sha256"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != out:
            rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out

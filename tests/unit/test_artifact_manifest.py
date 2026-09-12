import json
from pathlib import Path

from src.artifacts.checksums import sha256_file
from src.artifacts.manifest import build_inventory, verify_inventory


def test_inventory_uses_relative_paths_and_detects_changes(tmp_path: Path):
    root = tmp_path / "shared"
    root.mkdir()
    payload = root / "documents.parquet"
    payload.write_bytes(b"canonical")
    inventory = build_inventory(root)
    assert inventory["documents.parquet"]["sha256"] == sha256_file(payload)
    assert inventory["documents.parquet"]["size"] == len(b"canonical")

    manifest = tmp_path / "artifacts.sha256.json"
    manifest.write_text(json.dumps(inventory), encoding="utf-8")
    assert verify_inventory(root, manifest) == []

    payload.write_bytes(b"changed")
    assert verify_inventory(root, manifest) == ["checksum mismatch: documents.parquet"]

    payload.unlink()
    assert verify_inventory(root, manifest) == ["missing: documents.parquet"]

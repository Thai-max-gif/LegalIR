"""Builder for assembling and sealing immutable production bundles with mandatory file validation."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pyarrow.parquet as pq

from src.core.hashing import sha256_file
from src.core.manifests import BundleManifest


MANDATORY_BUNDLE_FILES: List[str] = [
    "final_training_pairs.parquet",
    "public_candidates.parquet",
    "public_evidence.parquet",
    "production_lock.json",
    "fusion_model.json",
    "static_cache_provenance.json",
    "validation_summary.json",
    "dataset_provenance.json",
]

HEX_40_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class ProductionBundleBuilder:
    """Collects, copies, and seals the artifacts that constitute the production bundle."""

    def __init__(
        self,
        bundle_dir: Union[str, Path],
        runtime_commit: str,
        dataset_fingerprint: str,
        config_sha256: str,
        bundle_version: str = "v1",
        strict_mandatory_check: bool = True,
    ):
        self.bundle_dir = Path(bundle_dir)
        self.runtime_commit = str(runtime_commit).strip()
        self.dataset_fingerprint = str(dataset_fingerprint).strip()
        self.config_sha256 = str(config_sha256).strip()
        self.bundle_version = bundle_version
        self.strict_mandatory_check = strict_mandatory_check
        self.files_map: Dict[str, Path] = {}

        # Validate hashes are not placeholder defaults
        if self.strict_mandatory_check:
            if not HEX_40_RE.match(self.runtime_commit):
                raise ValueError(
                    f"runtime_commit must be a real 40-char git commit SHA, got: '{self.runtime_commit}'"
                )
            if not HEX_64_RE.match(self.dataset_fingerprint):
                raise ValueError(
                    f"dataset_fingerprint must be a real 64-char SHA-256 digest, got: '{self.dataset_fingerprint}'"
                )
            if not HEX_64_RE.match(self.config_sha256):
                raise ValueError(
                    f"config_sha256 must be a real 64-char SHA-256 digest, got: '{self.config_sha256}'"
                )

    def add_file(self, rel_path: str, src_path: Union[str, Path]) -> None:
        """Register a file to be copied into the bundle under rel_path."""
        src = Path(src_path)
        if not src.is_file():
            raise FileNotFoundError(f"Bundle source file not found: {src}")
        self.files_map[rel_path] = src

        # Auto-discover associated learned model payload if adding fusion_model.json
        if rel_path == "fusion_model.json":
            try:
                import json
                with open(src, "r", encoding="utf-8") as f:
                    desc = json.load(f)
                if desc.get("winning_method") == "learned_ranker":
                    p_name = desc.get("learned_model", {}).get("file")
                    if p_name and p_name not in self.files_map:
                        candidate_payload = src.parent / p_name
                        if candidate_payload.is_file():
                            self.files_map[p_name] = candidate_payload
            except Exception:
                pass

    def freeze(self) -> BundleManifest:
        """Copy all files into bundle directory, verify mandatory set, and write bundle_manifest.json."""
        if self.strict_mandatory_check:
            missing = [f for f in MANDATORY_BUNDLE_FILES if f not in self.files_map]
            if missing:
                raise ValueError(f"Cannot freeze production bundle: missing mandatory files {missing}")

        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        file_entries: Dict[str, Dict[str, Any]] = {}

        for rel_path, src_p in sorted(self.files_map.items()):
            dst_p = self.bundle_dir / rel_path
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            if src_p.resolve() != dst_p.resolve():
                shutil.copy2(src_p, dst_p)

            digest = sha256_file(dst_p)
            size = dst_p.stat().st_size
            meta: Dict[str, Any] = {
                "sha256": digest,
                "size_bytes": size,
            }

            # If Parquet, record row count
            if dst_p.suffix == ".parquet":
                try:
                    pq_meta = pq.read_metadata(str(dst_p))
                    meta["num_rows"] = pq_meta.num_rows
                except Exception:
                    pass

            file_entries[rel_path] = meta

        manifest = BundleManifest(
            bundle_version=self.bundle_version,
            runtime_commit=self.runtime_commit,
            dataset_version="v2",
            dataset_fingerprint=self.dataset_fingerprint,
            config_sha256=self.config_sha256,
            files=file_entries,
            status="PASS",
        )

        manifest.save(self.bundle_dir / "bundle_manifest.json")
        return manifest

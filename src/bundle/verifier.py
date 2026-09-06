"""Cryptographic verifier for immutable production bundles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Union
import pyarrow.parquet as pq

from src.bundle.builder import MANDATORY_BUNDLE_FILES, HEX_40_RE, HEX_64_RE
from src.core.hashing import sha256_file
from src.core.manifests import BundleManifest


def verify_production_bundle(
    bundle_dir: Union[str, Path],
    strict_mandatory: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Cryptographically verify all artifacts inside the production bundle against
    the sealed bundle_manifest.json. Returns (is_valid, list_of_errors).
    """
    bundle_p = Path(bundle_dir)
    errors: List[str] = []

    manifest_p = bundle_p / "bundle_manifest.json"
    if not manifest_p.is_file():
        return False, [f"Missing bundle_manifest.json at {manifest_p}"]

    try:
        manifest = BundleManifest.load(manifest_p)
    except Exception as e:
        return False, [f"Failed to load bundle_manifest.json: {e}"]

    if manifest.status != "PASS":
        errors.append(f"Bundle manifest status is not PASS: '{manifest.status}'")

    if not manifest.files:
        errors.append("Bundle manifest contains no file entries.")

    # Validate commit and fingerprints
    if strict_mandatory:
        if not HEX_40_RE.match(manifest.runtime_commit):
            errors.append(f"Invalid runtime_commit in manifest: '{manifest.runtime_commit}' (must be 40-char SHA)")
        if not HEX_64_RE.match(manifest.dataset_fingerprint):
            errors.append(f"Invalid dataset_fingerprint in manifest: '{manifest.dataset_fingerprint}' (must be 64-char hex)")
        if not HEX_64_RE.match(manifest.config_sha256):
            errors.append(f"Invalid config_sha256 in manifest: '{manifest.config_sha256}' (must be 64-char hex)")

        # Verify all mandatory files are present
        for mf in MANDATORY_BUNDLE_FILES:
            if mf not in manifest.files:
                errors.append(f"Missing mandatory file '{mf}' from bundle manifest.")
            elif not (bundle_p / mf).is_file():
                errors.append(f"Missing mandatory file on disk: '{mf}'.")

    for rel_path, meta in manifest.files.items():
        artifact_p = bundle_p / rel_path
        if not artifact_p.is_file():
            errors.append(f"Missing bundle file on disk: {rel_path}")
            continue

        expected_sha = meta.get("sha256")
        expected_size = meta.get("size_bytes")
        expected_rows = meta.get("num_rows")

        actual_size = artifact_p.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            errors.append(f"Size mismatch on {rel_path}: expected {expected_size}, got {actual_size}")

        actual_sha = sha256_file(artifact_p)
        if expected_sha and actual_sha != expected_sha:
            errors.append(
                f"Digest mismatch on {rel_path}: expected {expected_sha}, got {actual_sha}"
            )

        if expected_rows is not None and artifact_p.suffix == ".parquet":
            try:
                actual_rows = pq.read_metadata(str(artifact_p)).num_rows
                if actual_rows != expected_rows:
                    errors.append(f"Row count mismatch on {rel_path}: expected {expected_rows}, got {actual_rows}")
            except Exception as e:
                errors.append(f"Failed inspecting row count on {rel_path}: {e}")

    # Semantic check on fusion_model.json descriptor if present
    fusion_desc_p = bundle_p / "fusion_model.json"
    if fusion_desc_p.is_file():
        try:
            import json
            desc = json.loads(fusion_desc_p.read_text(encoding="utf-8"))
            if desc.get("schema_version") != 1:
                errors.append(f"Invalid fusion_model.json schema_version: expected 1, got {desc.get('schema_version')}")
            w_method = desc.get("winning_method")
            if w_method not in ("reciprocal_rank_fusion", "learned_ranker"):
                errors.append(f"Invalid winning_method in fusion_model.json: '{w_method}'")
            elif w_method == "reciprocal_rank_fusion":
                if "rrf" not in desc or not isinstance(desc["rrf"], dict):
                    errors.append("fusion_model.json missing mandatory 'rrf' section for reciprocal_rank_fusion winner")
                if "learned_model" in desc:
                    errors.append("fusion_model.json contains 'learned_model' section but winning_method is reciprocal_rank_fusion")
            elif w_method == "learned_ranker":
                if "learned_model" not in desc or not isinstance(desc["learned_model"], dict):
                    errors.append("fusion_model.json missing mandatory 'learned_model' section for learned_ranker winner")
                else:
                    lm = desc["learned_model"]
                    payload_name = lm.get("file")
                    expected_sha = lm.get("sha256")
                    if not payload_name:
                        errors.append("fusion_model.json learned_model missing 'file'")
                    if not expected_sha:
                        errors.append("fusion_model.json learned_model missing 'sha256'")
                    if payload_name:
                        payload_p = bundle_p / payload_name
                        if not payload_p.is_file():
                            errors.append(f"Missing learned fusion payload in bundle: '{payload_name}'")
                        elif expected_sha:
                            actual_sha = sha256_file(payload_p)
                            if actual_sha != expected_sha:
                                errors.append(
                                    f"Learned fusion payload digest mismatch for '{payload_name}': expected {expected_sha}, got {actual_sha}"
                                )
        except Exception as e:
            errors.append(f"Failed parsing fusion_model.json: {e}")

    is_valid = len(errors) == 0
    return is_valid, errors

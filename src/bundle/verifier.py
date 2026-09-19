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
    strict_semantic: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Cryptographically verify all artifacts inside the production bundle against
    the sealed bundle_manifest.json and perform semantic cross-artifact checks.
    Returns (is_valid, list_of_errors).
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

    # Semantic cross-artifact checks
    if strict_semantic:
        import json
        import pandas as pd

        # 1. production_lock.json cross-verification
        lock_p = bundle_p / "production_lock.json"
        if lock_p.is_file():
            try:
                lock = json.loads(lock_p.read_text(encoding="utf-8"))
                if lock.get("runtime_commit") != manifest.runtime_commit:
                    errors.append(
                        f"runtime_commit mismatch: bundle manifest has '{manifest.runtime_commit}' "
                        f"but production_lock.json has '{lock.get('runtime_commit')}'"
                    )
                if lock.get("dataset_sha256") != manifest.dataset_fingerprint:
                    errors.append(
                        f"dataset_fingerprint mismatch: bundle manifest has '{manifest.dataset_fingerprint}' "
                        f"but production_lock.json dataset_sha256 has '{lock.get('dataset_sha256')}'"
                    )
                if lock.get("config_sha256") != manifest.config_sha256:
                    errors.append(
                        f"config_sha256 mismatch: bundle manifest has '{manifest.config_sha256}' "
                        f"but production_lock.json has '{lock.get('config_sha256')}'"
                    )
            except Exception as e:
                errors.append(f"Failed semantic check on production_lock.json: {e}")

        # 2. dataset_provenance.json verification
        ds_prov_p = bundle_p / "dataset_provenance.json"
        if ds_prov_p.is_file():
            try:
                ds_prov = json.loads(ds_prov_p.read_text(encoding="utf-8"))
                if "dataset_fingerprint" in ds_prov and ds_prov["dataset_fingerprint"] != manifest.dataset_fingerprint:
                    errors.append(
                        f"dataset_provenance fingerprint mismatch: expected '{manifest.dataset_fingerprint}', "
                        f"got '{ds_prov.get('dataset_fingerprint')}'"
                    )
                if ds_prov.get("doc_count") is not None and ds_prov["doc_count"] != 8532:
                    errors.append(f"dataset_provenance doc_count ({ds_prov['doc_count']}) != canonical v2 (8532)")
                if ds_prov.get("train_query_count") is not None and ds_prov["train_query_count"] != 7000:
                    errors.append(f"dataset_provenance train_query_count ({ds_prov['train_query_count']}) != canonical v2 (7000)")
                if ds_prov.get("public_query_count") is not None and ds_prov["public_query_count"] not in (1000, 2080):
                    errors.append(f"dataset_provenance public_query_count ({ds_prov['public_query_count']}) not in canonical official counts (1000, 2080)")
            except Exception as e:
                errors.append(f"Failed semantic check on dataset_provenance.json: {e}")

        # 3. static_cache_provenance.json verification
        sc_prov_p = bundle_p / "static_cache_provenance.json"
        if sc_prov_p.is_file():
            try:
                sc_prov = json.loads(sc_prov_p.read_text(encoding="utf-8"))
                if not sc_prov.get("label_free", True):
                    errors.append("static_cache_provenance indicates cache is not label_free")
                if sc_prov.get("qrels_used", False):
                    errors.append("static_cache_provenance indicates cache has qrels dependency")
            except Exception as e:
                errors.append(f"Failed semantic check on static_cache_provenance.json: {e}")

        # 4. validation_summary.json verification
        val_sum_p = bundle_p / "validation_summary.json"
        if val_sum_p.is_file():
            try:
                val_sum = json.loads(val_sum_p.read_text(encoding="utf-8"))
                folds = val_sum.get("folds", [])
                if len(folds) != 5:
                    errors.append(f"validation_summary does not contain 5/5 completed OOF folds (found {len(folds)})")
                if "doc_disjoint" not in val_sum:
                    errors.append("validation_summary missing completed doc_disjoint evaluation")
                if val_sum.get("leakage_violations", 0) != 0:
                    errors.append(f"validation_summary has non-zero leakage_violations: {val_sum.get('leakage_violations')}")
                if val_sum.get("duplicate_negative_violations", 0) != 0:
                    errors.append(
                        f"validation_summary has non-zero duplicate_negative_violations: "
                        f"{val_sum.get('duplicate_negative_violations')}"
                    )
            except Exception as e:
                errors.append(f"Failed semantic check on validation_summary.json: {e}")

        # 5. final_training_pairs.parquet query coverage
        pairs_p = bundle_p / "final_training_pairs.parquet"
        if pairs_p.is_file() and strict_mandatory:
            try:
                col_names = pq.read_schema(str(pairs_p)).names
                read_cols = ["query_id"]
                if "label" in col_names:
                    read_cols.append("label")
                df_pairs = pd.read_parquet(pairs_p, columns=read_cols)
                if "label" in df_pairs.columns:
                    pos = int((df_pairs["label"] > 0.5).sum())
                    neg = int((df_pairs["label"] <= 0.5).sum())
                    if pos == 0 or neg == 0:
                        errors.append(f"final_training_pairs missing required positive/negative coverage (pos={pos}, neg={neg})")
                uq_queries = df_pairs["query_id"].nunique()
                if uq_queries < 7000 and len(df_pairs) > 20:
                    errors.append(f"final_training_pairs query coverage gap: found {uq_queries} queries, expected 7000")
            except Exception as e:
                errors.append(f"Failed semantic check on final_training_pairs.parquet: {e}")

        # 6. public_candidates.parquet coverage
        pub_cands_p = bundle_p / "public_candidates.parquet"
        if pub_cands_p.is_file() and strict_mandatory:
            try:
                df_pub = pd.read_parquet(pub_cands_p, columns=["query_id"])
                pub_qids = df_pub["query_id"].nunique()
                if pub_qids < 1000 and len(df_pub) > 20:
                    errors.append(f"public_candidates query coverage gap: found {pub_qids} queries, expected 1000 official public qids")
            except Exception as e:
                errors.append(f"Failed semantic check on public_candidates.parquet: {e}")

        # 7. public_evidence.parquet coverage
        pub_ev_p = bundle_p / "public_evidence.parquet"
        if pub_ev_p.is_file() and strict_mandatory:
            try:
                df_ev = pd.read_parquet(pub_ev_p, columns=["query_id"])
                if df_ev.empty and (bundle_p / "public_candidates.parquet").is_file() and len(df_pub) > 20:
                    errors.append("public_evidence.parquet is empty but public_candidates has records")
            except Exception as e:
                errors.append(f"Failed semantic check on public_evidence.parquet: {e}")

        # 8. Semantic check on fusion_model.json descriptor if present
        fusion_desc_p = bundle_p / "fusion_model.json"
        if fusion_desc_p.is_file():
            try:
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

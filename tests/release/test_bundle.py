import json
import pytest
from pathlib import Path
from src.bundle.builder import ProductionBundleBuilder, MANDATORY_BUNDLE_FILES
from src.bundle.verifier import verify_production_bundle

REAL_40_SHA = "a" * 40
REAL_64_HASH1 = "b" * 64
REAL_64_HASH2 = "c" * 64


def test_bundle_build_and_verify(tmp_path):
    bundle_dir = tmp_path / "bundle"
    builder = ProductionBundleBuilder(
        bundle_dir=bundle_dir,
        runtime_commit=REAL_40_SHA,
        dataset_fingerprint=REAL_64_HASH1,
        config_sha256=REAL_64_HASH2,
        strict_mandatory_check=True,
    )

    # Add all mandatory files
    for mf in MANDATORY_BUNDLE_FILES:
        f = tmp_path / mf
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"content of {mf}", encoding="utf-8")
        builder.add_file(mf, f)

    manifest = builder.freeze()
    assert (bundle_dir / "bundle_manifest.json").is_file()
    assert manifest.status == "PASS"

    # Verifier check (cryptographic only for raw dummy text files)
    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=True, strict_semantic=False)
    assert is_valid is True, f"Verification failed: {errors}"
    assert len(errors) == 0


def test_bundle_missing_required_file_fails(tmp_path):
    bundle_dir = tmp_path / "incomplete_bundle"
    builder = ProductionBundleBuilder(
        bundle_dir=bundle_dir,
        runtime_commit=REAL_40_SHA,
        dataset_fingerprint=REAL_64_HASH1,
        config_sha256=REAL_64_HASH2,
        strict_mandatory_check=True,
    )

    # Add only one file, leaving 7 missing
    f = tmp_path / "final_training_pairs.parquet"
    f.write_text("dummy", encoding="utf-8")
    builder.add_file("final_training_pairs.parquet", f)

    with pytest.raises(ValueError, match="missing mandatory files"):
        builder.freeze()


def test_bundle_runtime_dataset_config_hashes_are_real(tmp_path):
    bundle_dir = tmp_path / "placeholder_bundle"
    # Placeholder commit e.g. "a0efb25" (not 40 chars) must fail
    with pytest.raises(ValueError, match="must be a real 40-char git commit SHA"):
        ProductionBundleBuilder(
            bundle_dir=bundle_dir,
            runtime_commit="a0efb25",
            dataset_fingerprint=REAL_64_HASH1,
            config_sha256=REAL_64_HASH2,
            strict_mandatory_check=True,
        )

    # Placeholder dataset fingerprint e.g. "canonical_v2_fingerprint" must fail
    with pytest.raises(ValueError, match="must be a real 64-char SHA-256 digest"):
        ProductionBundleBuilder(
            bundle_dir=bundle_dir,
            runtime_commit=REAL_40_SHA,
            dataset_fingerprint="canonical_v2_fingerprint",
            config_sha256=REAL_64_HASH2,
            strict_mandatory_check=True,
        )


def test_bundle_verifier_detects_tampered_file(tmp_path):
    bundle_dir = tmp_path / "tampered_bundle"
    builder = ProductionBundleBuilder(
        bundle_dir=bundle_dir,
        runtime_commit=REAL_40_SHA,
        dataset_fingerprint=REAL_64_HASH1,
        config_sha256=REAL_64_HASH2,
        strict_mandatory_check=False,
    )

    pairs_file = tmp_path / "final_training_pairs.parquet"
    pairs_file.write_text("original content", encoding="utf-8")
    builder.add_file("final_training_pairs.parquet", pairs_file)
    builder.freeze()

    # Tamper with file
    (bundle_dir / "final_training_pairs.parquet").write_text("tampered content", encoding="utf-8")

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=False)
    assert is_valid is False
    assert any("Digest mismatch" in e for e in errors)

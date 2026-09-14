"""
Cryptographic Fingerprinting, Git SHA Validation, and Gate Chain Governance.
Authoritative source for dataset manifests, config fingerprints, and protected key enforcement.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Union
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SHA_REGEX = re.compile(r"^[0-9a-fA-F]{40}$")

CRITICAL_DATASET_FILES: tuple[str, ...] = (
    "documents.parquet",
    "chunks.parquet",
    "queries_train.parquet",
    "qrels_train.parquet",
    "public-official.json",
)

PROTECTED_SCORE_KEYS: tuple[str, ...] = (
    # Candidate retrieval / RRF branch weights
    "weights",
    "branch_weights",
    "rrf_weights",
    "initial_weights",
    "field_weights",
    # Candidate counts & rerank targets
    "candidate_k",
    "top_k_candidates",
    "rerank_k",
    "top_k_for_rerank",
    "top_k_rerank",
    # Model identities & revisions
    "reranker_model",
    "model_name",
    "embedding_model",
    "reranker",
    "dense",
    "reranker_revision",
    "dense_revision",
    # Neural LoRA architecture & hyperparameters
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "loss_type",
    "loss",
    "learning_rate",
    "lr",
    # Fusion ranking policy & features
    "fusion_features",
    "feature_columns",
    "features",
    "fusion_policy",
    "selection_policy",
    "model_type",
    # Top-k selection logic
    "top_5_logic",
    "max_k",
    "min_k",
)


class ShaMismatchError(RuntimeError):
    """Raised when Git commit SHA does not match expected pinned revision."""


class DatasetFingerprintMismatchError(RuntimeError):
    """Raised when dataset bytes, manifest, or critical file hashes mismatch."""


class ProtectedKeyViolationError(RuntimeError):
    """Raised when a runtime profile attempts to mutate protected score-affecting algorithm settings."""


class GateChainValidationError(RuntimeError):
    """Raised when upstream gate reports are missing, failed, or mismatched."""


def compute_file_sha256(path: Union[Path, str]) -> str:
    """Compute standard hexadecimal SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_canonical_json_hash(data: Any) -> str:
    """Compute canonical SHA-256 digest of arbitrary structured data via sorted JSON."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_structured_config(config: Union[Mapping[str, Any], str, Path]) -> str:
    """Load and canonicalize structured configuration (dict or YAML/JSON path) into a SHA-256 hash."""
    if isinstance(config, (str, Path)):
        p = Path(config)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        text = p.read_text(encoding="utf-8")
        data = yaml.safe_load(text) if p.suffix in (".yaml", ".yml") else json.loads(text)
    else:
        data = dict(config)
    return compute_canonical_json_hash(data)


def get_git_head_sha(repo_root: Union[Path, str] = REPO_ROOT) -> str:
    """Derive actual release HEAD via git rev-parse HEAD."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except Exception as exc:
        raise ShaMismatchError(f"Failed to resolve git HEAD: {exc}") from exc

    if not head or not SHA_REGEX.match(head):
        raise ShaMismatchError(f"Resolved git HEAD is not a valid 40-character SHA: '{head}'")
    return head.lower()


def assert_exact_git_sha(
    expected_sha: Optional[str],
    repo_root: Union[Path, str] = REPO_ROOT,
    is_production: bool = True,
) -> str:
    """
    Validate that the repository HEAD precisely matches expected_sha.
    Strictly forbids 'main', 'master', empty, or unpinned SHAs in production.
    """
    actual_sha = get_git_head_sha(repo_root)

    if not expected_sha or not str(expected_sha).strip():
        if is_production:
            raise ShaMismatchError(
                f"Expected Git SHA must be provided for authoritative execution (found actual HEAD: {actual_sha})."
            )
        return actual_sha

    expected_clean = str(expected_sha).strip().lower()

    if expected_clean in ("main", "master"):
        raise ShaMismatchError(
            f"Literal branch name '{expected_clean}' is forbidden for authoritative execution. "
            f"Pass a detached 40-character commit SHA."
        )

    if not SHA_REGEX.match(expected_clean):
        raise ShaMismatchError(
            f"Expected Git SHA '{expected_sha}' is not a valid 40-character hexadecimal SHA."
        )

    if actual_sha != expected_clean:
        raise ShaMismatchError(
            f"Git SHA mismatch! Expected: {expected_clean}, Actual checked-out HEAD: {actual_sha}. "
            f"Execution must abort to prevent non-reproducible runs."
        )

    return actual_sha


def generate_dataset_manifest(dataset_dir: Union[Path, str]) -> dict[str, Any]:
    """Generate cryptographic dataset manifest for all critical files present."""
    data_dir = Path(dataset_dir)
    critical_files: dict[str, str] = {}

    for filename in CRITICAL_DATASET_FILES:
        target = data_dir / filename
        if target.is_file():
            critical_files[filename] = compute_file_sha256(target)

    # Compute manifest hash from canonical JSON of critical files mapping
    manifest_sha256 = compute_canonical_json_hash(critical_files)

    return {
        "schema_version": 3,
        "manifest_sha256": manifest_sha256,
        "critical_files": critical_files,
    }


@dataclasses.dataclass(frozen=True)
class DatasetVerificationResult:
    is_valid: bool
    manifest_sha256: str
    verified_files: dict[str, str]


def verify_dataset_fingerprint(
    dataset_dir: Union[Path, str],
    expected_manifest_hash: Optional[str] = None,
    critical_files_expected: Optional[dict[str, str]] = None,
) -> DatasetVerificationResult:
    """
    Cryptographically verify all critical dataset files against their SHA-256 digests.
    Fails closed if any file is missing, mutated by even 1 byte, or manifest hash mismatches.
    """
    data_dir = Path(dataset_dir)
    manifest_path = data_dir / "dataset_manifest.json"

    if manifest_path.is_file():
        try:
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise DatasetFingerprintMismatchError(f"Corrupt dataset_manifest.json: {exc}") from exc
    else:
        stored_manifest = None

    actual_files: dict[str, str] = {}
    for filename in CRITICAL_DATASET_FILES:
        target = data_dir / filename
        if not target.is_file():
            raise DatasetFingerprintMismatchError(f"Missing required canonical dataset file: {target}")
        actual_hash = compute_file_sha256(target)
        actual_files[filename] = actual_hash

        # Verify against stored manifest
        if stored_manifest and "critical_files" in stored_manifest:
            expected_hash = stored_manifest["critical_files"].get(filename)
            if expected_hash and expected_hash != actual_hash:
                raise DatasetFingerprintMismatchError(
                    f"Dataset file '{filename}' failed checksum verification! "
                    f"Stored: {expected_hash}, Actual: {actual_hash}"
                )

        # Verify against explicit expected files dictionary
        if critical_files_expected:
            expected_hash = critical_files_expected.get(filename)
            if expected_hash and expected_hash != actual_hash:
                raise DatasetFingerprintMismatchError(
                    f"Dataset file '{filename}' failed explicit checksum verification! "
                    f"Expected: {expected_hash}, Actual: {actual_hash}"
                )

    actual_manifest_hash = compute_canonical_json_hash(actual_files)

    if stored_manifest:
        stored_hash = stored_manifest.get("manifest_sha256")
        if stored_hash and stored_hash != actual_manifest_hash:
            raise DatasetFingerprintMismatchError(
                f"Manifest checksum mismatch! Stored: {stored_hash}, Computed: {actual_manifest_hash}"
            )

    if expected_manifest_hash and expected_manifest_hash != actual_manifest_hash:
        raise DatasetFingerprintMismatchError(
            f"Dataset manifest hash mismatch! Expected: {expected_manifest_hash}, Computed: {actual_manifest_hash}"
        )

    return DatasetVerificationResult(
        is_valid=True,
        manifest_sha256=actual_manifest_hash,
        verified_files=actual_files,
    )


def _extract_all_key_paths(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively extract flattened dot-separated key paths from nested mapping."""
    flat: dict[str, Any] = {}
    for k, v in d.items():
        curr = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            flat.update(_extract_all_key_paths(v, curr))
        else:
            flat[curr] = v
    return flat


def validate_runtime_overrides(
    algorithm_config: Mapping[str, Any],
    runtime_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Validate that runtime profile does not override any protected score-affecting algorithm keys.
    Returns the resolved/merged configuration dictionary.
    """
    algo_flat = _extract_all_key_paths(algorithm_config)
    runtime_flat = _extract_all_key_paths(runtime_profile)

    for path, runtime_val in runtime_flat.items():
        key_name = path.split(".")[-1]
        # Check if key is protected
        if key_name in PROTECTED_SCORE_KEYS or any(f".{p}" in path or path.startswith(f"{p}.") for p in PROTECTED_SCORE_KEYS):
            # Check if it was in algorithm config and differs
            if path in algo_flat and algo_flat[path] != runtime_val:
                raise ProtectedKeyViolationError(
                    f"Runtime profile illegally mutated protected score-affecting key '{path}': "
                    f"base={algo_flat[path]!r}, override={runtime_val!r}."
                )

    # Deep merge runtime into algorithm config
    def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out

    return deep_merge(dict(algorithm_config), runtime_profile)


@dataclasses.dataclass(frozen=True)
class GateChainResult:
    is_valid: bool
    kaggle_report_sha256: str
    colab_t4_report_sha256: str


def verify_prior_gate_reports(
    kaggle_report: Optional[dict[str, Any]],
    colab_t4_report: Optional[dict[str, Any]],
    expected_sha: str,
    expected_dataset_hash: str,
    expected_config_hash: str,
) -> GateChainResult:
    """
    Verify upstream Kaggle dual-T4 and Colab single-T4 reports for A100 production run.
    Ensures verdicts are 'PASS' and SHA, dataset, and algorithm config hashes strictly match.
    """
    if not kaggle_report:
        raise GateChainValidationError("Kaggle T4x2 report missing. Upstream Gate B1.1 required.")

    if not colab_t4_report:
        raise GateChainValidationError("Colab T4 report missing. Upstream Gate B1.15 required.")

    for name, report in [("Kaggle T4x2", kaggle_report), ("Colab T4", colab_t4_report)]:
        verdict = report.get("verdict")
        if verdict != "PASS":
            raise GateChainValidationError(f"{name} gate did not pass (verdict: '{verdict}', expected: 'PASS').")

        devices = report.get("devices", [])
        if any("mock" in str(d).lower() for d in devices):
            raise GateChainValidationError(f"{name} gate report contains mock hardware devices: {devices}")

        r_sha = report.get("git_sha")
        if not r_sha or r_sha.lower() != expected_sha.lower():
            raise GateChainValidationError(
                f"{name} Git SHA mismatch! Report has '{r_sha}', expected '{expected_sha}'."
            )

        r_data = report.get("dataset_manifest_sha256")
        if not r_data or r_data != expected_dataset_hash:
            raise GateChainValidationError(
                f"{name} Dataset hash mismatch! Report has '{r_data}', expected '{expected_dataset_hash}'."
            )

        r_cfg = report.get("algorithm_config_sha256")
        if not r_cfg or r_cfg != expected_config_hash:
            raise GateChainValidationError(
                f"{name} Algorithm config hash mismatch! Report has '{r_cfg}', expected '{expected_config_hash}'."
            )

    k_hash = compute_canonical_json_hash(kaggle_report)
    c_hash = compute_canonical_json_hash(colab_t4_report)

    return GateChainResult(
        is_valid=True,
        kaggle_report_sha256=k_hash,
        colab_t4_report_sha256=c_hash,
    )

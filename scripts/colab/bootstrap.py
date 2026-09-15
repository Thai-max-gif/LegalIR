"""CPU-only Colab preflight and canonical Kaggle dataset acquisition."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.release.fingerprints import (
    CRITICAL_DATASET_FILES, assert_exact_git_sha, fingerprint_structured_config,
    verify_dataset_fingerprint, verify_prior_gate_reports,
)

REQUIRED_FILES = (*CRITICAL_DATASET_FILES, "manifest.json", "audit_report.json", "dataset_manifest.json")


def configure_kaggle_credentials():
    """Modern bearer tokens are not legacy username/key pairs."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    key = os.environ.get("KAGGLE_KEY", "")
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key
        del os.environ["KAGGLE_KEY"]
    elif key and not os.environ.get("KAGGLE_USERNAME"):
        raise RuntimeError("Legacy KAGGLE_KEY requires KAGGLE_USERNAME; use KAGGLE_API_TOKEN for modern tokens.")


def verify_launch(expected_sha, kaggle_report, freeze_file, repo_root=REPO_ROOT):
    """Reject stale gate evidence before allocating an expensive GPU VM.

    Kaggle dual-T4 (B1.1) is the sole pre-A100 hardware gate.
    """
    sha = assert_exact_git_sha(expected_sha, repo_root=repo_root)
    freeze = json.loads(Path(freeze_file).read_text(encoding="utf-8"))
    runtime_sha = str(freeze.get("git_sha", "")).strip().lower()
    if not runtime_sha:
        raise RuntimeError("Production freeze is missing git_sha!")
    if runtime_sha != sha.lower():
        # Two-commit model: release checkout may descend from the frozen runtime
        # with evidence-only diffs (gate reports, freeze, notebooks).
        from src.release.provenance import validate_runtime_release_lineage
        lineage_ok, lineage_errors = validate_runtime_release_lineage(runtime_sha, sha, repo_root)
        if not lineage_ok:
            raise RuntimeError(
                "Production freeze is for another runtime. Run the Kaggle T4 gate for this SHA and refresh approval before A100. "
                f"Details: {'; '.join(lineage_errors)}"
            )
    config_hash = fingerprint_structured_config(Path(repo_root) / "configs/algorithm/legalir_v2.yaml")
    if freeze.get("algorithm_config_sha256") != config_hash:
        raise RuntimeError("Production freeze algorithm config mismatch")
    verify_prior_gate_reports(
        kaggle_report=json.loads(Path(kaggle_report).read_text(encoding="utf-8")),
        expected_sha=runtime_sha,
        expected_dataset_hash=freeze["dataset"]["manifest_sha256"],
        expected_config_hash=config_hash,
    )
    return freeze


def prepare_dataset(dataset_dir: Path, freeze_file: Path | None = None) -> Path:
    dataset_dir = Path(dataset_dir)
    configure_kaggle_credentials()
    if not all((dataset_dir / name).is_file() for name in REQUIRED_FILES):
        dataset_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "phucdangg/legalir-task1-clean-data",
            "-p", str(dataset_dir), "--unzip", "--force",
        ], check=True)
    missing = [name for name in REQUIRED_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Canonical Kaggle dataset is incomplete: {missing}")
    freeze = json.loads(Path(freeze_file).read_text(encoding="utf-8")) if freeze_file else {}
    verify_dataset_fingerprint(
        dataset_dir,
        expected_manifest_hash=freeze.get("dataset", {}).get("manifest_sha256"),
        critical_files_expected=freeze.get("dataset", {}).get("critical_files"),
    )
    return dataset_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--kaggle-report", default="artifacts/task1/gates/kaggle_t4x2_report.json")
    parser.add_argument("--freeze-file", default="artifacts/task1/freeze/production_freeze.json")
    args = parser.parse_args()
    try:
        verify_launch(args.expected_sha, args.kaggle_report, args.freeze_file)
    except Exception as exc:
        print(f"[!] Launch blocked before GPU allocation: {exc}", file=sys.stderr)
        return 1
    print("[+] Runtime SHA, freeze, algorithm config, and upstream gate reports match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

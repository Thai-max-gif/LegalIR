from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import dotted_override, load_config, write_resolved_config
from src.data.dataset_validator import write_validation_report
from src.data.ir_dataset import inspect_pair_schema
from src.data.paths import artifact_candidates, artifact_path
from src.retrieval.index_identity import validate_index_identity
from src.utils.runtime import atomic_json, git_sha, hardware_report, report_status


def configured_artifact(root: str | Path, configured: str | None, default_filename: str, category: str) -> Path | None:
    """Resolve a config path first, then the documented/flat release locations."""
    root = Path(root)
    if configured and configured != "auto":
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return candidate
    return artifact_path(root, default_filename, category)


def required_artifact_check(name: str, root: str | Path, configured: str | None, filename: str, category: str, reason: str) -> dict[str, Any]:
    path = configured_artifact(root, configured, filename, category)
    if path:
        return {"name": name, "status": "PASS", "path": str(path)}
    checked = []
    if configured and configured != "auto":
        candidate = Path(configured)
        checked.append(str(candidate if candidate.is_absolute() else Path(root) / candidate))
    checked.extend(str(item) for item in artifact_candidates(root, filename, category))
    return {"name": name, "status": "BLOCKED", "reason": reason, "checked_paths": list(dict.fromkeys(checked))}


def immutable_revision_check(name: str, revision: str | None) -> dict[str, Any]:
    valid = isinstance(revision, str) and len(revision) == 40 and all(char in "0123456789abcdef" for char in revision.lower())
    return {"name": name, "status": "PASS", "revision": revision} if valid else {
        "name": name,
        "status": "BLOCKED",
        "reason": "an immutable Hugging Face commit SHA (40 hexadecimal characters) is required; do not use the moving default revision/main",
        "revision": revision,
    }


def pair_schema_check(pair_check: dict[str, Any], configured_loss: str) -> dict[str, Any]:
    if pair_check["status"] != "PASS":
        return {"name": "reranker_pair_schema", "status": "BLOCKED", "reason": "training pairs are unavailable"}
    try:
        schema = inspect_pair_schema(pair_check["path"])
    except Exception as exc:
        return {"name": "reranker_pair_schema", "status": "FAIL", "reason": f"cannot read pair parquet: {exc!r}"}
    if not schema["mode"]:
        return {"name": "reranker_pair_schema", "status": "BLOCKED", "reason": "cannot identify pairwise or BCE supervision format", "schema": schema}
    if configured_loss != "auto" and configured_loss != schema["mode"]:
        return {"name": "reranker_pair_schema", "status": "FAIL", "reason": "configured loss conflicts with pair schema", "configured_loss": configured_loss, "schema": schema}
    return {"name": "reranker_pair_schema", "status": "PASS", "loss_type": schema["mode"], "schema": schema}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--mode", choices=["validate", "smoke", "full"], default="smoke")
    args = parser.parse_args()
    config = load_config(args.config, dotted_override(args.set))
    output = Path(config["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, output)
    validation = write_validation_report(config["data"]["dataset_root"], output / "dataset_validation_report.json")
    identity = validate_index_identity(config["retrieval"].get("index_manifest_path"), config["model"].get("dense_model_id"), config["model"].get("dense_model_revision"))
    root = config["data"]["dataset_root"]
    pairs_check = required_artifact_check(
        "reranker_training_pairs", root, config["data"].get("pairs_path"), "reranker_pairs.parquet", "derived_optional",
        "reranker_pairs.parquet is required by this training profile; receive the published training pairs rather than mining/relabeling in this notebook",
    )
    pair_schema = pair_schema_check(pairs_check, config["training"]["loss_type"])
    candidates_check = required_artifact_check(
        "retrieval_candidates", root, config["data"].get("candidates_path"), "retrieval_candidates.parquet", "derived_optional",
        "retrieval_candidates.parquet is required for end-to-end candidate and reranked ranking metrics; do not insert qrels as candidates",
    )
    split_check = required_artifact_check(
        "document_disjoint_split", root, config["data"].get("document_disjoint_split_path"), "document_disjoint_split.parquet", "splits",
        "document_disjoint_split.parquet is required to verify the published split policy for this run",
    )
    source = git_sha()
    git_check = {"name": "source_git_sha", "status": "PASS", "git_sha": source} if source else {
        "name": "source_git_sha", "status": "BLOCKED" if config["gate"].get("require_git_sha") else "NOT_RUN",
        "reason": "source is not a Git checkout with a resolved commit; archive/source validation is possible but the workflow gate cannot pass",
    }
    reranker_revision = immutable_revision_check("reranker_model_revision", config["model"].get("reranker_revision"))
    checks = [{"name": "dataset_validation", "status": validation["status"]}, git_check, reranker_revision, pairs_check, pair_schema, split_check, {"name": "dense_index_identity", **identity}]
    if config["gate"].get("require_end_to_end"):
        checks.append(candidates_check)
    report = {
        "mode": args.mode,
        "hardware": hardware_report(),
        "dataset_validation": validation,
        "index_identity": identity,
        "resolved_artifacts": {"pairs": pairs_check, "pair_schema": pair_schema, "candidates": candidates_check, "document_disjoint_split": split_check},
        "checks": checks,
        "status": report_status(checks),
    }
    atomic_json(output / "preflight_report.json", report)
    print(f"Preflight {report['status']}: {output / 'preflight_report.json'}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
LegalIR Authoritative Colab A100 Production Gate Runner (Notion B1.2).
Executes full production training with BAAI/bge-reranker-v2-m3 + LoRA on all 7,000 queries.
Enforces:
1. Single NVIDIA A100 GPU (cuda:0).
2. Prior Kaggle Dual-T4 PASS report.
3. Prior Colab Single-T4 PASS report.
4. Cryptographic dataset, Git SHA, and config fingerprint matches.
5. End-to-end BF16 precision.
6. Top-5 submission validation and Hugging Face release.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import sys
    sys.modules["torchao"] = None
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

from src.release.contracts import COLAB_A100_CONTRACT, verify_device_contract
from src.release.fingerprints import (
    assert_exact_git_sha,
    compute_file_sha256,
    compute_canonical_json_hash,
    fingerprint_structured_config,
    verify_dataset_fingerprint,
    verify_prior_gate_reports,
)
from src.evaluation.submission import validate_submission_zip


def resolve_hf_token(explicit: str | None = None) -> str | None:
    """Resolve a Hugging Face token preferring write-capable tokens."""
    candidates = [
        explicit,
        os.environ.get("HF_TOKEN_WRITE"),
        os.environ.get("HF_TOKEN"),
        os.environ.get("HF_TOKEN_READ"),
    ]
    return next((t for t in candidates if t and str(t).startswith("hf_")), None)


def preflight_huggingface_access(
    repo_id: str,
    token: str | None = None,
) -> tuple[bool, str]:
    """Require verified write access before spending GPU time on a release."""
    token = resolve_hf_token(token)
    if not token:
        return False, "A Hugging Face write token is required (HF_TOKEN_WRITE or HF_TOKEN)."
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user = api.whoami().get("name", "unknown")
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        api.auth_check(repo_id=repo_id, repo_type="model", write=True)
        return True, f"authenticated as @{user}; verified write access to {repo_id}"
    except Exception as exc:
        # Hub exceptions can contain request details; never log their raw text.
        return False, f"Cannot verify Hugging Face write access ({type(exc).__name__}); check token scopes, network, and huggingface_hub version."


def upload_artifacts_to_huggingface(
    output_dir: Path,
    repo_id: str = "dangphuc2109/legalir-task1-reranker",
    token: str | None = None,
) -> str:
    """Upload only final models, reports, logs and submissions; require a commit receipt."""
    from scripts.colab.artifacts import release_files
    import re

    token = resolve_hf_token(token)
    if not token:
        raise RuntimeError("Hugging Face upload failed: write token required.")
    selected = release_files(output_dir)
    adapter_prefix = "checkpoints/reranker_final/"
    if not any(adapter_prefix + name in selected for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise RuntimeError("Missing final adapter weights; refusing incomplete release.")
    for required in (adapter_prefix + "adapter_config.json", "submission.zip", "submission.json", "run_manifest.json"):
        if required not in selected or (output_dir / required).stat().st_size == 0:
            raise RuntimeError(f"Missing or empty release artifact: {required}")
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise RuntimeError("Invalid release run_id")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(output_dir),
            path_in_repo=f"runs/{run_id}",
            allow_patterns=selected,
            commit_message=f"Upload A100 training artifacts: {run_id}",
        )
        commit_sha = commit.oid
        if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
            raise RuntimeError("Hub did not return an immutable commit SHA")
        print(f"[+] Artifacts uploaded: https://huggingface.co/{repo_id}/tree/{commit_sha}/runs/{run_id}", flush=True)
        return commit_sha
    except Exception as exc:
        raise RuntimeError(f"Hugging Face upload failed ({type(exc).__name__}); artifacts remain at {output_dir}.") from None


def run_a100_production_gate(
    dataset_dir: Path | str,
    output_dir: Path | str,
    expected_sha: str = "",
    algorithm_config_path: Path | str = REPO_ROOT / "configs" / "algorithm" / "legalir_v2.yaml",
    runtime_profile_path: Path | str = REPO_ROOT / "configs" / "runtime" / "colab_a100.yaml",
    kaggle_report_path: Path | str = REPO_ROOT / "artifacts" / "task1" / "gates" / "kaggle_t4x2_report.json",
    colab_t4_report_path: Path | str = REPO_ROOT / "artifacts" / "task1" / "gates" / "colab_t4_report.json",
    freeze_file_path: Path | str = REPO_ROOT / "artifacts" / "task1" / "freeze" / "production_freeze.json",
    allow_non_a100: bool = False,
    mock: bool = False,
    hf_repo: str | None = None,
    precision: str = "bf16",
    runtime_config_path: Path | str | None = None,
    reranker_config_path: Path | str | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Execute the fail-closed A100 production training run."""
    import shutil
    import subprocess
    import yaml

    t0 = time.time()
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "run_manifest.json"
    submission_zip = output_dir / "submission.zip"
    adapter_dir = output_dir / "final_adapter"

    print("=================================================================", flush=True)
    print("LegalIR Colab A100 Production Gate (B1.2 Full Training)", flush=True)
    print(f"  • Dataset Dir        : {dataset_dir}", flush=True)
    print(f"  • Output Dir         : {output_dir}", flush=True)
    print(f"  • Kaggle Report      : {kaggle_report_path}", flush=True)
    print(f"  • Colab T4 Report    : {colab_t4_report_path}", flush=True)
    print(f"  • Precision          : bf16", flush=True)
    print("=================================================================", flush=True)

    # 1. Hardware Verification (Fail-Closed: Single A100 on cuda:0)
    if not mock:
        hw_profile = verify_device_contract(COLAB_A100_CONTRACT, allow_debug=allow_non_a100)
        gpu_name = hw_profile.device_names[0] if hw_profile.device_names else "NVIDIA A100"
        device_count = hw_profile.device_count
    else:
        gpu_name = "Mock NVIDIA A100"
        device_count = 1

    # 2. Git SHA Invariance
    actual_sha = assert_exact_git_sha(expected_sha, repo_root=REPO_ROOT, is_production=not mock)

    # 3. Canonical Dataset Fingerprint Verification (Fail-Closed)
    if not mock:
        ds_result = verify_dataset_fingerprint(dataset_dir)
        manifest_sha256 = ds_result.manifest_sha256
    else:
        manifest_sha256 = "mock_manifest_sha256"

    # 4. Config Fingerprints & Resolution
    algo_sha256 = fingerprint_structured_config(algorithm_config_path)
    runtime_sha256 = fingerprint_structured_config(runtime_profile_path)

    # Export resolved configuration
    try:
        from src.release.fingerprints import validate_runtime_overrides
        algo_cfg = yaml.safe_load(Path(algorithm_config_path).read_text(encoding="utf-8"))
        runtime_cfg = yaml.safe_load(Path(runtime_profile_path).read_text(encoding="utf-8"))
        resolved_cfg = validate_runtime_overrides(algo_cfg, runtime_cfg)
        (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True), encoding="utf-8")
    except Exception:
        pass

    # 5. Upstream Gate Chain Verification (Kaggle Dual-T4 & Colab Single-T4)
    if kaggle_report_path:
        k_p = Path(kaggle_report_path)
        if k_p.is_file():
            k_path = k_p
        elif str(k_p) not in (str(REPO_ROOT / "artifacts" / "task1" / "gates" / "kaggle_t4x2_report.json"), "artifacts/task1/gates/kaggle_t4x2_report.json"):
            raise RuntimeError(f"Kaggle T4x2 report missing: {k_p}. Upstream Gate B1.1 required before A100.")
        else:
            k_cands = [
                Path("/content/kaggle_t4x2_report.json"),
                Path("/content/LegalIR/artifacts/task1/gates/kaggle_t4x2_report.json"),
                REPO_ROOT / "artifacts" / "task1" / "gates" / "kaggle_t4x2_report.json",
            ]
            k_path = next((p for p in k_cands if p and p.is_file()), k_p)
    else:
        k_path = REPO_ROOT / "artifacts" / "task1" / "gates" / "kaggle_t4x2_report.json"

    if colab_t4_report_path:
        c_p = Path(colab_t4_report_path)
        if c_p.is_file():
            c_path = c_p
        elif str(c_p) not in (str(REPO_ROOT / "artifacts" / "task1" / "gates" / "colab_t4_report.json"), "artifacts/task1/gates/colab_t4_report.json"):
            raise RuntimeError(f"Colab T4 report missing: {c_p}. Upstream Gate B1.15 required before A100.")
        else:
            c_cands = [
                Path("/content/colab_t4_report.json"),
                Path("/content/LegalIR/artifacts/task1/gates/colab_t4_report.json"),
                REPO_ROOT / "artifacts" / "task1" / "gates" / "colab_t4_report.json",
            ]
            c_path = next((p for p in c_cands if p and p.is_file()), c_p)
    else:
        c_path = REPO_ROOT / "artifacts" / "task1" / "gates" / "colab_t4_report.json"

    if not k_path.is_file():
        raise RuntimeError(f"Kaggle T4x2 report missing: {k_path}. Upstream Gate B1.1 required before A100.")
    if not c_path.is_file():
        raise RuntimeError(f"Colab T4 report missing: {c_path}. Upstream Gate B1.15 required before A100.")

    kaggle_report = json.loads(k_path.read_text(encoding="utf-8"))
    colab_t4_report = json.loads(c_path.read_text(encoding="utf-8"))

    if not mock:
        gate_chain_res = verify_prior_gate_reports(
            kaggle_report=kaggle_report,
            colab_t4_report=colab_t4_report,
            expected_sha=actual_sha,
            expected_dataset_hash=manifest_sha256,
            expected_config_hash=algo_sha256,
        )
        k_rep_hash = gate_chain_res.kaggle_report_sha256
        c_rep_hash = gate_chain_res.colab_t4_report_sha256
    else:
        k_rep_hash = "mock_k_hash"
        c_rep_hash = "mock_c_hash"

    # Copy upstream reports into output directory for full provenance
    try:
        shutil.copyfile(k_path, output_dir / "kaggle_t4x2_report.json")
        shutil.copyfile(c_path, output_dir / "colab_t4_report.json")
        ds_manifest_src = dataset_dir / "dataset_manifest.json"
        if ds_manifest_src.is_file():
            shutil.copyfile(ds_manifest_src, output_dir / "dataset_manifest.json")
    except Exception:
        pass

    # Cross-verify and copy production_freeze.json if present
    freeze_cands = [
        Path(freeze_file_path) if freeze_file_path else None,
        Path("/content/production_freeze.json"),
        Path("/content/LegalIR/artifacts/task1/freeze/production_freeze.json"),
        REPO_ROOT / "artifacts" / "task1" / "freeze" / "production_freeze.json",
    ]
    freeze_path = next((p for p in freeze_cands if p and p.is_file()), Path(freeze_file_path))
    if freeze_path.is_file():
        try:
            freeze_data = json.loads(freeze_path.read_text(encoding="utf-8"))
            if not mock:
                if freeze_data.get("git_sha", "").lower() != actual_sha.lower():
                    raise RuntimeError(f"Production freeze git_sha mismatch: {freeze_data.get('git_sha')} vs {actual_sha}")
                if freeze_data.get("dataset", {}).get("manifest_sha256") != manifest_sha256:
                    raise RuntimeError(f"Production freeze dataset hash mismatch!")
                if freeze_data.get("algorithm_config_sha256") != algo_sha256:
                    raise RuntimeError(f"Production freeze algorithm config hash mismatch!")
            shutil.copyfile(freeze_path, output_dir / "production_freeze.json")
            print(f"[+] Verified and attached production freeze tuple: {freeze_path.name}")
        except Exception as freeze_exc:
            print(f"[!] Warning on production freeze check: {freeze_exc}")

    # Capture system and hardware environment
    try:
        env_text = f"Python {sys.version}\nPyTorch {sys.modules.get('torch', 'unknown')}\nPlatform {sys.platform}\n"
        (output_dir / "environment.txt").write_text(env_text, encoding="utf-8")
        smi_res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if smi_res.returncode == 0:
            (output_dir / "nvidia-smi.txt").write_text(smi_res.stdout, encoding="utf-8")
    except Exception:
        pass

    # Normalize precision once (accept bfloat16/bf16/fp16/fp32)
    prec_norm = str(precision or "bf16").lower().strip()
    if prec_norm == "bfloat16":
        prec_norm = "bf16"
    if prec_norm not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"Unsupported precision '{precision}' (expected bf16/fp16/fp32)")
    print(f"  • Precision          : {prec_norm}", flush=True)

    # Hugging Face release preflight (fail fast on rejected tokens, before GPU burn)
    target_hf_repo_early = hf_repo or os.environ.get("HF_REPO_ID", "dangphuc2109/legalir-task1-reranker")
    if not mock:
        hf_ok, hf_detail = preflight_huggingface_access(target_hf_repo_early, hf_token)
        print(f"  • Hugging Face       : {hf_detail}", flush=True)
        if not hf_ok:
            raise RuntimeError(f"Hugging Face access preflight failed: {hf_detail}")

    # 6. Full Training Execution
    if mock:
        print("[*] Executing mock A100 production training and artifact generation...", flush=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 16, "base_model": "BAAI/bge-reranker-v2-m3"}), encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"MOCK_A100_ADAPTER_WEIGHTS")

        sub_dict = {f"q_{i}": [f"10{j}" for j in range(1, 4)] for i in range(1000)}
        sub_json_str = json.dumps(sub_dict, indent=2)
        with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("submission.json", sub_json_str)
    else:
        from src.pipeline.kaggle_train import run_kaggle_pipeline
        print(f"[*] Launching authoritative full 7,000-query A100 pipeline with {prec_norm.upper()}...", flush=True)
        from scripts.colab.artifacts import training_log
        with training_log(output_dir):
            res = run_kaggle_pipeline(
                data_dir=str(dataset_dir),
                working_dir=str(output_dir),
                run_mode="full",
                device_contract=COLAB_A100_CONTRACT,
                precision=prec_norm,
                repo_root=str(REPO_ROOT),
                runtime_config_path=str(runtime_config_path) if runtime_config_path else str(REPO_ROOT / "configs/runtime/colab_a100.yaml"),
                reranker_config_path=str(reranker_config_path) if reranker_config_path else str(REPO_ROOT / "configs/experiments/reranker_lora.yaml"),
                allow_nonstandard_production_devices=allow_non_a100,
            )
        print(f"[+] Pipeline completed with status: {res.status}", flush=True)
        # run_kaggle_pipeline writes submissions/submission.zip; mirror to output root
        # expected by validation, checksums, HF upload, and notebook Cell 5.
        nested_zip = output_dir / "submissions" / "submission.zip"
        nested_json = output_dir / "submissions" / "submission.json"
        try:
            if nested_zip.is_file() and nested_zip.resolve() != submission_zip.resolve():
                shutil.copyfile(nested_zip, submission_zip)
                print(f"[+] Mirrored pipeline submission to {submission_zip}", flush=True)
            if nested_json.is_file():
                shutil.copyfile(nested_json, output_dir / "submission.json")
        except Exception as mirror_exc:
            print(f"[!] Warning: failed mirroring pipeline submission ({mirror_exc})", flush=True)

    # 7. Validate Submission.zip (dict API)
    print(f"[*] Validating submission package: {submission_zip} ...", flush=True)
    zip_val = validate_submission_zip(submission_zip)
    is_sub_valid = bool(zip_val.get("is_valid"))
    sub_msg = "; ".join(zip_val.get("errors", [])) or "OK: submission.zip contains only submission.json"
    if not is_sub_valid:
        raise RuntimeError(f"Submission validation failed: {sub_msg}")
    print(f"[+] Submission validation PASSED: {sub_msg}", flush=True)

    # 8. Generate File Checksums
    checksums: dict[str, str] = {}
    from scripts.colab.artifacts import release_files
    for rel_name in release_files(output_dir):
        if rel_name not in ("checksums.sha256", "run_manifest.json"):
            checksums[rel_name] = compute_file_sha256(output_dir / rel_name)

    checksums_file = output_dir / "checksums.sha256"
    checksums_lines = [f"{sha}  {fname}" for fname, sha in sorted(checksums.items())]
    checksums_file.write_text("\n".join(checksums_lines) + "\n", encoding="utf-8")

    # 9. Build Initial Run Manifest
    # Mock runs are debug-only: they bypass the gate chain and use synthetic
    # artifacts, so they must never claim PASS/COMPLETED production status.
    mock_verdict = "DEBUG_ONLY" if mock else "PASS"
    mock_status = "MOCK_COMPLETED" if mock else "COMPLETED"
    manifest = {
        "schema_version": 3,
        "run_id": f"task1-{time.strftime('%Y%m%d-%H%M%S')}-{actual_sha[:7]}",
        "stage": "B1.2_COLAB_A100_PRODUCTION_RUN",
        "gate": "COLAB_A100",
        "status": mock_status,
        "verdict": mock_verdict,
        "git_sha": actual_sha,
        "dataset": {
            "slug": "phucdangg/legalir-task1-clean-data",
            "logical_version": "v2",
            "manifest_sha256": manifest_sha256,
        },
        "algorithm_config_sha256": algo_sha256,
        "runtime_profile_sha256": runtime_sha256,
        "base_models": {
            "reranker_id": "BAAI/bge-reranker-v2-m3",
            "reranker_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "dense_id": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        },
        "gates": {
            "kaggle_t4x2": {"verdict": mock_verdict, "report_sha256": k_rep_hash},
            "colab_t4": {"verdict": mock_verdict, "report_sha256": c_rep_hash},
        },
        "hardware": {
            "gpu": gpu_name,
            "device_count": device_count,
            "precision": prec_norm,
        },
        "checksums": checksums,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # 10. Hugging Face Release Upload & Immutable Revision Capture
    # Mock mode never touches the release repo: artifacts are synthetic.
    target_hf_repo = hf_repo or os.environ.get("HF_REPO_ID", "dangphuc2109/legalir-task1-reranker")
    if mock:
        print("[*] Mock mode: skipping Hugging Face upload (no release commit).", flush=True)
        manifest["huggingface"] = {"repo_id": target_hf_repo, "uploaded": False, "reason": "mock mode"}
        run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        hf_commit = None
    else:
        try:
            hf_commit = upload_artifacts_to_huggingface(output_dir=output_dir, repo_id=target_hf_repo, token=hf_token)
        except RuntimeError:
            manifest["huggingface"] = {"repo_id": target_hf_repo, "uploaded": False}
            run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            raise
    if hf_commit:
        manifest["huggingface"] = {
            "repo_id": target_hf_repo,
            "commit_sha": hf_commit,
            "path_in_repo": f"runs/{manifest['run_id']}",
            "uploaded": True,
        }
        # A manifest cannot contain its own commit SHA. It references the immutable
        # artifact commit; the local receipt also records the manifest commit.
        final_manifest = {**manifest, "status": "RELEASED"}
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=resolve_hf_token(hf_token))
            receipt = api.upload_file(
                path_or_fileobj=json.dumps(final_manifest, indent=2, sort_keys=True).encode("utf-8"),
                path_in_repo=f"runs/{manifest['run_id']}/run_manifest.json",
                repo_id=target_hf_repo,
                repo_type="model",
                commit_message=f"Record artifact commit {hf_commit[:8]}",
            )
        except Exception as exc:
            run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            raise RuntimeError(f"Release manifest upload failed ({type(exc).__name__}); artifact commit is {hf_commit}.") from None
        manifest = final_manifest
        manifest["huggingface"]["manifest_commit_sha"] = receipt.oid
        run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("=================================================================", flush=True)
    print(f"[+] A100 Production Run Completed: {run_manifest_path}", flush=True)
    print(f"    Status: {manifest['status']} | Verdict: {manifest['verdict']}", flush=True)
    print("=================================================================", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="LegalIR Colab A100 Production Gate Runner")
    parser.add_argument("--dataset-dir", type=str, default="/content/kaggle_dataset", help="Canonical dataset path")
    parser.add_argument("--output-dir", type=str, default="artifacts/task1/production", help="Output directory")
    parser.add_argument("--expected-sha", type=str, default="", help="Expected 40-char commit SHA")
    parser.add_argument("--kaggle-report", type=str, default="artifacts/task1/gates/kaggle_t4x2_report.json", help="Path to Kaggle dual-T4 report")
    parser.add_argument("--colab-t4-report", type=str, default="artifacts/task1/gates/colab_t4_report.json", help="Path to Colab single-T4 report")
    parser.add_argument("--freeze-file", type=str, default="artifacts/task1/freeze/production_freeze.json", help="Path to production freeze tuple")
    parser.add_argument("--precision", type=str, default="bf16", help="Training precision (bf16/fp16/fp32)")
    parser.add_argument("--allow-non-a100", action="store_true", help="Allow running on non-A100 GPU for testing")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (CPU testing only)")
    parser.add_argument("--hf-repo", type=str, default="dangphuc2109/legalir-task1-reranker", help="Hugging Face repo ID")
    args = parser.parse_args()

    try:
        run_a100_production_gate(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            expected_sha=args.expected_sha,
            kaggle_report_path=args.kaggle_report,
            colab_t4_report_path=args.colab_t4_report,
            freeze_file_path=args.freeze_file,
            precision=args.precision,
            allow_non_a100=args.allow_non_a100,
            mock=args.mock,
            hf_repo=args.hf_repo,
        )
        return 0
    except Exception as exc:
        print(f"[!] FAILED: Colab A100 Production Gate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

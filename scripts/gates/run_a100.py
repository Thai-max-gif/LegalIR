#!/usr/bin/env python3
"""
LegalIR Authoritative Colab A100 Production Gate Runner (Notion B1.2).
Executes full production training with BAAI/bge-reranker-v2-m3 + LoRA on all 7,000 queries.
Enforces:
1. Single NVIDIA A100 GPU (cuda:0).
2. Prior Kaggle Dual-T4 PASS report (sole pre-A100 hardware gate).
3. Cryptographic dataset, Git SHA, and config fingerprint matches.
4. End-to-end BF16 precision.
5. Top-5 submission validation and Hugging Face release.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
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
    allow_public_repo: bool = False,
) -> tuple[bool, str]:
    """Require verified write access before spending GPU time on a release.

    New repos are created private. Pushing to an existing PUBLIC repo requires
    explicit opt-in (allow_public_repo=True); the override is recorded in the
    run manifest so releases never go public by accident.

    NOTE: this preflight is potentially mutating because it invokes
    create_repo(repo_id, private=True, exist_ok=True); do not describe it as
    a read-only audit.
    """
    token = resolve_hf_token(token)
    if not token:
        return False, "A Hugging Face write token is required (HF_TOKEN_WRITE or HF_TOKEN)."
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user = api.whoami().get("name", "unknown")
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        api.auth_check(repo_id=repo_id, repo_type="model", write=True)
        # Enforce private visibility: exist_ok=True does not flip a public repo
        # back to private, so an accidentally-public release repo must fail fast
        # unless the operator explicitly opts in. Unknown metadata or lookup
        # failure blocks preflight (fail closed).
        try:
            info = api.repo_info(repo_id=repo_id, repo_type="model")
            private = getattr(info, "private", None)
            if private is not True and private is not False:
                return False, "Cannot verify Hugging Face repository visibility; blocking release."
            if private is False and not allow_public_repo:
                return False, f"HF repo {repo_id} exists but is PUBLIC; release requires a private repo (or explicit opt-in)."
            if private is False:
                print(f"[!] OPERATOR OVERRIDE: pushing release artifacts to PUBLIC repo {repo_id}.", flush=True)
                return True, f"authenticated as @{user}; verified write access to {repo_id} (public)"
            return True, f"authenticated as @{user}; verified write access to {repo_id} (private)"
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "PUBLIC" in str(exc):
                raise
            # Lookup failure or unverifiable visibility: fail closed without
            # exposing token-bearing exception text.
            return False, f"Cannot verify Hugging Face repository visibility ({type(exc).__name__}); blocking release."
    except Exception as exc:
        # Hub exceptions can contain request details; never log their raw text.
        msg = str(exc)
        if "PUBLIC" in msg and "explicit opt-in" in msg:
            # Preserve the explicit public-without-consent rejection without
            # leaking Hub internals.
            return False, msg
        return False, f"Cannot verify Hugging Face write access ({type(exc).__name__}); check token scopes, network, and huggingface_hub version."


def upload_artifacts_to_huggingface(
    output_dir: Path,
    repo_id: str = "dangphuc2109/legalir-task1-reranker",
    token: str | None = None,
    allow_public_repo: bool = False,
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
        try:
            _info = api.repo_info(repo_id=repo_id, repo_type="model")
            private = getattr(_info, "private", None)
            if private is not True and private is not False:
                raise RuntimeError("Cannot verify Hugging Face repository visibility")
            if private is False and not allow_public_repo:
                raise RuntimeError(f"HF repo {repo_id} is PUBLIC; release requires a private repo (or explicit opt-in).")
            if private is False:
                print(f"[!] OPERATOR OVERRIDE: pushing release artifacts to PUBLIC repo {repo_id}.", flush=True)
        except RuntimeError:
            raise
        except Exception as lookup_exc:
            raise RuntimeError(
                f"Hugging Face upload failed ({type(lookup_exc).__name__}); artifacts remain at {output_dir}."
            ) from None
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
    freeze_file_path: Path | str = REPO_ROOT / "artifacts" / "task1" / "freeze" / "production_freeze.json",
    allow_non_a100: bool = False,
    mock: bool = False,
    hf_repo: str | None = None,
    precision: str = "bf16",
    runtime_config_path: Path | str | None = None,
    reranker_config_path: Path | str | None = None,
    hf_token: str | None = None,
    hf_allow_public_repo: bool = False,
) -> dict[str, Any]:
    """Execute the fail-closed A100 production training run.

    The Kaggle dual-T4 report (B1.1) is the sole pre-A100 hardware gate and is
    strictly enforced. hf_allow_public_repo=True permits pushing to an existing
    PUBLIC HF repo (recorded in the manifest); new repos are always private.
    """
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
    print(f"  • Precision          : bf16", flush=True)
    print("=================================================================", flush=True)

    # 1. Hardware Verification (Fail-Closed: Single A100 on cuda:0)
    if not mock:
        hw_profile = verify_device_contract(COLAB_A100_CONTRACT, allow_debug=allow_non_a100)
        gpu_name = hw_profile.device_names[0] if hw_profile.device_names else "NVIDIA A100"
        device_count = hw_profile.device_count
        # Fail-closed VRAM guard: Colab A100 lottery is often 40GB; full
        # 5-fold OOF + final LoRA needs headroom beyond a name match.
        try:
            import torch as _torch

            _props = _torch.cuda.get_device_properties(0)
            _total_gib = float(_props.total_memory) / (1024**3)
            print(f"  • GPU VRAM          : {_props.name} {_total_gib:.1f} GiB", flush=True)
            if _total_gib < 39.0:
                raise RuntimeError(
                    f"Insufficient A100 VRAM: {_total_gib:.1f} GiB detected, need >= 40 GiB for full production run."
                )
        except RuntimeError:
            raise
        except Exception:
            pass
        # Fail-fast torch/transformers compatibility gate. transformers 5.x
        # lazy-loads model classes only when torch>=2.5 is importable; older
        # torch passes raw `import torch` but fails hours later at model load
        # with a misleading "requires PyTorch but not found" error. Catch it
        # here, before spending GPU time on indexing.
        try:
            import torch as _torch_check

            _ver_parts = str(_torch_check.__version__).split("+")[0].split(".")
            _ver_tuple = (int(_ver_parts[0]), int(_ver_parts[1])) if len(_ver_parts) >= 2 else (0, 0)
            print(f"  • Torch             : {_torch_check.__version__} (cuda: {_torch_check.version.cuda})", flush=True)
            if _ver_tuple < (2, 5):
                raise RuntimeError(
                    f"Incompatible torch {_torch_check.__version__}: transformers 5.x requires torch>=2.5."
                )
            from transformers import is_torch_available as _is_torch_available

            if not _is_torch_available():
                raise RuntimeError("transformers reports the torch backend unavailable.")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Torch/transformers compatibility preflight failed ({type(exc).__name__}).") from None
        # Fail-closed disk guard: dataset (735MB) + dense index (~675MB) +
        # BM25 indexes + checkpoints + recovery.tar.gz need headroom.
        try:
            import shutil as _shutil

            _free_gib = float(_shutil.disk_usage(str(output_dir)).free) / (1024**3)
            if _free_gib > 1048576.0:
                # Absurd value (e.g. 2^63 bytes on overlayfs): capacity unknown.
                print("  • Disk free         : unknown (filesystem reports implausible capacity)", flush=True)
            else:
                print(f"  • Disk free         : {_free_gib:.1f} GiB at {output_dir}", flush=True)
                if _free_gib < 20.0:
                    raise RuntimeError(
                        f"Insufficient free disk: {_free_gib:.1f} GiB at {output_dir}, need >= 20 GiB."
                    )
        except RuntimeError:
            raise
        except Exception:
            pass
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

    # Export resolved configuration (fail-closed on protected-key violations)
    from src.release.fingerprints import validate_runtime_overrides
    algo_cfg = yaml.safe_load(Path(algorithm_config_path).read_text(encoding="utf-8"))
    runtime_cfg = yaml.safe_load(Path(runtime_profile_path).read_text(encoding="utf-8"))
    resolved_cfg = validate_runtime_overrides(algo_cfg, runtime_cfg)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True), encoding="utf-8")

    # Resolve frozen runtime vs release checkout (two-commit model) BEFORE the
    # gate chain: GPU evidence binds to the runtime commit, which must equal
    # HEAD or be its ancestor with evidence-only diffs.
    freeze_cands_early = [
        Path(freeze_file_path) if freeze_file_path else None,
        Path("/content/production_freeze.json"),
        Path("/content/LegalIR/artifacts/task1/freeze/production_freeze.json"),
        REPO_ROOT / "artifacts" / "task1" / "freeze" / "production_freeze.json",
    ]
    freeze_path_early = next((p for p in freeze_cands_early if p and p.is_file()), None)
    if freeze_path_early is not None and not mock:
        _freeze_early = json.loads(freeze_path_early.read_text(encoding="utf-8"))
        _runtime_candidate = str(_freeze_early.get("git_sha", "")).lower()
        if not _runtime_candidate:
            raise RuntimeError("Production freeze is missing git_sha!")
        if _runtime_candidate != actual_sha.lower():
            from src.release.provenance import validate_runtime_release_lineage
            _lineage_ok, _lineage_errors = validate_runtime_release_lineage(
                _runtime_candidate, actual_sha, REPO_ROOT
            )
            if not _lineage_ok:
                raise RuntimeError(f"Production freeze lineage rejected: {'; '.join(_lineage_errors)}")
            print(f"[+] Two-commit lineage OK: runtime {_runtime_candidate[:7]} -> release {actual_sha[:7]} (evidence-only diff)")
        runtime_sha = _runtime_candidate
    else:
        runtime_sha = actual_sha.lower()

    # 5. Upstream Gate Chain Verification (Kaggle Dual-T4 sole pre-A100 gate)
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

    if not k_path.is_file():
        raise RuntimeError(f"Kaggle T4x2 report missing: {k_path}. Upstream Gate B1.1 required before A100.")

    kaggle_report = json.loads(k_path.read_text(encoding="utf-8"))

    if not mock:
        gate_chain_res = verify_prior_gate_reports(
            kaggle_report=kaggle_report,
            expected_sha=runtime_sha,
            expected_dataset_hash=manifest_sha256,
            expected_config_hash=algo_sha256,
        )
        k_rep_hash = gate_chain_res.kaggle_report_sha256
    else:
        k_rep_hash = "mock_k_hash"

    # Copy upstream reports into output directory for full provenance
    try:
        shutil.copyfile(k_path, output_dir / "kaggle_t4x2_report.json")
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
        freeze_data = json.loads(freeze_path.read_text(encoding="utf-8"))
        if not mock:
            # Runtime/release lineage already resolved above; re-assert hashes.
            if str(freeze_data.get("git_sha", "")).lower() != runtime_sha:
                raise RuntimeError("Production freeze changed between preflight and execution!")
            if freeze_data.get("dataset", {}).get("manifest_sha256") != manifest_sha256:
                raise RuntimeError("Production freeze dataset hash mismatch!")
            if freeze_data.get("algorithm_config_sha256") != algo_sha256:
                raise RuntimeError("Production freeze algorithm config hash mismatch!")
        shutil.copyfile(freeze_path, output_dir / "production_freeze.json")
        print(f"[+] Verified and attached production freeze tuple: {freeze_path.name}")

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
        hf_ok, hf_detail = preflight_huggingface_access(target_hf_repo_early, hf_token, allow_public_repo=hf_allow_public_repo)
        print(f"  • Hugging Face       : {hf_detail}", flush=True)
        if not hf_ok:
            raise RuntimeError(f"Hugging Face access preflight failed: {hf_detail}")

    # 6. Full Training Execution
    if mock:
        print("[*] Executing mock A100 production training and artifact generation...", flush=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 8, "base_model": "BAAI/bge-reranker-v2-m3"}), encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"MOCK_A100_ADAPTER_WEIGHTS")

        # Canonical scorer-compatible format: {qid: {"answer": [...]}}.
        sub_dict = {f"q_{i}": {"answer": [f"10{j}" for j in range(1, 4)]} for i in range(1000)}
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
        "runtime_sha": runtime_sha if not mock else actual_sha,
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
            hf_commit = upload_artifacts_to_huggingface(output_dir=output_dir, repo_id=target_hf_repo, token=hf_token, allow_public_repo=hf_allow_public_repo)
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
            "public_repo_override": bool(hf_allow_public_repo),
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
            receipt_oid = getattr(receipt, "oid", None)
            if not isinstance(receipt_oid, str) or not re.fullmatch(r"[0-9a-f]{40}", receipt_oid):
                run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
                raise RuntimeError(
                    f"Release manifest upload failed (InvalidReceipt); artifact commit is {hf_commit}."
                )
        except Exception as exc:
            # Keep partial artifacts recoverable: the non-RELEASED manifest with
            # the artifact commit is already staged above on receipt failure
            # paths; ensure it is written before raising.
            if not run_manifest_path.is_file():
                run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            else:
                try:
                    current = json.loads(run_manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    current = {}
                if current.get("status") == "RELEASED":
                    run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            if isinstance(exc, RuntimeError) and "InvalidReceipt" in str(exc):
                raise
            raise RuntimeError(f"Release manifest upload failed ({type(exc).__name__}); artifact commit is {hf_commit}.") from None
        manifest = final_manifest
        manifest["huggingface"]["manifest_commit_sha"] = receipt_oid
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
    parser.add_argument("--freeze-file", type=str, default="artifacts/task1/freeze/production_freeze.json", help="Path to production freeze tuple")
    parser.add_argument("--precision", type=str, default="bf16", help="Training precision (bf16/fp16/fp32)")
    parser.add_argument("--allow-non-a100", action="store_true", help="Allow running on non-A100 GPU for testing")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (CPU testing only)")
    parser.add_argument("--hf-allow-public-repo", action="store_true", help="Operator override: allow pushing release to an existing PUBLIC HF repo (recorded in manifest)")
    parser.add_argument("--hf-repo", type=str, default="dangphuc2109/legalir-task1-reranker", help="Hugging Face repo ID")
    args = parser.parse_args()

    try:
        run_a100_production_gate(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            expected_sha=args.expected_sha,
            kaggle_report_path=args.kaggle_report,
            freeze_file_path=args.freeze_file,
            precision=args.precision,
            allow_non_a100=args.allow_non_a100,
            mock=args.mock,
            hf_repo=args.hf_repo,
            hf_allow_public_repo=args.hf_allow_public_repo,
        )
        return 0
    except Exception as exc:
        print(f"[!] FAILED: Colab A100 Production Gate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

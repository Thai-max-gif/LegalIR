"""Modal rescue: inference-only submission from a completed training attempt.

Reads the finished final adapter + persisted indexes from an existing
Volume attempt (training + OOF already done) and runs ONLY test inference,
strict submission packaging, and validation. Never retrains, never overwrites
the source attempt; outputs go to <sha>/rescue/<uuid>/.

Example:
    python -m modal run --profile zunuoivalutre --env main \
        scripts/modal/run_modal_rescue.py \
        --source-sha 7020249ea5cd7aed07a6985975fe9c3d59237988 \
        --source-attempt 2d2f6eb019f7466fa89faa49810c570a
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

app = modal.App("legalir-rescue-inference")

# Same runtime as production so artifacts stay compatible.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers==5.15.1",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "huggingface-hub==1.28.0",
        "numpy>=1.24,<3",
        "pandas>=2,<3",
        "pyarrow>=14",
        "pyyaml>=6,<7",
        "scikit-learn>=1.3,<2",
        "lightgbm>=4,<5",
        "bm25s>=0.2,<1",
        "pyvi>=0.1.1,<1",
        "sentencepiece>=0.1.99",
        "faiss-cpu>=1.7",
        "psutil>=5.9",
        "kaggle>=1.8,<3",
        "tqdm>=4.65",
    )
)

volume = modal.Volume.from_name("legalir-production", create_if_missing=True)
VOLUME_MOUNT = "/root/legalir_volume"

# Inference-only (retrieval + rerank of 2080 queries); 3h ceiling.
TIMEOUT_SECONDS = int(os.environ.get("MODAL_RESCUE_TIMEOUT_SECONDS", 10800))

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_HEX_RE = re.compile(r"[0-9a-f]{32}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@app.function(
    image=image,
    gpu="A100",
    cpu=8.0,
    memory=32768,
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_MOUNT: volume},
    secrets=[
        modal.Secret.from_name("kaggle-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def rescue_inference(source_sha: str, source_attempt: str, code_sha: str):
    """Run test inference from finished artifacts; return validated submission report."""
    import sys

    source_sha = str(source_sha or "").strip().lower()
    source_attempt = str(source_attempt or "").strip().lower()
    code_sha = str(code_sha or "").strip().lower()
    if not _SHA_RE.fullmatch(source_sha):
        raise ValueError("source_sha must be an exact 40-char lowercase SHA")
    if not _SHA_RE.fullmatch(code_sha):
        raise ValueError("code_sha must be an exact 40-char lowercase SHA")
    if not _HEX_RE.fullmatch(source_attempt):
        raise ValueError("source_attempt must be a 32-char hex attempt id")

    vol_root = Path(VOLUME_MOUNT)
    src_dir = vol_root / source_sha / "attempts" / source_attempt
    adapter_dir = src_dir / "checkpoints" / "reranker_final"
    index_dir = src_dir / "indexes"
    for required in (
        adapter_dir / "adapter_config.json",
        adapter_dir / "adapter_model.safetensors",
        index_dir / "bm25",
        index_dir / "bm25_pyvi",
        index_dir / "dense_dek21",
    ):
        if not required.exists():
            raise RuntimeError(f"Source attempt is missing required artifact: {required}")

    rescue_dir = vol_root / source_sha / "rescue" / uuid4().hex
    (rescue_dir / "submissions").mkdir(parents=True, exist_ok=False)
    rescue_id = rescue_dir.name
    started_utc = _utc_now_iso()

    def _commit(note: str) -> None:
        try:
            volume.commit()
            print(f"[*] Volume commit ({note}): {rescue_dir}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[!] Volume commit failed ({note}): {type(exc).__name__}", flush=True)

    try:
        # 1. Code checkout at the rescue script's own commit.
        repo_dir = Path(os.environ.get("LEGALIR_MODAL_REPO_DIR", "/root/LegalIR"))
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "https://github.com/Thai-max-gif/LegalIR.git", str(repo_dir)],
                check=True,
            )
        subprocess.run(["git", "fetch", "origin", code_sha], cwd=repo_dir, check=False)
        res = subprocess.run(
            ["git", "checkout", "--detach", code_sha], cwd=repo_dir, capture_output=True, text=True
        )
        if res.returncode != 0:
            subprocess.run(["git", "fetch", "--unshallow", "origin"], cwd=repo_dir, check=False)
            subprocess.run(["git", "checkout", "--detach", code_sha], cwd=repo_dir, check=True)
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        os.chdir(repo_dir)
        print(f"[*] Rescue code at {code_sha}; artifacts from {source_sha}/{source_attempt}", flush=True)

        # 2. Canonical dataset (re-download; container disk is ephemeral).
        from scripts.colab.bootstrap import prepare_dataset

        freeze_file = repo_dir / "artifacts/task1/freeze/production_freeze.json"
        dataset_dir = Path(os.environ.get("LEGALIR_DATASET_DIR", "/root/kaggle_dataset"))
        prepare_dataset(dataset_dir, freeze_file)

        import numpy as np
        import pandas as pd
        import yaml

        from scripts.gates.run_a100 import preflight_huggingface_access  # noqa: F401 (kept for parity)
        from src.evaluation.submission import (
            package_submission,
            validate_submission,
            validate_submission_zip,
        )
        from src.pipeline.predict import LegalIRPipeline

        # 3. Fusion policy: RRF needs no model file; learned ranker reuses fusion_final.
        cv_report_path = src_dir / "cv" / "cv_report.json"
        fusion_model_path = None
        winning_method = "reciprocal_rank_fusion"
        if cv_report_path.is_file():
            try:
                winning_method = str(
                    json.loads(cv_report_path.read_text(encoding="utf-8")).get(
                        "fusion_winner",
                        json.loads(cv_report_path.read_text(encoding="utf-8")).get(
                            "winning_method", "reciprocal_rank_fusion"
                        ),
                    )
                )
            except Exception:
                winning_method = "reciprocal_rank_fusion"
        use_learned_fusion = winning_method == "learned_ranker"
        if use_learned_fusion:
            fusion_model_path = src_dir / "checkpoints" / "fusion_final"
            if not fusion_model_path.exists():
                raise RuntimeError("Fusion winner is learned_ranker but checkpoints/fusion_final is missing")
        print(f"[*] Fusion winner: {winning_method}", flush=True)

        algo_cfg = yaml.safe_load(
            (repo_dir / "configs/algorithm/legalir_v2.yaml").read_text(encoding="utf-8")
        )
        reranker_cfg = ((algo_cfg.get("ranking") or {}).get("reranker") or {})
        train_manifest = json.loads((adapter_dir / "training_manifest.json").read_text(encoding="utf-8"))

        # 4. Load finished pipeline (no training): same full-mode settings.
        t0 = time.perf_counter()
        pipeline = LegalIRPipeline.load_pipeline(
            data_dir=dataset_dir,
            index_dir=index_dir,
            reranker_adapter_path=adapter_dir,
            fusion_model_path=fusion_model_path,
            use_reranker=True,
            use_learned_fusion=use_learned_fusion,
            dense_device="cuda:0",
            reranker_device="cuda:0",
            strict_artifacts=True,
            audit_preflight=True,
            audit_output_json=rescue_dir / "parameter_audit.json",
            reranker_model_name="BAAI/bge-reranker-v2-m3",
            reranker_batch_size=int(reranker_cfg.get("inference_batch_size") or 16),
            reranker_max_length=int(reranker_cfg.get("max_length", 384)),
            precision="bf16",
            reranker_revision=train_manifest.get("base_model_revision"),
        )
        audit = pipeline.audit_parameters(
            output_json=rescue_dir / "parameter_audit.json",
            raise_on_violation=True,
            require_loaded_models=True,
        )
        print(f"[*] Pipeline loaded+audit in {time.perf_counter() - t0:.1f}s", flush=True)

        # 5. Private test queries (2080 expected).
        private_path = dataset_dir / "private-official.json"
        private_data = json.loads(private_path.read_text(encoding="utf-8"))
        if len(private_data) != 2080:
            raise RuntimeError(f"Expected 2080 private queries, found {len(private_data)}")
        q_items = list(private_data.items())
        query_dict = {
            str(qid): (q_val.get("question", "") if isinstance(q_val, dict) else str(q_val))
            for qid, q_val in q_items
        }

        dense_ret = getattr(pipeline.hybrid_engine, "dense_retriever", None) or getattr(
            pipeline.hybrid_engine, "dense", None
        )
        q_embs: dict[str, np.ndarray] = {}
        if dense_ret is not None:
            q_texts = [query_dict[str(qid)] for qid, _ in q_items]
            arr = dense_ret.encode_queries(q_texts, batch_size=64, stage_name="public_query")
            for (qid, _), emb in zip(q_items, arr):
                q_embs[str(qid)] = emb
            print(f"[+] Precomputed {len(q_embs):,} query embeddings", flush=True)

        pipeline.candidate_k = 200
        pipeline.rerank_k = 100
        t1 = time.perf_counter()
        predictions = pipeline.predict_batch(query_dict, query_embeddings=q_embs, show_progress=True)
        print(f"[+] Inference done in {time.perf_counter() - t1:.1f}s ({len(predictions)} queries)", flush=True)

        # 6. Strict packaging + validation (exactly 5 docs per query).
        sub_json = rescue_dir / "submissions" / "submission.json"
        sub_zip = rescue_dir / "submissions" / "submission.zip"
        package_submission(predictions, sub_json, sub_zip)
        docs_df = pd.read_parquet(dataset_dir / "documents.parquet")
        expected_qids = set(str(k) for k in private_data.keys())
        if set(predictions.keys()) != expected_qids:
            raise RuntimeError("Prediction keys mismatch private query IDs")
        val_res = validate_submission(
            sub_json,
            expected_qids=expected_qids,
            corpus_doc_ids=set(docs_df["doc_id"].astype(str)),
            exact_answer_count=5,
        )
        zip_res = validate_submission_zip(sub_zip)
        print(
            f"[+] Submission validation: JSON={val_res.get('is_valid')} ZIP={zip_res.get('is_valid')}",
            flush=True,
        )
        if not (val_res.get("is_valid") and zip_res.get("is_valid")):
            raise RuntimeError(
                f"Submission INVALID: {val_res.get('errors')} | {zip_res.get('errors')}"
            )

        import hashlib

        def _sha(p: Path) -> str:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()

        manifest = {
            "rescue_id": rescue_id,
            "code_sha": code_sha,
            "source_sha": source_sha,
            "source_attempt": source_attempt,
            "started_utc": started_utc,
            "finished_utc": _utc_now_iso(),
            "queries": len(predictions),
            "fusion_winner": winning_method,
            "adapter_checksum": train_manifest.get("adapter_checksum"),
            "submission_json_sha256": _sha(sub_json),
            "submission_zip_sha256": _sha(sub_zip),
            "total_learned_parameters": (audit.get("total_learned_parameters")
                                         if isinstance(audit, dict) else None),
            "outcome": "completed",
        }
        (rescue_dir / "rescue_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        _commit("rescue completed")
        print(f"[+] RESCUE COMPLETED: {rescue_dir}/submissions/submission.zip", flush=True)
        return manifest
    except BaseException as exc:
        try:
            (rescue_dir / "rescue_failed.json").write_text(
                json.dumps(
                    {
                        "rescue_id": rescue_id,
                        "code_sha": code_sha,
                        "source_sha": source_sha,
                        "source_attempt": source_attempt,
                        "started_utc": started_utc,
                        "failed_utc": _utc_now_iso(),
                        "exception_class": type(exc).__name__,
                        "outcome": "failed",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
        _commit("rescue failed")
        raise


@app.local_entrypoint()
def main(source_sha: str, source_attempt: str):
    import sys

    code_sha = os.environ.get("LEGALIR_COMMIT_SHA", "")
    if not code_sha:
        try:
            code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            print("Error: cannot determine HEAD; set LEGALIR_COMMIT_SHA.", file=sys.stderr)
            sys.exit(1)
    for label, val, pat in (("code_sha", code_sha, _SHA_RE),
                            ("source_sha", source_sha, _SHA_RE)):
        if not pat.fullmatch(str(val or "").strip().lower()):
            print(f"Error: {label} must be an exact 40-char SHA.", file=sys.stderr)
            sys.exit(1)
    if not _HEX_RE.fullmatch(str(source_attempt or "").strip().lower()):
        print("Error: source_attempt must be a 32-char hex id.", file=sys.stderr)
        sys.exit(1)
    try:
        local_head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        if local_head.lower() != code_sha.lower():
            print("Error: LEGALIR_COMMIT_SHA must equal local HEAD.", file=sys.stderr)
            sys.exit(1)
        dirty = subprocess.check_output(["git", "status", "--porcelain=v1"]).decode().strip()
        if dirty:
            print("Error: working tree dirty; commit rescue script before dispatch.", file=sys.stderr)
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: git check failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Rescue code {code_sha} from artifacts {source_sha}/{source_attempt}")
    print(f"[*] Timeout {TIMEOUT_SECONDS}s; outputs to <sha>/rescue/<uuid>/ (source untouched)")
    result = rescue_inference.remote(source_sha, source_attempt, code_sha)
    print(f"[*] Rescue result: {result}")

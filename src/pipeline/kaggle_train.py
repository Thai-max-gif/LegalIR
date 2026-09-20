"""Production Kaggle T4 x2 End-to-End Orchestrator for LegalIR Task 1.

Executes the complete 24-step pipeline:
1. Multi-GPU environment detection and device allocation (GPU 0: Dense, GPU 1: Reranker).
2. Canonical dataset discovery, auto-build from raw archives if needed, and strict validation.
3. Preflight parameter budget audit against the strict <4B rule.
4. Dual BM25 indexing (raw legal + PyVi word-segmented) with document metadata enrichment.
5. DEk21 Dense Macro indexing on GPU 0.
6. Precomputing and caching train query dense embeddings on GPU 0.
7. Fold-isolated hard-negative training pair mining.
8. 5-Fold Out-of-Fold (OOF) cross-validation with fold-specific LoRA rerankers.
9. Full cross-fitted Learned Fusion (LightGBM) vs Tuned RRF evaluation.
10. Document-disjoint robustness split validation.
11. Full 7,000-query question memory indexing with precomputed embeddings.
12. Final LoRA reranker fine-tuning on all 7,000 training queries on GPU 1.
13. Final fusion model fitting on all OOF candidate features (if learned fusion won).
14. Public test query discovery and batch inference with the fully loaded production pipeline.
15. Strict submission invariant validation (bounds, types, no duplicates, valid corpus IDs).
16. Zip packaging containing strictly submission.json at root.
17. Full artifact manifest creation with SHA-256 hashes and parameter audits.
18. Support for run_mode="smoke" and run_mode="full".
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Literal
import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset.build_canonical import build_canonical_package
from src.dataset.validator import validate_canonical_dataset
from src.evaluation.submission import (
    compute_sha256,
    create_submission_manifest,
    package_submission,
    validate_submission,
    validate_submission_zip,
)
from src.models.parameter_audit import (
    MAX_PARAMETER_BUDGET,
    ParameterBudgetExceededError,
    audit_system_parameters,
)
from src.pipeline.oof_runner import OOFRunner
from src.pipeline.predict import LegalIRPipeline
from src.ranking.train_fusion import train_and_evaluate_fusion_cv
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.build_indexes import (
    build_bm25_index,
    build_bm25_pyvi_index,
    enrich_chunks_with_doc_metadata,
)
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.question_memory import TrainQuestionMemory
from src.training.build_pairs import build_training_pairs, project_pair_mining_seconds
from src.training.train_reranker import train_reranker


@dataclass
class StageTimingEntry:
    """Telemetry entry for a pipeline execution stage."""
    seconds: float
    cache_hit: bool = False
    start_time: float | None = None
    end_time: float | None = None
    workload_count: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StageTimingTelemetry:
    """Collects structured performance and cache-hit telemetry across all pipeline stages."""

    def __init__(self) -> None:
        self._stages: dict[str, StageTimingEntry] = {}

    def record(
        self,
        stage_name: str,
        elapsed_seconds: float,
        cache_hit: bool = False,
        start_time: float | None = None,
        end_time: float | None = None,
        workload_count: int | None = None,
        error: str | None = None,
        **metadata: Any,
    ) -> None:
        self._stages[stage_name] = StageTimingEntry(
            seconds=round(float(elapsed_seconds), 4),
            cache_hit=bool(cache_hit),
            start_time=round(float(start_time), 4) if start_time is not None else None,
            end_time=round(float(end_time), 4) if end_time is not None else None,
            workload_count=int(workload_count) if workload_count is not None else None,
            error=str(error) if error is not None else None,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "seconds": entry.seconds,
                "cache_hit": entry.cache_hit,
                **({"start_time": entry.start_time} if entry.start_time is not None else {}),
                **({"end_time": entry.end_time} if entry.end_time is not None else {}),
                **({"workload_count": entry.workload_count} if entry.workload_count is not None else {}),
                **({"error": entry.error} if entry.error is not None else {}),
                **entry.metadata,
            }
            for name, entry in self._stages.items()
        }


NOMINAL_BUDGET_SECONDS = 270 * 60  # 270-minute nominal work allocation
STRICT_GATE_SECONDS = int(os.environ.get("LEGALIR_TIME_GATE_SECONDS", 25200))  # 7-hour strict execution gate


def forecast_cold_total(
    setup_index_seconds: float = 0.0,
    static_retrieval_seconds: float = 0.0,
    assembly_seconds: float = 0.0,
    train_jobs: list[dict[str, Any]] | None = None,
    eval_queries: int = 0,
    eval_qps: float | None = None,
    fusion_seconds: float = 0.0,
    final_reload_public_seconds: float = 0.0,
    delivery_seconds: float = 0.0,
) -> dict[str, Any]:
    """Conservative cold-run forecast from measured stage bounds (fix.md section 11).

    ``T_total = setup + static + assembly + sum(updates*sec_per_update + overhead)
    + Q_eval/qps + fusion + reload/public + delivery``. Stage clocks must be
    exclusive; workload counts (not nominal configs) drive training/evaluation.
    Unknown throughput (``eval_qps`` missing/zero) fails closed to infinity —
    never a passing forecast. Headline speedups must be re-measured per stage
    after integration, never multiplied across the run.
    """
    jobs = list(train_jobs or [])
    training_seconds = 0.0
    total_updates = 0
    for job in jobs:
        updates = int(job.get("updates", 0))
        spu = float(job.get("sec_per_update", 0.0))
        overhead = float(job.get("load_save_overhead", 0.0))
        if updates < 0 or spu < 0 or overhead < 0:
            raise ValueError(f"negative training forecast component: {job}")
        total_updates += updates
        training_seconds += updates * spu + overhead
    if eval_queries and (not eval_qps or float(eval_qps) <= 0):
        evaluation_seconds = float("inf")
    else:
        evaluation_seconds = (int(eval_queries) / float(eval_qps)) if eval_queries else 0.0
    total = (
        float(setup_index_seconds)
        + float(static_retrieval_seconds)
        + float(assembly_seconds)
        + training_seconds
        + evaluation_seconds
        + float(fusion_seconds)
        + float(final_reload_public_seconds)
        + float(delivery_seconds)
    )
    # A forecast with no measured workload is unknown, never fitting: calling
    # with no training jobs and no evaluation must not report 0 seconds as
    # evidence of reduced runtime.
    complete = bool(jobs) and total_updates > 0 and (
        not eval_queries or (eval_qps is not None and float(eval_qps) > 0)
    )
    return {
        "total_seconds": total,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_updates": total_updates,
        "complete_measurements": complete,
        "fits_nominal_270m": bool(complete) and total <= NOMINAL_BUDGET_SECONDS,
        "fits_strict_300m": bool(complete) and total < STRICT_GATE_SECONDS,
    }


def build_attempt_acceptance_receipt(
    *,
    working_path: str | Path,
    cv_report: dict[str, Any],
    fusion_report: dict[str, Any],
    doc_disjoint_report: dict[str, Any],
    predictions: dict[str, Any],
    qrels_dict: dict[str, list[str]],
    splits: Any,
    corpus_doc_ids: set[str],
    final_reranker_dir: str | Path,
    final_reranker_report: dict[str, Any],
    reranker_model: str,
    reranker_revision: str | None,
    git_sha: str,
    backend: str,
    supervisor_start_utc: str,
    supervisor_end_utc: str,
    elapsed_seconds: float,
    submission_valid: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the machine-readable acceptance receipt for one FULL attempt and
    verify it against the attempt's own persisted artifacts (fix.md section 12).

    Uses only this attempt's outputs: per-query OOF predictions, fixed splits,
    qrels, corpus IDs, disjoint report, submission, and adapter files. The
    receipt is persisted by the caller; shutdown stays unconfirmed here because
    only the operator can confirm provider termination (re-verify afterwards
    with scripts/verify_end_to_end_acceptance.py).
    """
    import hashlib as _hl

    from src.release.acceptance import not_run_receipt, verify_acceptance, verify_adapter_reload_fresh

    working_path = Path(working_path)
    receipt = not_run_receipt(attempt_id=working_path.name)
    try:
        from src.models.bootstrap import MODEL_REGISTRY as _REG

        dense_rev = (_REG.get("CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2") or {}).get("revision")
    except Exception:
        dense_rev = None
    receipt.update({
        "runtime_sha": git_sha,
        "release_sha": git_sha,
        "backend": backend,
        "protocol": "confirmation",
        "selection_protocol": (fusion_report.get("comparison") or {}).get(
            "selection_protocol", fusion_report.get("selection_protocol", "predeclared_rrf")),
        "supervisor_start_utc": supervisor_start_utc,
        "supervisor_end_utc": supervisor_end_utc,
        "elapsed_seconds": float(elapsed_seconds),
        "model_revisions": {"reranker": reranker_revision, "dense": dense_rev},
    })
    receipt["dataset_digest"] = _hl.sha256(
        json.dumps({
            "queries": sorted(qrels_dict.keys()),
            "qrels": sum(len(v) for v in qrels_dict.values()),
            "corpus": len(corpus_doc_ids),
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()
    receipt["split_random_5fold_sha256"] = (cv_report.get("split_provenance") or {}).get("random_5fold", {}).get("sha256") \
        if isinstance(cv_report.get("split_provenance"), dict) else None
    if not receipt["split_random_5fold_sha256"]:
        receipt["split_random_5fold_sha256"] = cv_report.get("resolved_config_sha256")
    receipt["config_sha256"] = cv_report.get("resolved_config_sha256")
    comparison = fusion_report.get("comparison") or {}
    receipt["predeclared_policy_hash"] = _hl.sha256(json.dumps({
        "winning_method": fusion_report.get("winning_method"),
        "feature_columns": (fusion_report.get("manifest") or {}).get("feature_columns"),
        "selection_protocol": comparison.get("selection_protocol"),
    }, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    receipt["split_exposure_disclosure"] = (
        "Frozen seed-42 splits; fusion predeclared (no outer-label selection). "
        "Development history may include prior runs on these splits; see repo docs."
    )
    receipt["expected_oof_queries"] = int(cv_report.get("total_evaluated_queries", 0))
    receipt["per_fold"] = [
        {"fold": int(f.get("fold", i)),
         "count": int(f.get("val_queries", 0)),
         "recall@5": float(f.get("recall@5", 0.0))}
        for i, f in enumerate(cv_report.get("folds", []))
    ]
    overall = cv_report.get("overall_aggregate_metrics") or {}
    receipt["pooled_oof_recall@5"] = overall.get("recall@5")
    receipt["pooled_oof_precision@5"] = overall.get("precision@5")
    dj_sys = (doc_disjoint_report or {}).get("trained_reranker_system") or {}
    receipt["doc_disjoint"] = {
        "recall@5": dj_sys.get("recall@5", (doc_disjoint_report or {}).get("recall@5")),
        "count": int(dj_sys.get("val_queries", 0)),
        "complete": bool(doc_disjoint_report),
    }
    receipt["training_jobs"] = [
        {"job": f"fold_{f.get('fold', i)}", "updates": int(f.get("reranker_optimizer_steps", 0))}
        for i, f in enumerate(cv_report.get("folds", []))
    ] + [{"job": "final", "updates": int(
        final_reranker_report.get("optimizer_steps", final_reranker_report.get("global_steps", 0)) or 0)}]
    reload_ok, reload_detail = verify_adapter_reload_fresh(
        final_reranker_dir, reranker_model, reranker_revision)
    receipt["reload_ok"] = bool(reload_ok)
    receipt["reload_detail"] = reload_detail
    receipt["submission_path"] = "submissions/submission.json"
    # OOF predictions for scoring come from the persisted per-query file, never
    # from the public submission (different query population). Bundle a copy of
    # every verifier input inside the attempt so the receipt stays
    # re-verifiable after delivery (scripts/confirm_attempt_shutdown.py).
    oof_pred_path = working_path / "cv" / "oof_predictions.json"
    try:
        oof_predictions = json.loads(oof_pred_path.read_text(encoding="utf-8")) if oof_pred_path.is_file() else {}
    except Exception:
        oof_predictions = {}
    if not isinstance(oof_predictions, dict):
        oof_predictions = {}
    inputs_dir = working_path / "acceptance_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "oof_predictions.json": oof_predictions,
        "qrels.json": qrels_dict,
        "splits.json": splits,
        "corpus_ids.json": sorted(str(x) for x in corpus_doc_ids),
        "doc_disjoint_report.json": doc_disjoint_report or {},
        "submission.json": predictions,
    }
    verify_inputs: dict[str, str] = {}
    for name, payload_obj in bundle.items():
        try:
            (inputs_dir / name).write_text(json.dumps(payload_obj), encoding="utf-8")
            verify_inputs[name] = f"acceptance_inputs/{name}"
        except Exception:
            pass
    receipt["verify_inputs"] = verify_inputs
    artifacts: list[dict[str, Any]] = []
    for rel in (
        "checkpoints/reranker_final/adapter_model.safetensors",
        "checkpoints/reranker_final/adapter_model.bin",
        "checkpoints/reranker_final/adapter_config.json",
        "checkpoints/reranker_final/training_manifest.json",
        "checkpoints/reranker_final/complete.json",
        "submissions/submission.json",
        "submissions/submission.zip",
        "submissions/submission_manifest.json",
        "cv/oof_predictions.json",
        "acceptance_inputs/oof_predictions.json",
        "acceptance_inputs/qrels.json",
        "acceptance_inputs/splits.json",
        "acceptance_inputs/corpus_ids.json",
        "acceptance_inputs/doc_disjoint_report.json",
        "acceptance_inputs/submission.json",
    ):
        target = working_path / rel
        if target.is_file():
            artifacts.append({"path": rel, "sha256": _hl.sha256(target.read_bytes()).hexdigest()})
    receipt["artifacts"] = artifacts
    receipt["delivery_confirmed"] = bool(submission_valid)
    receipt["delivery_note"] = (
        "Delivered to attempt directory (durable Volume path on Modal). "
        "Provider shutdown and HF publication require separate operator confirmation."
    )
    receipt["shutdown_confirmed"] = False
    verification = verify_acceptance(
        receipt,
        oof_predictions=oof_predictions,
        qrels=qrels_dict,
        splits=splits,
        corpus_doc_ids=set(corpus_doc_ids),
        disjoint_report=doc_disjoint_report or None,
        submission=predictions,
        artifacts_dir=working_path,
    )
    receipt["verdict"] = verification["verdict"]
    receipt["reasons"] = verification["reasons"]
    return receipt, verification


@dataclass
class KaggleRunResult:
    """Structured result of an end-to-end LegalIR Kaggle pipeline run."""

    is_valid: bool
    submission_path: Path
    submission_zip_path: Path
    manifest_path: Path
    audit_report: dict[str, Any]
    cv_report: dict[str, Any]
    fusion_report: dict[str, Any]
    public_predictions_count: int
    run_mode: str
    execution_time_seconds: float
    artifacts_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "COMPLETED"


def get_process_rss_mb() -> float:
    """Return process resident set size in megabytes."""
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0), 2)
    except Exception:
        return 0.0


def get_peak_process_rss_mb() -> float:
    """Return true historical peak process resident set size in megabytes."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(usage / (1024.0 * 1024.0), 2)
        return round(usage / 1024.0, 2)
    except Exception:
        return 0.0


def resource_inventory() -> dict[str, Any]:
    """Local measurement-protocol inventory: host/GPU resources for forecasts.

    CPU-only and offline-safe. Records logical CPU count, total host RAM, and
    visible GPU names/count so A100 qualification can select worker counts and
    memory budgets from measurements, not arbitrary reservations.
    """
    import platform

    inv: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_logical": os.cpu_count(),
    }
    try:
        import psutil

        inv["host_ram_total_mb"] = round(psutil.virtual_memory().total / (1024.0 * 1024.0), 2)
        inv["host_ram_available_mb"] = round(psutil.virtual_memory().available / (1024.0 * 1024.0), 2)
    except Exception:
        inv["host_ram_total_mb"] = None
        inv["host_ram_available_mb"] = None
    try:
        if torch.cuda.is_available():
            inv["gpu_count"] = int(torch.cuda.device_count())
            inv["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            inv["gpu_count"] = 0
            inv["gpu_names"] = []
    except Exception:
        inv["gpu_count"] = None
        inv["gpu_names"] = []
    return inv


def resolve_repo_path(value: str | Path | None, repo_root: str | Path) -> Path:
    """Resolve a path relative to the repo root if it is not already absolute."""
    if value is None:
        return Path(repo_root)
    p = Path(value)
    if not p.is_absolute():
        p = Path(repo_root) / p
    return p.resolve()


def query_ids_sha(query_ids: list[str]) -> str:
    import hashlib

    return hashlib.sha256(",".join(str(x) for x in query_ids).encode("utf-8")).hexdigest()


def query_text_content_sha(query_ids: list[str], query_texts: list[str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for qid, txt in zip(query_ids, query_texts):
        h.update(str(qid).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(txt).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def write_train_query_embedding_cache(
    index_dir: str | Path,
    query_ids: list[str],
    query_texts: list[str],
    embeddings: Any,
    encoder_model: str,
    encoder_revision: str | None,
) -> None:
    index_dir = Path(index_dir)
    np.save(str(index_dir / "train_query_embeddings.npy"), np.asarray(embeddings))
    emb_arr = np.asarray(embeddings)
    (index_dir / "train_query_embeddings.meta.json").write_text(
        json.dumps(
            {
                "query_ids": [str(x) for x in query_ids],
                "count": len(query_ids),
                "qids_sha256": query_ids_sha([str(x) for x in query_ids]),
                "qid_text_sha256": query_text_content_sha(
                    [str(x) for x in query_ids], [str(x) for x in query_texts]
                ),
                "encoder_model": str(encoder_model),
                "encoder_revision": encoder_revision,
                "embedding_dim": int(emb_arr.shape[1]) if emb_arr.ndim > 1 else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_validated_train_query_embeddings(
    index_dir: str | Path,
    query_ids: list[str],
    query_texts: list[str],
    encoder_model: str,
    encoder_revision: str | None,
    expected_dim: int | None = None,
) -> Any | None:
    """Return cached embeddings only when id order, text content, encoder match.

    Returns ``None`` when the cache is missing, obsolete (no sidecar), or
    mismatched. Callers must recompute on ``None``. When the id set matches
    but order differs, rows are remapped by id.
    """
    index_dir = Path(index_dir)
    npy_path = index_dir / "train_query_embeddings.npy"
    meta_path = index_dir / "train_query_embeddings.meta.json"
    if not npy_path.is_file():
        return None
    if not meta_path.is_file():
        raise ValueError("embedding sidecar meta.json missing (obsolete cache)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("count", -1)) != len(query_ids):
        raise ValueError(
            f"count mismatch (meta={meta.get('count')} expected={len(query_ids)})"
        )
    if str(meta.get("qids_sha256") or "") != query_ids_sha([str(x) for x in query_ids]):
        # Allow reorder-with-same-set via remap below; otherwise fail.
        meta_ids = [str(x) for x in (meta.get("query_ids") or [])]
        if set(meta_ids) != set(str(x) for x in query_ids):
            raise ValueError("query-id order/content mismatch")
    if str(meta.get("qid_text_sha256") or "") != query_text_content_sha(
        [str(x) for x in query_ids], [str(x) for x in query_texts]
    ):
        # Text change with same ids still invalidates; but if order differs,
        # the text hash above (current order) cannot match meta order. Fall
        # back to set-based text check via meta ids when possible.
        meta_ids = [str(x) for x in (meta.get("query_ids") or [])]
        if not meta_ids or set(meta_ids) != set(str(x) for x in query_ids):
            raise ValueError("query-text content mismatch")
        # Recompute expected hash in meta order for a fair comparison.
        by_text = {qid: txt for qid, txt in zip(query_ids, query_texts)}
        try:
            meta_order_texts = [by_text[qid] for qid in meta_ids]
        except KeyError as exc:
            raise ValueError("query-text content mismatch") from exc
        if str(meta.get("qid_text_sha256") or "") != query_text_content_sha(meta_ids, meta_order_texts):
            raise ValueError("query-text content mismatch")
    if str(meta.get("encoder_model") or "") != str(encoder_model):
        raise ValueError(
            f"encoder model mismatch (meta={meta.get('encoder_model')} expected={encoder_model})"
        )
    if (meta.get("encoder_revision") or None) != (encoder_revision or None):
        raise ValueError(
            f"encoder revision mismatch (meta={meta.get('encoder_revision')} expected={encoder_revision})"
        )
    loaded = np.load(str(npy_path))
    if loaded.ndim != 2:
        raise ValueError(f"query embeddings must be 2-D, got ndim={loaded.ndim}.")
    if loaded.shape[0] != len(query_ids):
        raise ValueError(f"row-count mismatch (npy={loaded.shape[0]} expected={len(query_ids)})")
    meta_dim = meta.get("embedding_dim")
    if meta_dim is not None and int(meta_dim) != int(loaded.shape[1]):
        raise ValueError(f"query embedding dim mismatch (meta={meta_dim} npy={loaded.shape[1]}).")
    if expected_dim is not None and int(loaded.shape[1]) != int(expected_dim):
        raise ValueError(f"query embedding dim mismatch (npy={loaded.shape[1]} expected={expected_dim}).")
    try:
        if not bool(np.isfinite(np.asarray(loaded)).all()):
            raise ValueError("query embeddings contain non-finite values.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"query embedding integrity check failed: {exc}") from exc
    meta_ids = [str(x) for x in (meta.get("query_ids") or [])]
    if meta_ids and meta_ids != [str(x) for x in query_ids]:
        by_row = {qid: loaded[i] for i, qid in enumerate(meta_ids)}
        loaded = np.stack([by_row[qid] for qid in (str(x) for x in query_ids)])
    return loaded


def resolve_kaggle_devices(devices: list[str] | None = None) -> tuple[str, str]:
    """
    Allocate multi-GPU devices intentionally:
    GPU 0: Dense embedding / index / query encoding
    GPU 1: Reranker training + reranker inference
    """
    if devices and len(devices) >= 2:
        return str(devices[0]), str(devices[1])
    elif devices and len(devices) == 1:
        return str(devices[0]), str(devices[0])

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_count >= 2:
        return "cuda:0", "cuda:1"
    elif device_count == 1:
        return "cuda:0", "cuda:0"
    else:
        return "cpu", "cpu"


def resolve_pipeline_device_allocation(
    device_contract: Any | None = None,
    run_mode: str = "full",
    devices: list[str] | None = None,
    dense_device: str | None = None,
    reranker_device: str | None = None,
    allow_nonstandard_production_devices: bool = False,
) -> tuple[str, str]:
    """
    Resolve and validate device allocation based on a DeviceContract or defaults.
    If device_contract is provided, enforces its constraints (min_cuda_devices, placement, accelerator_contains).
    """
    from src.models.device import resolve_device

    is_smoke = str(run_mode).lower() == "smoke"
    is_gpu_smoke = str(run_mode).lower() == "gpu_smoke"
    is_full = str(run_mode).lower() == "full"

    if is_smoke:
        return "cpu", "cpu"

    if device_contract is not None:
        from src.release.contracts import verify_device_contract
        verify_device_contract(device_contract, allow_debug=allow_nonstandard_production_devices)
        d_dev = dense_device or device_contract.dense_device
        r_dev = reranker_device or device_contract.reranker_device
        return resolve_device(d_dev), resolve_device(r_dev)

    if (is_gpu_smoke or is_full) and not allow_nonstandard_production_devices:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{run_mode} mode requires CUDA but torch.cuda.is_available() is False")
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                f"{run_mode} mode requires Kaggle T4 x2 / >=2 CUDA devices (found {torch.cuda.device_count()} device(s))"
            )

    if dense_device is None or reranker_device is None:
        d_dev_def, r_dev_def = resolve_kaggle_devices(devices)
        dense_device = dense_device or d_dev_def
        reranker_device = reranker_device or r_dev_def

    dense_device = resolve_device(dense_device)
    reranker_device = resolve_device(reranker_device)

    if (is_gpu_smoke or is_full) and not allow_nonstandard_production_devices:
        if dense_device != "cuda:0":
            raise RuntimeError(f"{run_mode} mode requires dense_device == 'cuda:0', got '{dense_device}'")
        if reranker_device != "cuda:1":
            raise RuntimeError(f"{run_mode} mode requires reranker_device == 'cuda:1', got '{reranker_device}'")

    return dense_device, reranker_device


def get_git_commit(repo_root: Path | None = None) -> str:
    """Retrieve Git commit SHA safely."""
    try:
        cwd = repo_root or Path.cwd()
        commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode("utf-8")
            .strip()
        )
        return commit_sha
    except Exception:
        return "unknown"


OFFICIAL_DOCUMENT_COUNT = 8532
OFFICIAL_TOTAL_CHUNKS = 1153876
OFFICIAL_MICRO_CHUNKS = 934416
OFFICIAL_MACRO_CHUNKS = 219460
OFFICIAL_TRAIN_QUERIES = 7000
OFFICIAL_QRELS = 7637
OFFICIAL_PUBLIC_QUERY_COUNT = 1000
OFFICIAL_PRIVATE_QUERY_COUNT = 2080
OFFICIAL_TEST_QUERY_COUNTS = {OFFICIAL_PUBLIC_QUERY_COUNT, OFFICIAL_PRIVATE_QUERY_COUNT}
OFFICIAL_DATASET_NAME = "task1_canonical"
OFFICIAL_VERSION = "v2"
OFFICIAL_SCHEMA = "hierarchical_micro_macro_v2"


def validate_official_task1_identity(
    *,
    data_dir: str | Path,
    public_json_path: str | Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate official Task 1 v2 dataset identity against canonical manifest, audit report, and public queries.

    Verifies:
    - manifest.dataset == "task1_canonical"
    - manifest.version == "v2"
    - manifest.schema == "hierarchical_micro_macro_v2"
    - exact official production counts:
        - 8,532 documents
        - 1,153,876 total chunks (934,416 micro, 219,460 macro)
        - 7,000 train queries
        - 7,637 qrels
    - audit_report.is_valid is True and audit_report.errors is empty
    - public query count == 1,000 when public file is provided/present
    """
    d_path = Path(data_dir)
    errors: list[str] = []

    manifest_file = d_path / "manifest.json"
    audit_file = d_path / "audit_report.json"

    if not manifest_file.exists():
        errors.append(f"Missing manifest.json in {d_path}")
    if not audit_file.exists():
        errors.append(f"Missing audit_report.json in {d_path}")

    manifest_data: dict[str, Any] = {}
    audit_data: dict[str, Any] = {}

    if manifest_file.exists():
        try:
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"Failed to parse manifest.json: {e}")

    if audit_file.exists():
        try:
            audit_data = json.loads(audit_file.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"Failed to parse audit_report.json: {e}")

    dataset_name = manifest_data.get("dataset")
    version = manifest_data.get("version")
    schema = manifest_data.get("schema")

    is_mock = (version == "mock") or (manifest_data.get("mock") is True)

    # Public queries validation
    public_count = 0
    if public_json_path is not None:
        pub_p = Path(public_json_path)
        if pub_p.exists():
            try:
                pub_data = json.loads(pub_p.read_text(encoding="utf-8"))
                if isinstance(pub_data, dict):
                    public_count = len(pub_data)
                elif isinstance(pub_data, list):
                    public_count = len(pub_data)
            except Exception as e:
                errors.append(f"Failed to parse public test json {pub_p}: {e}")
        else:
            errors.append(f"Provided public_json_path does not exist: {pub_p}")

        if public_count not in OFFICIAL_TEST_QUERY_COUNTS:
            errors.append(
                f"Official test queries count mismatch: expected one of {sorted(OFFICIAL_TEST_QUERY_COUNTS)}, got {public_count}"
            )

    if is_mock:
        is_valid = len(errors) == 0
        report = {
            "is_valid": is_valid,
            "dataset": dataset_name or OFFICIAL_DATASET_NAME,
            "version": version or "mock",
            "schema": schema or OFFICIAL_SCHEMA,
            "documents": OFFICIAL_DOCUMENT_COUNT,
            "chunks": OFFICIAL_TOTAL_CHUNKS,
            "micro_chunks": OFFICIAL_MICRO_CHUNKS,
            "macro_chunks": OFFICIAL_MACRO_CHUNKS,
            "train_queries": OFFICIAL_TRAIN_QUERIES,
            "qrels": OFFICIAL_QRELS,
            "public_queries": public_count or OFFICIAL_PUBLIC_QUERY_COUNT,
            "audit_valid": True,
            "audit_errors": [],
            "errors": errors,
        }
        if strict and not is_valid:
            raise ValueError(f"Official Task 1 dataset identity validation failed: {'; '.join(errors)}")
        return report

    if dataset_name != OFFICIAL_DATASET_NAME:
        errors.append(f"Manifest dataset mismatch: expected '{OFFICIAL_DATASET_NAME}', got '{dataset_name}'")
    if version != OFFICIAL_VERSION:
        errors.append(f"Manifest version mismatch: expected '{OFFICIAL_VERSION}', got '{version}'")
    if schema != OFFICIAL_SCHEMA:
        errors.append(f"Manifest schema mismatch: expected '{OFFICIAL_SCHEMA}', got '{schema}'")

    def _extract_count(d: dict[str, Any], *keys: str) -> int:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return int(d[k])
                except (ValueError, TypeError):
                    pass
        if "counts" in d and isinstance(d["counts"], dict):
            for k in keys:
                if k in d["counts"] and d["counts"][k] is not None:
                    try:
                        return int(d["counts"][k])
                    except (ValueError, TypeError):
                        pass
        return 0

    m_docs = _extract_count(manifest_data, "total_documents", "documents")
    m_chunks = _extract_count(manifest_data, "total_chunks", "chunks")
    m_micro = _extract_count(manifest_data, "total_micro_chunks", "micro_chunks", "micro")
    m_macro = _extract_count(manifest_data, "total_macro_chunks", "macro_chunks", "macro")
    m_queries = _extract_count(manifest_data, "total_queries", "train_queries", "queries")
    m_qrels = _extract_count(manifest_data, "total_qrels", "qrels")

    if m_docs != OFFICIAL_DOCUMENT_COUNT:
        errors.append(f"Manifest documents count mismatch: expected {OFFICIAL_DOCUMENT_COUNT}, got {m_docs}")
    if m_chunks != OFFICIAL_TOTAL_CHUNKS:
        errors.append(f"Manifest chunks count mismatch: expected {OFFICIAL_TOTAL_CHUNKS}, got {m_chunks}")
    if m_micro != OFFICIAL_MICRO_CHUNKS:
        errors.append(f"Manifest micro chunks count mismatch: expected {OFFICIAL_MICRO_CHUNKS}, got {m_micro}")
    if m_macro != OFFICIAL_MACRO_CHUNKS:
        errors.append(f"Manifest macro chunks count mismatch: expected {OFFICIAL_MACRO_CHUNKS}, got {m_macro}")
    if m_queries != OFFICIAL_TRAIN_QUERIES:
        errors.append(f"Manifest train queries count mismatch: expected {OFFICIAL_TRAIN_QUERIES}, got {m_queries}")
    if m_qrels != OFFICIAL_QRELS:
        errors.append(f"Manifest qrels count mismatch: expected {OFFICIAL_QRELS}, got {m_qrels}")

    # Audit report validation
    audit_is_valid = bool(audit_data.get("is_valid", False))
    audit_errors = audit_data.get("errors", [])
    if not audit_is_valid:
        errors.append(f"Audit report is_valid is False: {audit_errors}")
    if audit_errors:
        errors.append(f"Audit report contains errors: {audit_errors}")

    a_docs = _extract_count(audit_data, "total_documents", "documents")
    a_chunks = _extract_count(audit_data, "total_chunks", "chunks")
    a_micro = _extract_count(audit_data, "total_micro_chunks", "micro_chunks", "micro")
    a_macro = _extract_count(audit_data, "total_macro_chunks", "macro_chunks", "macro")
    a_queries = _extract_count(audit_data, "total_queries", "train_queries", "queries")
    a_qrels = _extract_count(audit_data, "total_qrels", "qrels")

    if a_docs != OFFICIAL_DOCUMENT_COUNT:
        errors.append(f"Audit report documents count mismatch: expected {OFFICIAL_DOCUMENT_COUNT}, got {a_docs}")
    if a_chunks != OFFICIAL_TOTAL_CHUNKS:
        errors.append(f"Audit report chunks count mismatch: expected {OFFICIAL_TOTAL_CHUNKS}, got {a_chunks}")
    if a_micro != OFFICIAL_MICRO_CHUNKS:
        errors.append(f"Audit report micro chunks count mismatch: expected {OFFICIAL_MICRO_CHUNKS}, got {a_micro}")
    if a_macro != OFFICIAL_MACRO_CHUNKS:
        errors.append(f"Audit report macro chunks count mismatch: expected {OFFICIAL_MACRO_CHUNKS}, got {a_macro}")
    if a_queries != OFFICIAL_TRAIN_QUERIES:
        errors.append(f"Audit report train queries count mismatch: expected {OFFICIAL_TRAIN_QUERIES}, got {a_queries}")
    if a_qrels != OFFICIAL_QRELS:
        errors.append(f"Audit report qrels count mismatch: expected {OFFICIAL_QRELS}, got {a_qrels}")

    # Public queries validation
    public_count = 0
    if public_json_path is not None:
        pub_p = Path(public_json_path)
        if pub_p.exists():
            try:
                pub_data = json.loads(pub_p.read_text(encoding="utf-8"))
                if isinstance(pub_data, dict):
                    public_count = len(pub_data)
                elif isinstance(pub_data, list):
                    public_count = len(pub_data)
            except Exception as e:
                errors.append(f"Failed to parse public test json {pub_p}: {e}")
        else:
            errors.append(f"Provided public_json_path does not exist: {pub_p}")

        if public_count not in OFFICIAL_TEST_QUERY_COUNTS:
            errors.append(
                f"Official test queries count mismatch: expected one of {sorted(OFFICIAL_TEST_QUERY_COUNTS)}, got {public_count}"
            )

    is_valid = len(errors) == 0

    report = {
        "is_valid": is_valid,
        "dataset": dataset_name or OFFICIAL_DATASET_NAME,
        "version": version or OFFICIAL_VERSION,
        "schema": schema or OFFICIAL_SCHEMA,
        "documents": m_docs or a_docs or OFFICIAL_DOCUMENT_COUNT,
        "chunks": m_chunks or a_chunks or OFFICIAL_TOTAL_CHUNKS,
        "micro_chunks": m_micro or a_micro or OFFICIAL_MICRO_CHUNKS,
        "macro_chunks": m_macro or a_macro or OFFICIAL_MACRO_CHUNKS,
        "train_queries": m_queries or a_queries or OFFICIAL_TRAIN_QUERIES,
        "qrels": m_qrels or a_qrels or OFFICIAL_QRELS,
        "public_queries": public_count or OFFICIAL_PUBLIC_QUERY_COUNT,
        "audit_valid": audit_is_valid,
        "audit_errors": audit_errors,
        "errors": errors,
    }

    if strict and not is_valid:
        raise ValueError(f"Official Task 1 dataset identity validation failed: {'; '.join(errors)}")

    return report


@dataclass(frozen=True)
class SplitArtifactResolution:
    random_5fold_path: Path
    doc_disjoint_path: Path
    random_source: str
    doc_disjoint_source: str
    random_sha256: str = ""
    doc_disjoint_sha256: str = ""


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    import hashlib
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_split_artifacts(
    canonical_data_dir: str | Path,
    working_dir: str | Path,
    repo_root: str | Path | None = None,
) -> SplitArtifactResolution:
    """Deterministically resolve split artifacts across input, repo, and generated locations.

    Resolution order:
    1. canonical_data_dir/splits/<file>.json
    2. repo_root/artifacts/task1/data/splits/<file>.json
    3. generate deterministic seed=42 files in working_dir/splits/
    """
    canonical_data_dir = Path(canonical_data_dir)
    working_dir = Path(working_dir)
    repo = Path(repo_root) if repo_root else Path.cwd()

    cand_input_random = canonical_data_dir / "splits/random_5fold.json"
    cand_input_disjoint = canonical_data_dir / "splits/doc_disjoint_split.json"

    cand_repo_random = (
        (repo / "kaggle_dataset/splits/random_5fold.json")
        if (repo / "kaggle_dataset/splits/random_5fold.json").exists()
        else repo / "artifacts/task1/data/splits/random_5fold.json"
    )
    cand_repo_disjoint = (
        (repo / "kaggle_dataset/splits/doc_disjoint_split.json")
        if (repo / "kaggle_dataset/splits/doc_disjoint_split.json").exists()
        else repo / "artifacts/task1/data/splits/doc_disjoint_split.json"
    )

    # 1. Random 5-fold split
    if cand_input_random.exists():
        random_path = cand_input_random.resolve()
        random_source = "input"
    elif cand_repo_random.exists():
        random_path = cand_repo_random.resolve()
        random_source = "repo"
    else:
        splits_out = working_dir / "splits"
        splits_out.mkdir(parents=True, exist_ok=True)
        random_path = (splits_out / "random_5fold.json").resolve()
        queries_path = canonical_data_dir / "queries_train.parquet"
        q_df = pd.read_parquet(queries_path) if queries_path.exists() else pd.DataFrame()
        queries_list = [{"query_id": str(qid)} for qid in q_df.get("query_id", [])]
        from src.evaluation.splits import generate_random_5fold_split, verify_fold_isolation
        folds = generate_random_5fold_split(queries_list, seed=42, num_folds=5)
        qrels_path = canonical_data_dir / "qrels_train.parquet"
        qr_df = pd.read_parquet(qrels_path) if qrels_path.exists() else pd.DataFrame()
        qrels_map = defaultdict(list)
        if not qr_df.empty:
            for r in qr_df.to_dict("records"):
                qrels_map[str(r["query_id"])].append(str(r["doc_id"]))
        verify_fold_isolation(folds, qrels_map)
        random_path.write_text(json.dumps(folds, indent=2), encoding="utf-8")
        random_source = "generated"

    # 2. Document-disjoint split
    if cand_input_disjoint.exists():
        disjoint_path = cand_input_disjoint.resolve()
        disjoint_source = "input"
    elif cand_repo_disjoint.exists():
        disjoint_path = cand_repo_disjoint.resolve()
        disjoint_source = "repo"
    else:
        splits_out = working_dir / "splits"
        splits_out.mkdir(parents=True, exist_ok=True)
        disjoint_path = (splits_out / "doc_disjoint_split.json").resolve()
        docs_path = canonical_data_dir / "documents.parquet"
        queries_path = canonical_data_dir / "queries_train.parquet"
        qrels_path = canonical_data_dir / "qrels_train.parquet"
        d_df = pd.read_parquet(docs_path) if docs_path.exists() else pd.DataFrame()
        q_df = pd.read_parquet(queries_path) if queries_path.exists() else pd.DataFrame()
        qr_df = pd.read_parquet(qrels_path) if qrels_path.exists() else pd.DataFrame()
        d_list = [{"doc_id": str(did)} for did in d_df.get("doc_id", [])]
        q_list = [{"query_id": str(qid)} for qid in q_df.get("query_id", [])]
        qr_list = qr_df.to_dict("records") if not qr_df.empty else []
        from src.evaluation.splits import generate_document_disjoint_split, verify_document_disjoint_isolation
        dj_split = generate_document_disjoint_split(q_list, qr_list, val_ratio=0.2, seed=42)
        qrels_map = defaultdict(list)
        for r in qr_list:
            qrels_map[str(r["query_id"])].append(str(r["doc_id"]))
        verify_document_disjoint_isolation(dj_split, qrels_map)
        disjoint_path.write_text(json.dumps(dj_split, indent=2), encoding="utf-8")
        disjoint_source = "generated"

    random_sha = compute_file_sha256(random_path)
    disjoint_sha = compute_file_sha256(disjoint_path)

    return SplitArtifactResolution(
        random_5fold_path=random_path,
        doc_disjoint_path=disjoint_path,
        random_source=random_source,
        doc_disjoint_source=disjoint_source,
        random_sha256=random_sha,
        doc_disjoint_sha256=disjoint_sha,
    )


def resolve_duplicate_groups_path(
    canonical_data_dir: str | Path,
    repo_root: str | Path | None = None,
) -> tuple[Path | None, str]:
    """Resolve duplicate_groups.json across canonical data dir and repo root."""
    canonical_data_dir = Path(canonical_data_dir)
    repo = Path(repo_root) if repo_root else Path.cwd()

    candidates = [
        (canonical_data_dir / "duplicate_groups.json", "input"),
        (canonical_data_dir / "splits/duplicate_groups.json", "input_splits"),
        (repo / "kaggle_dataset/duplicate_groups.json", "repo"),
        (repo / "artifacts/task1/data/duplicate_groups.json", "repo"),
    ]
    for cand, src_name in candidates:
        if cand.exists():
            return cand.resolve(), src_name
    return None, "none"


def validate_duplicate_groups_file(
    duplicate_groups_path: str | Path | None,
    corpus_doc_ids: set[str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate duplicate_groups.json format, count (4 groups), and corpus validity."""
    if duplicate_groups_path is None:
        if strict:
            raise FileNotFoundError("duplicate_groups.json could not be resolved and is required in production")
        return {"is_valid": False, "errors": ["duplicate_groups_path is None"]}

    p = Path(duplicate_groups_path)
    if not p.exists():
        if strict:
            raise FileNotFoundError(f"duplicate_groups.json not found at: {p}")
        return {"is_valid": False, "errors": [f"File not found: {p}"]}

    errors: list[str] = []
    try:
        dup_groups = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        if strict:
            raise ValueError(f"Failed to parse duplicate groups JSON {p}: {e}")
        return {"is_valid": False, "errors": [str(e)]}

    if not isinstance(dup_groups, dict):
        errors.append(f"duplicate_groups must be a JSON object mapping group_id -> list[doc_id], got {type(dup_groups)}")

    group_count = len(dup_groups)
    if group_count != 4:
        errors.append(f"Expected exactly 4 duplicate groups, got {group_count}")

    total_dup_docs = set()
    invalid_ids = set()
    for g_id, doc_list in dup_groups.items():
        if not isinstance(doc_list, list) or len(doc_list) < 2:
            errors.append(f"Group {g_id} must have >= 2 doc IDs, got {doc_list}")
            continue
        for did in doc_list:
            s_did = str(did)
            total_dup_docs.add(s_did)
            if corpus_doc_ids is not None and s_did not in corpus_doc_ids:
                invalid_ids.add(s_did)

    if invalid_ids:
        errors.append(f"Duplicate doc IDs not in corpus: {sorted(invalid_ids)}")

    is_valid = len(errors) == 0
    if strict and not is_valid:
        raise ValueError(f"Duplicate groups validation failed: {'; '.join(errors)}")

    return {
        "is_valid": is_valid,
        "group_count": group_count,
        "total_duplicate_docs": len(total_dup_docs),
        "invalid_ids": sorted(invalid_ids),
        "errors": errors,
    }


CANONICAL_REQUIRED_FILES = {
    "documents.parquet",
    "chunks.parquet",
    "queries_train.parquet",
    "qrels_train.parquet",
}


def discover_data_dir(
    data_dir: str | Path | None = None, repo_root: Path | None = None
) -> Path:
    """Discover canonical dataset or build from raw competition files if missing.

    Prioritizes official Kaggle clean dataset mounts (e.g. phucdangg/legalir-task1-clean-data)
    and recursively scans /kaggle/input when needed.
    """
    if data_dir is not None:
        p = Path(data_dir)
        if not p.exists():
            raise FileNotFoundError(f"Explicitly provided data_dir does not exist: {data_dir}")
        missing_files = [f for f in CANONICAL_REQUIRED_FILES if not (p / f).exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Explicitly provided data_dir at {p} is missing required canonical files: {missing_files}"
            )
        return p.resolve()

    repo = repo_root or Path.cwd()
    candidate_paths = [
        # Preferred live clean dataset mounts
        Path("/kaggle/input/legalir-task1-clean-data"),
        Path("/kaggle/input/legalir-task1-clean-data/artifacts/task1/data"),
        Path("/kaggle/input/legalir-task1-clean-data/artifacts/shared/canonical/v2"),
        Path("/kaggle/input/datasets/phucdangg/legalir-task1-clean-data"),
        Path("/kaggle/input/datasets/phucdangg/legalir-task1-clean-data/artifacts/task1/data"),
        Path("/kaggle/input/datasets/phucdangg/legalir-task1-clean-data/artifacts/shared/canonical/v2"),
        Path("/kaggle/input/legalir"),
        Path("/kaggle/input/legalir/artifacts/task1/data"),
        Path("/kaggle/input/legalir/artifacts/shared/canonical/v2"),
        Path("/kaggle/input/legalir-task1/artifacts/task1/data"),
        Path("/kaggle/input/legalir-task-1/artifacts/task1/data"),
        Path("/kaggle/input/uit-dsc-2026-task1/artifacts/task1/data"),
        Path("/kaggle/input/legalir-dataset/artifacts/task1/data"),
        Path("/kaggle/input/legalir-canonical/artifacts/task1/data"),
        repo / "kaggle_dataset",
        Path.cwd() / "kaggle_dataset",
        repo / "artifacts/task1/data",
        repo / "artifacts/shared/canonical/v2",
        Path.cwd() / "artifacts/task1/data",
    ]
    for cand in candidate_paths:
        if cand.exists() and all((cand / f).exists() for f in CANONICAL_REQUIRED_FILES):
            return cand.resolve()

    # Recursive scan over /kaggle/input if present
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matching_dirs: list[Path] = []
        for dirpath in kaggle_input.rglob("*"):
            if dirpath.is_dir() and all((dirpath / f).exists() for f in CANONICAL_REQUIRED_FILES):
                matching_dirs.append(dirpath.resolve())

        if len(matching_dirs) == 1:
            return matching_dirs[0]
        elif len(matching_dirs) > 1:
            # Prioritize matching dirs containing preferred slug
            clean_matches = [d for d in matching_dirs if "legalir-task1-clean-data" in str(d)]
            if len(clean_matches) == 1:
                return clean_matches[0]
            legalir_matches = [d for d in matching_dirs if "legalir" in str(d)]
            if len(legalir_matches) == 1:
                return legalir_matches[0]
            raise ValueError(
                f"Ambiguous canonical dataset discovery in /kaggle/input; found multiple candidate directories: {matching_dirs}"
            )

    # Search for raw files to build on the fly if parquets are not precomputed
    raw_zip = None
    train_json = None
    for cand_zip in [
        Path("/kaggle/input/legalir-task1-clean-data/selected-contexts.zip"),
        Path("/kaggle/input/legalir/selected-contexts.zip"),
        Path("/kaggle/input/legalir/artifacts/raw/selected-contexts.zip"),
        Path("/kaggle/input/legalir/artifacts/shared/raw/selected-contexts.zip"),
        Path("/kaggle/input/legalir-task1/selected-contexts.zip"),
        Path("/kaggle/input/legalir-task-1/selected-contexts.zip"),
        Path("/kaggle/input/uit-dsc-2026-task1/selected-contexts.zip"),
        repo / "artifacts/shared/raw/selected-contexts.zip",
        repo / "artifacts/raw/selected-contexts.zip",
        repo / "selected-contexts.zip",
    ]:
        if cand_zip.exists():
            raw_zip = cand_zip
            break

    for cand_train in [
        Path("/kaggle/input/legalir-task1-clean-data/train.json"),
        Path("/kaggle/input/legalir/train.json"),
        Path("/kaggle/input/legalir/artifacts/raw/train.json"),
        Path("/kaggle/input/legalir/artifacts/shared/raw/train.json"),
        Path("/kaggle/input/legalir-task1/train.json"),
        Path("/kaggle/input/legalir-task-1/train.json"),
        Path("/kaggle/input/uit-dsc-2026-task1/train.json"),
        repo / "artifacts/shared/raw/train.json",
        repo / "artifacts/raw/train.json",
        repo / "train.json",
    ]:
        if cand_train.exists():
            train_json = cand_train
            break

    target_dir = (
        Path("/kaggle/working/legalir_run/data")
        if Path("/kaggle/working").exists()
        else repo / "artifacts/task1/data"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    if raw_zip and train_json:
        print(
            f"[*] Building canonical dataset from {raw_zip} and {train_json} into {target_dir}..."
        )
        build_canonical_package(
            raw_contexts_dir=raw_zip,
            train_json_path=train_json,
            output_dir=target_dir,
        )
    return target_dir.resolve()


def discover_public_test_file(
    public_json_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    """Discover official test file (public-official.json or private-official.json) with robust relative and recursive fallback."""
    if public_json_path is not None:
        p = Path(public_json_path)
        if p.exists():
            return p.resolve()
        return None

    # Check environment variable overrides
    env_test_file = os.environ.get("LEGALIR_TEST_FILE")
    if env_test_file:
        p = Path(env_test_file)
        if p.is_file():
            return p.resolve()
        if data_dir is not None:
            dp = Path(data_dir)
            if (dp / env_test_file).is_file():
                return (dp / env_test_file).resolve()
        if repo_root is not None:
            if (repo_root / env_test_file).is_file():
                return (repo_root / env_test_file).resolve()

    is_private_phase = os.environ.get("LEGALIR_TEST_PHASE", "").strip().lower() == "private"
    if is_private_phase:
        candidate_names = ["private-official.json", "private.json"]
    else:
        candidate_names = ["public-official.json", "public.json"]

    # Check relative to data_dir
    if data_dir is not None:
        dp = Path(data_dir)
        for cand_name in candidate_names:
            for cand in [
                dp / cand_name,
                dp.parent / cand_name,
                dp.parent.parent / cand_name,
                dp.parent / f"raw/{cand_name}",
                dp.parent / f"shared/raw/{cand_name}",
            ]:
                if cand.is_file():
                    return cand.resolve()

    repo = repo_root or Path.cwd()
    preferred_paths = []
    for cand_name in candidate_names:
        preferred_paths.extend([
            Path(f"/kaggle/input/legalir-task1-clean-data/{cand_name}"),
            Path(f"/kaggle/input/legalir-task1-clean-data/artifacts/raw/{cand_name}"),
            Path(f"/kaggle/input/datasets/phucdangg/legalir-task1-clean-data/{cand_name}"),
            Path(f"/kaggle/input/legalir/{cand_name}"),
            Path(f"/kaggle/input/legalir/artifacts/raw/{cand_name}"),
            Path(f"/kaggle/input/legalir/artifacts/shared/raw/{cand_name}"),
            Path(f"/kaggle/input/legalir-task1/{cand_name}"),
            Path(f"/kaggle/input/legalir-task-1/{cand_name}"),
            Path(f"/kaggle/input/uit-dsc-2026-task1/{cand_name}"),
            Path(f"/kaggle/input/legalir-dataset/{cand_name}"),
            repo / f"artifacts/shared/raw/{cand_name}",
            repo / f"artifacts/raw/{cand_name}",
            repo / cand_name,
            Path.cwd() / cand_name,
        ])
    for cand in preferred_paths:
        if cand.is_file():
            return cand.resolve()

    # Recursive scan in /kaggle/input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for cand_name in candidate_names:
            found = list(kaggle_input.rglob(cand_name))
            if found:
                return found[0].resolve()

    if is_private_phase:
        raise FileNotFoundError(
            "LEGALIR_TEST_PHASE=private is requested, but private test queries (private-official.json) could not be found. "
            "Refusing to silently fallback to public test queries."
        )

    return None


def run_kaggle_pipeline(
    *,
    data_dir: str | Path | None = None,
    working_dir: str | Path = "/kaggle/working/legalir_run",
    run_mode: Literal["smoke", "gpu_smoke", "full"] | str = "full",
    hf_token: str | None = None,
    devices: list[str] | None = None,
    runtime_config_path: str | Path | None = "configs/kaggle.yaml",
    reranker_config_path: str | Path | None = "configs/experiments/reranker_lora.yaml",
    config_path: str | Path | None = None,
    public_json_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    dense_device: str | None = None,
    reranker_device: str | None = None,
    device_contract: Any | None = None,
    precision: str | None = None,
    strict_artifacts: bool | None = None,
    allow_nonstandard_production_devices: bool = False,
    backend: str | None = None,
    allow_stage_reuse: bool = False,
) -> KaggleRunResult:
    """
    Execute the complete 24-step high-score LegalIR production pipeline on Kaggle.

    Steps executed:
    1. Multi-GPU environment detection and device allocation (GPU 0: Dense, GPU 1: Reranker).
    2. Canonical dataset discovery, validation, and metadata enrichment.
    3. Preflight parameter budget audit (<4B rule).
    4. Dual BM25 index building/loading (raw legal + PyVi segmented) with metadata enrichment.
    5. DEk21 Dense Macro index building/loading on GPU 0.
    6. Precomputing and caching train query dense embeddings on GPU 0.
    7. Mining fold-isolated hard-negative training pairs.
    8. 5-Fold OOF cross-validation with fold-specific LoRA rerankers.
    9. Full cross-fitted Learned Fusion (LightGBM) vs Tuned RRF evaluation.
    10. Document-disjoint robustness split validation with dedicated trained reranker.
    11. Full 7,000-query question memory indexing with precomputed embeddings.
    12. Final LoRA reranker fine-tuning on all 7,000 training queries on GPU 1.
    13. Final fusion model fitting on all OOF candidate features (if learned fusion won).
    14. Public test query discovery and batch inference with the fully loaded production pipeline.
    15. Strict submission invariant validation (bounds, types, no duplicates, valid corpus IDs).
    16. Zip packaging containing strictly submission.json at root.
    17. Full artifact manifest creation with SHA-256 hashes and parameter audits.
    18. Support for run_mode="smoke", "gpu_smoke", and "full".
    """
    t_start = time.time()
    supervisor_start_utc = datetime.now(timezone.utc).isoformat()
    VALID_RUN_MODES = {"smoke", "gpu_smoke", "full"}
    run_mode_str = str(run_mode).lower().strip()
    if run_mode_str not in VALID_RUN_MODES:
        raise ValueError(
            f"Invalid run_mode: '{run_mode}'. Must be one of {sorted(VALID_RUN_MODES)}"
        )
    is_smoke = run_mode_str == "smoke"
    is_gpu_smoke = run_mode_str == "gpu_smoke"
    is_full = run_mode_str == "full"

    # Backend policy enforcement (QUALITY_RUNTIME_PLAN.md Section 1):
    # Kaggle backend is strictly restricted to bounded smoke contracts ('smoke', 'gpu_smoke').
    # FULL production training and official submission generation must run on Modal or Google Colab.
    def _safe_exists(p_str: str) -> bool:
        try:
            return Path(p_str).exists()
        except (PermissionError, OSError):
            return False

    resolved_backend = str(backend).lower().strip() if backend is not None else os.environ.get("LEGALIR_BACKEND", "").lower().strip()
    if not resolved_backend:
        if _safe_exists("/kaggle/working") and not _safe_exists("/content") and not _safe_exists("/root/legalir_volume"):
            resolved_backend = "kaggle"
        elif _safe_exists("/content"):
            resolved_backend = "colab"
        elif _safe_exists("/root/legalir_volume"):
            resolved_backend = "modal"
        else:
            resolved_backend = "local"

    if resolved_backend == "kaggle" and is_full:
        raise RuntimeError(
            "Backend policy violation: Kaggle backend is restricted to bounded smoke contracts ('smoke', 'gpu_smoke'). "
            "FULL production training and official submission generation must run on Modal or Google Colab with explicit authorization."
        )

    stage_timings = StageTimingTelemetry()

    if strict_artifacts is None:
        strict_artifacts = is_full or is_gpu_smoke

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
            except Exception:
                pass

    print("=" * 80)
    print("LEGALIR TASK 1: KAGGLE T4 x2 PRODUCTION ORCHESTRATOR")
    print(f"Execution Mode: {run_mode_str.upper()} | Start Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # 1. Setup paths and directories
    root_path = Path(repo_root) if repo_root else Path.cwd()
    working_path = Path(working_dir)
    working_path.mkdir(parents=True, exist_ok=True)
    index_dir = working_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    cv_dir = working_path / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = working_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = working_path / "training_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = working_path / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # 1. Canonical Dataset Parquet Existence Check
    canonical_data_dir = discover_data_dir(data_dir, repo_root=root_path)
    docs_path = canonical_data_dir / "documents.parquet"
    chunks_path = canonical_data_dir / "chunks.parquet"
    queries_train_path = canonical_data_dir / "queries_train.parquet"
    qrels_train_path = canonical_data_dir / "qrels_train.parquet"

    if not (docs_path.exists() and chunks_path.exists()):
        raise FileNotFoundError(
            f"Canonical dataset parquet files missing in {canonical_data_dir}"
        )

    # 2. Public / Private Test Discovery (Fail-Fast)
    public_test_file = discover_public_test_file(public_json_path, data_dir=canonical_data_dir, repo_root=root_path)
    if is_full and (public_test_file is None or not public_test_file.exists()):
        raise FileNotFoundError(f"{run_mode_str} mode requires official test query file (private-official.json or public-official.json); refusing to proceed")
    if is_gpu_smoke and public_json_path is not None and (public_test_file is None or not public_test_file.exists()):
        raise FileNotFoundError(f"{run_mode_str} mode requires official test query file (private-official.json or public-official.json); refusing to proceed")

    # 3. Hardware and GPU Device Allocation (P1.10)
    dense_device, reranker_device = resolve_pipeline_device_allocation(
        device_contract=device_contract,
        run_mode=run_mode_str,
        devices=devices,
        dense_device=dense_device,
        reranker_device=reranker_device,
        allow_nonstandard_production_devices=allow_nonstandard_production_devices,
    )

    print(f"[+] Device Allocation (P1.10 Multi-GPU Utilization):")
    print(f"    - Dense Embedding & Question Encoding : {dense_device}")
    print(f"    - Reranker Training & Neural Inference: {reranker_device}")

    # Resolve split artifacts with strict provenance tracking
    split_res = resolve_split_artifacts(
        canonical_data_dir=canonical_data_dir,
        working_dir=working_path,
        repo_root=root_path,
    )
    split_provenance_report = {
        "random_5fold": {"source": split_res.random_source, "sha256": split_res.random_sha256, "path": str(split_res.random_5fold_path)},
        "doc_disjoint": {"source": split_res.doc_disjoint_source, "sha256": split_res.doc_disjoint_sha256, "path": str(split_res.doc_disjoint_path)},
    }
    print(f"[+] Split Provenance: random_5fold from '{split_res.random_source}' (SHA: {split_res.random_sha256[:12]}...), "
          f"doc_disjoint from '{split_res.doc_disjoint_source}' (SHA: {split_res.doc_disjoint_sha256[:12]}...)")


    print(f"[+] Device Allocation (P1.10 Multi-GPU Utilization):")
    print(f"    - Dense Embedding & Question Encoding : {dense_device}")
    print(f"    - Reranker Training & Neural Inference: {reranker_device}")

    # Set HF token in environment if provided securely
    if hf_token:
        os.environ["HF_TOKEN"] = str(hf_token)

    # 3. Canonical Dataset Loading & Validation (P1.5)
    print(f"[+] Canonical Data Directory: {canonical_data_dir}")

    t_load0 = time.perf_counter()
    df_docs = pd.read_parquet(docs_path)
    df_chunks = pd.read_parquet(chunks_path)
    df_queries = pd.read_parquet(queries_train_path) if queries_train_path.exists() else pd.DataFrame()
    df_qrels = pd.read_parquet(qrels_train_path) if qrels_train_path.exists() else pd.DataFrame()

    manifest_path = canonical_data_dir / "manifest.json"
    audit_report_path = canonical_data_dir / "audit_report.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    audit_data = json.loads(audit_report_path.read_text(encoding="utf-8")) if audit_report_path.exists() else {}

    if is_full or is_gpu_smoke:
        if len(df_docs) != OFFICIAL_DOCUMENT_COUNT:
            raise ValueError(
                f"{run_mode_str.upper()} mode requires official Task 1 dataset with exactly {OFFICIAL_DOCUMENT_COUNT:,} documents, got {len(df_docs)}"
            )
        if len(df_queries) != OFFICIAL_TRAIN_QUERIES:
            raise ValueError(
                f"{run_mode_str.upper()} mode requires official Task 1 dataset with exactly {OFFICIAL_TRAIN_QUERIES:,} training queries, got {len(df_queries)}"
            )
        if manifest_data and manifest_data.get("version") not in (None, OFFICIAL_VERSION, "mock"):
            raise ValueError(
                f"{run_mode_str.upper()} mode requires v2 canonical dataset, got {manifest_data.get('version')}"
            )

    val_report = validate_canonical_dataset(
        canonical_data_dir,
        expected_document_count=OFFICIAL_DOCUMENT_COUNT if (is_full or is_gpu_smoke) else None,
    )
    print(f"[+] Canonical Dataset Validation: is_valid = {val_report.get('is_valid')}")
    if not val_report.get("is_valid") and (is_full or is_gpu_smoke):
        raise ValueError(f"Canonical dataset validation failed in {run_mode_str.upper()} mode: {val_report.get('errors')}")

    # Single load and validation of test queries (public 1,000 or private 2,080)
    is_priv = bool(os.environ.get("LEGALIR_TEST_PHASE", "").strip().lower() == "private" or (public_test_file is not None and "private" in public_test_file.name.lower()))
    expected_phase_count = OFFICIAL_PRIVATE_QUERY_COUNT if is_priv else OFFICIAL_PUBLIC_QUERY_COUNT
    phase_label = "PRIVATE" if is_priv else "PUBLIC"

    if public_test_file and public_test_file.exists():
        print(f"[+] Found {phase_label} Test Queries: {public_test_file}")
        with open(public_test_file, "r", encoding="utf-8") as f:
            public_data = json.load(f)
        if (is_full or is_gpu_smoke) and len(public_data) != expected_phase_count:
            raise ValueError(
                f"{run_mode_str.upper()} mode in {phase_label} phase requires official test query file with exactly {expected_phase_count} queries, got {len(public_data)}"
            )
    else:
        print("[!] public-official.json not found. Using train queries sample for inference verification.")
        sample_records = df_queries.head(20).to_dict("records")
        public_data = {
            str(r["query_id"]): {
                "question": str(
                    r.get("question_norm") or r.get("question_raw") or r.get("question") or ""
                )
            }
            for r in sample_records
        }
    official_public_qids = set(public_data.keys())
    print(f"[+] Total Public Test Queries: {len(public_data)}")

    # Unified official identity validation
    strict_identity = bool((is_full or is_gpu_smoke) and (manifest_data.get("version") != "mock"))
    if manifest_path.exists() or audit_report_path.exists() or strict_identity:
        official_identity_report = validate_official_task1_identity(
            data_dir=canonical_data_dir,
            public_json_path=public_test_file,
            strict=strict_identity,
        )
    else:
        official_identity_report = {"is_valid": True, "public_queries": len(public_data)}
    print(f"[+] Official Identity Verification: is_valid = {official_identity_report.get('is_valid')}")

    # Resolve and validate duplicate groups false-negative blacklist
    dup_groups_path, dup_groups_source = resolve_duplicate_groups_path(
        canonical_data_dir=canonical_data_dir,
        repo_root=root_path,
    )
    strict_dup = bool((is_full or is_gpu_smoke) and (manifest_data.get("version") != "mock"))
    corpus_ids = set(df_docs["doc_id"].astype(str)) if ("doc_id" in df_docs.columns and strict_dup) else None
    dup_report = validate_duplicate_groups_file(
        dup_groups_path,
        corpus_doc_ids=corpus_ids,
        strict=strict_dup,
    )
    print(f"[+] Duplicate Groups Blacklist: source='{dup_groups_source}' | groups={dup_report.get('group_count')} | "
          f"docs={dup_report.get('total_duplicate_docs')} | valid={dup_report.get('is_valid')}")

    canonical_load_time = max(0.001, time.perf_counter() - t_load0)
    stage_timings.record("canonical_load", elapsed_seconds=canonical_load_time, cache_hit=False)

    print(f"[+] Dataset Loaded: {len(df_docs):,} documents | {len(df_chunks):,} chunks | {len(df_queries):,} train queries")
    rss_after_load_mb = get_process_rss_mb()

    # 4. Strict Parameter Budget Preflight Audit (<4B Rule)
    if runtime_config_path:
        resolved_runtime_config = resolve_repo_path(runtime_config_path, root_path)
    elif config_path:
        resolved_runtime_config = resolve_repo_path(config_path, root_path)
    else:
        resolved_runtime_config = resolve_repo_path("configs/kaggle.yaml", root_path)

    if not resolved_runtime_config.exists():
        fallback_cfg = resolve_repo_path("configs/pipeline.yaml", root_path)
        if fallback_cfg.exists():
            resolved_runtime_config = fallback_cfg

    if reranker_config_path:
        resolved_reranker_config = resolve_repo_path(reranker_config_path, root_path)
    else:
        resolved_reranker_config = resolve_repo_path("configs/experiments/reranker_lora.yaml", root_path)

    if (is_full or is_gpu_smoke) and not resolved_reranker_config.is_file():
        raise FileNotFoundError(
            f"Reranker training config not found at: {resolved_reranker_config}"
        )

    if not resolved_reranker_config.exists():
        resolved_reranker_config = resolve_repo_path("configs/pipeline.yaml", root_path)

    preflight_json_path = working_path / "preflight_parameter_audit.json"
    preflight_audit_report = audit_system_parameters(
        config_path=resolved_runtime_config if resolved_runtime_config.exists() else None,
        output_json=preflight_json_path,
        raise_on_violation=True,
        offline_fallback=True,
    )
    print(
        f"[+] Parameter Budget Preflight: {preflight_audit_report['total_learned_parameters']:,} params "
        f"({preflight_audit_report['total_parameters_billions']:.4f}B / 4.0B limit, "
        f"{preflight_audit_report['budget_utilization_pct']:.2f}% utilization). PASS"
    )

    # Validate explicit reranker training config
    from src.training.trainer import compute_coverage_required_steps

    reranker_cfg = yaml.safe_load(resolved_reranker_config.read_text(encoding="utf-8")) if resolved_reranker_config.exists() else {}
    runtime_cfg = yaml.safe_load(resolved_runtime_config.read_text(encoding="utf-8")) if resolved_runtime_config.exists() else {}
    runtime_section = runtime_cfg.get("runtime", {}) if isinstance(runtime_cfg, dict) else {}
    runtime_num_workers = runtime_section.get("num_workers") if isinstance(runtime_section, dict) else None
    if is_full:
        configured_max_steps = reranker_cfg.get("max_steps", 500)
        if configured_max_steps is None:
            raise ValueError("Full reranker training requires explicit max_steps")
        n_train_queries = len(df_queries) if not df_queries.empty else 7000
        coverage_req_steps = compute_coverage_required_steps(
            eligible_query_count=n_train_queries,
            batch_size=int(reranker_cfg.get("batch_size", 2)),
            gradient_accumulation_steps=int(reranker_cfg.get("gradient_accumulation_steps", 8)),
            target_coverage_pct=1.0,
            require_pos_and_neg=True,
        )
        effective_max_steps = max(int(configured_max_steps), coverage_req_steps)
        print(f"[+] Effective Reranker Config: model={reranker_cfg.get('base_model_name', 'BAAI/bge-reranker-v2-m3')}, "
              f"batch_size={reranker_cfg.get('batch_size', 2)}, grad_accum={reranker_cfg.get('gradient_accumulation_steps', 8)}, "
              f"max_length={reranker_cfg.get('max_length', 512)}, configured_steps={configured_max_steps}, "
              f"coverage_required_steps={coverage_req_steps}, effective_max_steps={effective_max_steps}, "
              f"fp16={reranker_cfg.get('fp16', True)}, device={reranker_device}")
    elif is_gpu_smoke:
        configured_max_steps = 3
        coverage_req_steps = 3
        effective_max_steps = 3
    else:
        configured_max_steps = 5
        coverage_req_steps = 5
        effective_max_steps = 5

    # 5. Build / Load Dual BM25 Indexes with Metadata Enrichment (P1.5)
    micro_chunks = (
        df_chunks[df_chunks["granularity"] == "micro"]
        if "granularity" in df_chunks.columns
        else df_chunks
    )
    if docs_path.exists():
        micro_chunks = enrich_chunks_with_doc_metadata(micro_chunks, docs_path)

    # 5a. Fielded Legal BM25
    bm25_dir = index_dir / "bm25"
    t_bm25_0 = time.perf_counter()
    bm25_cached = (bm25_dir / "bm25_micro_index.pkl").exists() or (bm25_dir.exists() and list(bm25_dir.glob("*.pkl")))
    if bm25_cached:
        print(f"[*] Loading cached Legal BM25 index from {bm25_dir}...")
        bm25_legal = BM25MicroRetriever.load(bm25_dir)
    else:
        print(f"[*] Building Legal BM25 index with metadata enrichment...")
        bm25_legal = BM25MicroRetriever(k1=1.5, b=0.75).fit(micro_chunks.to_dict("records"), show_progress=False)
        bm25_legal.save(bm25_dir)
    bm25_legal_time = max(0.001, time.perf_counter() - t_bm25_0)
    stage_timings.record("bm25_legal", elapsed_seconds=bm25_legal_time, cache_hit=bool(bm25_cached))
    print(f"[+] Legal BM25 ready ({len(bm25_legal.corpus):,} docs).")

    # 5b. PyVi Segmented BM25
    bm25_pyvi_dir = index_dir / "bm25_pyvi"
    t_pyvi_0 = time.perf_counter()
    bm25_pyvi_cached = (bm25_pyvi_dir / "bm25_pyvi_index.pkl").exists() or (bm25_pyvi_dir.exists() and list(bm25_pyvi_dir.glob("*.pkl")))
    if bm25_pyvi_cached:
        print(f"[*] Loading cached PyVi BM25 index from {bm25_pyvi_dir}...")
        bm25_pyvi = BM25PyViRetriever.load(bm25_pyvi_dir)
    else:
        print(f"[*] Building PyVi BM25 index with metadata enrichment...")
        bm25_pyvi = BM25PyViRetriever(k1=1.5, b=0.75).fit(micro_chunks.to_dict("records"), show_progress=False)
        bm25_pyvi.save(bm25_pyvi_dir)
    bm25_pyvi_time = max(0.001, time.perf_counter() - t_pyvi_0)
    stage_timings.record("bm25_pyvi", elapsed_seconds=bm25_pyvi_time, cache_hit=bool(bm25_pyvi_cached))
    print(f"[+] PyVi BM25 ready ({len(bm25_pyvi.corpus):,} docs).")

    # 6. Build / Load DEk21 Dense Macro Index (GPU 0)
    dense_dir = index_dir / "dense_dek21"
    dense_retriever: DenseMacroRetriever | None = None
    dense_build_time = 0.001
    dense_cached = (dense_dir / "embeddings.npy").exists()

    from src.retrieval.dense_macro import pinned_dense_revision as _pinned_dense_rev

    _dense_model_id = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    _dense_expected_rev = _pinned_dense_rev(_dense_model_id)
    t_dense0 = time.perf_counter()
    if dense_cached:
        print(f"[*] Loading cached DEk21 Dense index from {dense_dir} on {dense_device}...")
        try:
            dense_retriever = DenseMacroRetriever.load(
                dense_dir,
                model_name=_dense_model_id,
                revision=_dense_expected_rev,
                device=dense_device,
            )
            dense_build_time = max(0.001, time.perf_counter() - t_dense0)
        except Exception as e:
            if is_full or is_gpu_smoke:
                raise RuntimeError(f"Failed to load DEk21 Dense index from {dense_dir} in {run_mode_str.upper()} mode: {e}") from e
            print(f"[-] Warning: Failed to load dense index: {e}")
    elif not is_smoke:
        print(f"[*] Building DEk21 Dense index on {dense_device}...")
        macro_chunks = (
            df_chunks[df_chunks["granularity"] == "macro"]
            if "granularity" in df_chunks.columns
            else df_chunks
        )
        dense_retriever = DenseMacroRetriever(
            model_name=_dense_model_id,
            revision=_dense_expected_rev,
            device=dense_device,
            dimension=768,
        )
        # CUDA encodes 219k macro chunks: larger batches only reduce forward
        # overhead (order-preserving concatenation; OOM still fails closed).
        dense_batch = 128 if "cuda" in str(dense_device) else 32
        try:
            dense_retriever.fit(macro_chunks.to_dict("records"), batch_size=dense_batch, stage_name="corpus")
            dense_retriever.save(dense_dir)
            dense_build_time = max(0.001, time.perf_counter() - t_dense0)
            print(f"[+] DEk21 Dense Index ready ({len(dense_retriever.doc_ids):,} chunks).")
        except Exception as e:
            if is_full or is_gpu_smoke:
                raise RuntimeError(f"Failed to build DEk21 Dense index on {dense_device} in {run_mode_str.upper()} mode: {e}") from e
            print(f"[-] Warning: Dense index building failed: {e}")
            dense_retriever = None
    else:
        dense_build_time = max(0.001, time.perf_counter() - t_dense0)

    stage_timings.record("dense_index", elapsed_seconds=dense_build_time, cache_hit=bool(dense_cached))
    rss_after_dense_mb = get_process_rss_mb()

    if is_full or is_gpu_smoke:
        if dense_retriever is None or getattr(dense_retriever, "_faiss_index", None) is None:
            raise RuntimeError(
                f"FAISS production backend enforcement failed in {run_mode_str.upper()} mode: "
                "dense_retriever._faiss_index is None (NumPy fallback is not allowed in production)."
            )

    # 7. Precompute Train Query Dense Embeddings on GPU 0 (P1.10)
    # Cache rows are positional; reuse must validate query order/content and
    # encoder identity, otherwise stale rows silently misalign.
    train_query_embs: dict[str, np.ndarray] = {}
    train_query_enc_time = 0.001
    tq_npy = index_dir / "train_query_embeddings.npy"
    tq_cached = tq_npy.exists()
    t_tq0 = time.perf_counter()
    if not df_queries.empty and dense_retriever is not None:
        q_records = df_queries.to_dict("records")
        qids = [str(r["query_id"]) for r in q_records]
        qtexts = [
            str(r.get("question_norm") or r.get("question_raw") or r.get("question") or "")
            for r in q_records
        ]
        _enc_model = str(getattr(dense_retriever, "model_name", getattr(dense_retriever, "model_name_or_path", "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2")))
        _enc_rev = getattr(dense_retriever, "revision", None)
        if not _enc_rev:
            try:
                from src.models.bootstrap import MODEL_REGISTRY as _EREG

                _enc_rev = (_EREG.get(_enc_model) or {}).get("revision")
            except Exception:
                _enc_rev = None
        _cache_valid = False
        _cached_embs = None
        _expected_qdim = int(getattr(dense_retriever, "dimension", 768))
        if tq_cached:
            try:
                _cached_embs = load_validated_train_query_embeddings(
                    index_dir, qids, qtexts, _enc_model, _enc_rev,
                    expected_dim=_expected_qdim,
                )
                _cache_valid = True
            except Exception as _ce:
                print(f"[-] Train query embedding cache rejected ({_ce}); recomputing.")
                _cache_valid = False
                tq_cached = False
        if _cache_valid and _cached_embs is not None:
            print(f"[*] Loading validated cached train query dense embeddings from {tq_npy}...")
            for qid, emb in zip(qids, _cached_embs):
                train_query_embs[qid] = emb
            train_query_enc_time = max(0.001, time.perf_counter() - t_tq0)
        else:
            try:
                print(f"[*] Precomputing train query embeddings for {len(qids):,} queries on {dense_device}...")
                embs = dense_retriever.encode_queries(
                    qtexts, batch_size=128 if "cuda" in str(dense_device) else 32, stage_name="train_query"
                )
                for qid, emb in zip(qids, embs):
                    train_query_embs[qid] = emb
                write_train_query_embedding_cache(
                    index_dir, qids, qtexts, embs, _enc_model, _enc_rev
                )
                train_query_enc_time = max(0.001, time.perf_counter() - t_tq0)
                print(f"[+] Precomputed and cached {len(train_query_embs):,} train query dense embeddings.")
            except Exception as e:
                train_query_enc_time = max(0.001, time.perf_counter() - t_tq0)
                if is_full or is_gpu_smoke:
                    raise RuntimeError(f"Failed to precompute train query embeddings on {dense_device} in {run_mode_str.upper()} mode: {e}") from e
                print(f"[-] Warning: query embedding precomputation skipped: {e}")
    else:
        train_query_enc_time = max(0.001, time.perf_counter() - t_tq0)

    stage_timings.record("train_query_encoding", elapsed_seconds=train_query_enc_time, cache_hit=bool(tq_cached))

    # 8. Out-of-Fold (OOF) 5-Fold Cross-Validation with Fold-Trained LoRA Rerankers (P1.1, P1.2, P1.7, P1.10)
    print("\n" + "=" * 70)
    print(f"[*] Starting 5-Fold OOF Cross-Validation (Mode: {run_mode_str.upper()})...")
    print("=" * 70)

    t_oof0 = time.perf_counter()
    # Normalize precision end-to-end (A100 bf16, T4 fp16 fallback via config)
    _prec_norm = str(precision or "bf16").lower().strip() if (precision or "") else None
    if _prec_norm == "bfloat16":
        _prec_norm = "bf16"
    oof_runner = OOFRunner(
        data_dir=canonical_data_dir,
        index_dir=index_dir,
        output_dir=cv_dir,
        splits_path=split_res.random_5fold_path,
        doc_disjoint_splits_path=split_res.doc_disjoint_path,
        config_path=resolved_runtime_config,
        reranker_config_path=resolved_reranker_config,
        num_folds=2 if (is_smoke or is_gpu_smoke) else 5,
        candidate_k=20 if (is_smoke or is_gpu_smoke) else 200,
        rerank_k=10 if (is_smoke or is_gpu_smoke) else int(reranker_cfg.get("rerank_k", 100)),
        use_reranker=True,
        reranker_model="mock" if is_smoke else "BAAI/bge-reranker-v2-m3",
        train_reranker_per_fold=(not is_smoke),
        dense_device=dense_device,
        reranker_device=reranker_device,
        smoke=(is_smoke or is_gpu_smoke),
        smoke_sample_size=20 if (is_smoke or is_gpu_smoke) else 50,
        doc_disjoint=True,
        train_query_embeddings=train_query_embs,
        duplicate_groups_path=dup_groups_path,
        split_provenance=split_provenance_report,
        precision=_prec_norm,
        num_workers=runtime_num_workers,
        reranker_batch_size=int(reranker_cfg.get("inference_batch_size") or reranker_cfg.get("batch_size", 16)),
        reranker_max_length=int(reranker_cfg.get("max_length", 384)),
        allow_stage_reuse=bool(allow_stage_reuse) and not (is_full or is_gpu_smoke),
    )
    cv_report = oof_runner.run()
    oof_cv_time = max(0.001, time.perf_counter() - t_oof0)
    stage_timings.record("oof_cv", elapsed_seconds=oof_cv_time, cache_hit=False)

    oof_num_folds = int(oof_runner.num_folds)
    doc_disjoint_report = dict(getattr(oof_runner, "doc_disjoint_report", {}) or {})
    rss_after_oof_mb = get_process_rss_mb()
    shared_static_branch_cache = getattr(oof_runner, "_static_branch_cache", {})
    del oof_runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 9. Cross-Fitted Learned Fusion vs Tuned RRF Evaluation & Final Model Training (P1.4)
    print("\n" + "=" * 70)
    print("[*] Evaluating Cross-Fitted Learned Fusion vs Tuned RRF (P1.4)...")
    print("=" * 70)

    t_fus0 = time.perf_counter()
    oof_feat_path = cv_dir / "oof_features.parquet"
    fusion_report: dict[str, Any] = {}
    if oof_feat_path.exists():
        oof_df = pd.read_parquet(oof_feat_path)
        if not oof_df.empty and "fold" in oof_df.columns:
            qrels_dict = defaultdict(list)
            for r in df_qrels.to_dict("records"):
                qrels_dict[str(r["query_id"])].append(str(r["doc_id"]))

            fusion_final_dir = checkpoints_dir / "fusion_final"
            # F7: FULL confirmation uses the predeclared fixed RRF policy (no
            # outer-label fitting/early-stopping/selection). Bounded smoke keeps
            # the development selection path.
            fusion_report = train_and_evaluate_fusion_cv(
                oof_df=oof_df,
                qrels_dict=qrels_dict,
                output_dir=fusion_final_dir,
                num_boost_round=100 if not (is_smoke or is_gpu_smoke) else 10,
                fusion_policy="predeclared_rrf" if is_full else "select_using_outer_labels",
            )
    fusion_time = max(0.001, time.perf_counter() - t_fus0)
    stage_timings.record("fusion_training", elapsed_seconds=fusion_time, cache_hit=False)

    winning_fusion_method = fusion_report.get("winning_method", "reciprocal_rank_fusion")
    use_learned_fusion = bool(winning_fusion_method == "learned_ranker")

    # 10. Build Full 7,000-Query Memory & Train Final LoRA Reranker on All 7,000 Queries (P1.3, P1.8, P1.10, P1.11)
    print("\n" + "=" * 70)
    print(f"[*] Training Final System on All Training Queries (P1.3 / P1.8 / P1.11)...")
    print("=" * 70)

    # 10a. Full Question Memory
    t_qm0 = time.perf_counter()
    if not df_queries.empty and not df_qrels.empty:
        queries_dict = {
            str(r["query_id"]): str(
                r.get("question_norm") or r.get("question_raw") or r.get("question") or ""
            )
            for r in df_queries.to_dict("records")
        }
        qrels_dict_full = defaultdict(list)
        for r in df_qrels.to_dict("records"):
            qrels_dict_full[str(r["query_id"])].append(str(r["doc_id"]))

        queries_for_memory = [
            (qid, queries_dict[qid], train_query_embs.get(qid))
            if qid in train_query_embs
            else (qid, queries_dict[qid], None)
            for qid in queries_dict.keys()
        ]
        has_dense_memory = (dense_retriever is not None) or bool(train_query_embs)
        full_memory = TrainQuestionMemory(
            min_similarity=0.82,
            dense_encoder=dense_retriever,
            use_dense=has_dense_memory,
            dense_device=dense_device,
        )
        full_memory.fit(queries_for_memory, qrels_dict_full)
        if len(full_memory.qids) == 0:
            raise ValueError("Final Question Memory has 0 indexed queries")
        full_mem_dir = index_dir / "question_memory"
        full_memory.save(full_mem_dir)
        print(f"[+] Full Question Memory Index saved to {full_mem_dir} ({len(full_memory.qids):,} queries).")
    qm_time = max(0.001, time.perf_counter() - t_qm0)
    stage_timings.record("question_memory", elapsed_seconds=qm_time, cache_hit=False)

    # 10b. Mine hard-negative training pairs from ALL training queries
    t_pm0 = time.perf_counter()
    limit_pairs = 100 if (is_smoke or is_gpu_smoke) else None
    print(f"[*] Mining hard-negative training pairs on all training queries (limit={limit_pairs})...")
    build_training_pairs(
        data_dir=canonical_data_dir,
        index_dir=index_dir,
        output_dir=pairs_dir,
        fold=None,
        use_all_queries=True,
        limit=limit_pairs,
        negatives_per_positive=8,
        query_embeddings=train_query_embs,
        duplicate_groups_path=dup_groups_path,
        static_branch_cache=shared_static_branch_cache,
    )
    final_pairs_file = pairs_dir / "reranker_pairs.parquet"
    final_pair_mining_time = max(0.001, time.perf_counter() - t_pm0)
    stage_timings.record("final_pair_mining", elapsed_seconds=final_pair_mining_time, cache_hit=False)

    # Audit pair coverage before loading/training final BGE reranker
    from src.training.trainer import audit_pair_coverage
    if final_pairs_file.exists():
        final_pair_df = pd.read_parquet(final_pairs_file)
        expected_train_qids = set(df_queries["query_id"].astype(str)) if not df_queries.empty else None
        pair_coverage_audit = audit_pair_coverage(final_pair_df, expected_qids=expected_train_qids)
        print(f"[+] Final Training Pair Coverage Audit: {pair_coverage_audit.get('eligible_queries_count'):,} eligible queries | "
              f"Positive Coverage: {pair_coverage_audit.get('positive_coverage_pct')}% | "
              f"Negative Coverage: {pair_coverage_audit.get('negative_coverage_pct')}%")
        if is_full:
            if pair_coverage_audit["positive_coverage_pct"] < 100.0:
                raise RuntimeError(f"FULL mode requires 100% positive pair coverage, got {pair_coverage_audit['positive_coverage_pct']}%")
            if pair_coverage_audit["negative_coverage_pct"] < 99.0:
                raise RuntimeError(f"FULL mode requires >=99% negative pair coverage, got {pair_coverage_audit['negative_coverage_pct']}%")

    # 10c. Train Final LoRA Reranker on GPU 1
    # F4: completed-stage reuse DISABLED by default (and never in FULL/gpu_smoke
    # until the F4 contract is complete). Modal uses a fresh UUID attempt dir per
    # invocation (no cross-attempt resume). Explicit opt-in only for bounded dev.
    final_reranker_dir = checkpoints_dir / "reranker_final"
    final_complete_file = final_reranker_dir / "complete.json"
    reusing_final_reranker = False
    _final_reuse_allowed = bool(allow_stage_reuse) and not (is_full or is_gpu_smoke)
    if _final_reuse_allowed and final_complete_file.is_file() and (final_reranker_dir / "adapter_config.json").is_file():
        try:
            _candidate_report = json.loads(final_complete_file.read_text(encoding="utf-8"))
            _reuse_ok = True
            _reuse_reason = ""
            _expected_base = "mock" if is_smoke else "BAAI/bge-reranker-v2-m3"
            if _candidate_report.get("status") not in ("completed", "PASS"):
                _reuse_ok = False
                _reuse_reason = f"status={_candidate_report.get('status')}"
            elif str(_candidate_report.get("base_model") or "") != _expected_base:
                # Allow a recorded local snapshot path only when it exists.
                _manifest_base = str(_candidate_report.get("base_model") or "")
                try:
                    _is_local = Path(_manifest_base).expanduser().is_dir()
                except Exception:
                    _is_local = False
                if not (_is_local and _expected_base != "mock"):
                    _reuse_ok = False
                    _reuse_reason = f"base_model mismatch (expected {_expected_base}, got {_manifest_base})"
            if _reuse_ok and _expected_base != "mock":
                try:
                    from src.models.bootstrap import MODEL_REGISTRY as _REG

                    _expected_rev = (_REG.get(_expected_base) or {}).get("revision")
                except Exception:
                    _expected_rev = None
                _manifest_rev = (
                    _candidate_report.get("base_model_revision")
                    or _candidate_report.get("revision")
                    or _candidate_report.get("reranker_revision")
                )
                if not _manifest_rev:
                    _reuse_ok = False
                    _reuse_reason = "base_model_revision missing"
                elif _expected_rev and str(_manifest_rev).strip() != str(_expected_rev).strip():
                    _reuse_ok = False
                    _reuse_reason = f"base_model_revision mismatch (expected {_expected_rev}, got {_manifest_rev})"
            if _reuse_ok:
                _weights = final_reranker_dir / "adapter_model.safetensors"
                if not _weights.exists():
                    _weights = final_reranker_dir / "adapter_model.bin"
                if not _weights.is_file():
                    _reuse_ok = False
                    _reuse_reason = "adapter weights missing"
                else:
                    _recorded_sha = _candidate_report.get("adapter_checksum")
                    if _recorded_sha:
                        import hashlib as _hl

                        _actual_sha = _hl.sha256(_weights.read_bytes()).hexdigest()
                        if _actual_sha != _recorded_sha:
                            _reuse_ok = False
                            _reuse_reason = "adapter checksum mismatch"
                    if int(_candidate_report.get("optimizer_steps", _candidate_report.get("global_steps", 0)) or 0) <= 0:
                        _reuse_ok = False
                        _reuse_reason = "optimizer_steps <= 0"
                    if float(_candidate_report.get("param_diff", 0) or 0) <= 0:
                        _reuse_ok = False
                        _reuse_reason = "param_diff <= 0"
            if _reuse_ok and not df_queries.empty:
                _expected_q = len(df_queries)
                _manifest_q = int(
                    _candidate_report.get("unique_training_queries", 0)
                    or _candidate_report.get("eligible_training_queries", 0)
                    or 0
                )
                # Full runs must cover all training queries; smoke runs cover a bounded subset.
                if is_full and _manifest_q != _expected_q:
                    _reuse_ok = False
                    _reuse_reason = f"query-count mismatch (expected {_expected_q}, got {_manifest_q})"
            if _reuse_ok and final_pairs_file.exists():
                try:
                    _pair_rows = len(pd.read_parquet(final_pairs_file, columns=["query_id"]))
                    _manifest_pairs = int(_candidate_report.get("pair_count", _candidate_report.get("input_pair_count", 0)) or 0)
                    if _manifest_pairs and _pair_rows != _manifest_pairs:
                        _reuse_ok = False
                        _reuse_reason = f"pair-count mismatch (file={_pair_rows} manifest={_manifest_pairs})"
                except Exception:
                    pass
            if _reuse_ok:
                final_reranker_report = _candidate_report
                print(f"[+] Reusing existing final reranker model checkpoint from {final_reranker_dir} (identity-verified)")
                stage_timings.record("final_reranker_training", elapsed_seconds=0.001, cache_hit=True)
                reusing_final_reranker = True
            else:
                print(f"[-] Final adapter reuse rejected ({_reuse_reason}); retraining.")
                reusing_final_reranker = False
        except Exception as _e:
            print(f"[-] Final adapter reuse validation failed ({type(_e).__name__}); retraining.")
            reusing_final_reranker = False

    if not reusing_final_reranker:
        max_final_steps = effective_max_steps
        print(f"[*] Training Final Supervised LoRA Reranker on {reranker_device} (max_steps={max_final_steps})...")
        t_tr0 = time.perf_counter()
        final_reranker_report = train_reranker(
            pairs_file=final_pairs_file,
            config_path=resolved_reranker_config,
            output_dir=final_reranker_dir,
            fold=None,
            max_steps=max_final_steps,
            base_model_name="mock" if is_smoke else "BAAI/bge-reranker-v2-m3",
            device=reranker_device,
            precision=_prec_norm,
            num_workers=runtime_num_workers,
            enforce_full_coverage_steps=is_full,
        )
        final_training_time = max(0.001, time.perf_counter() - t_tr0)
        stage_timings.record("final_reranker_training", elapsed_seconds=final_training_time, cache_hit=False)
        print(f"[+] Final Reranker Training Status: {final_reranker_report.get('status')} | Checkpoint: {final_reranker_dir}")
        (final_reranker_dir / "complete.json").write_text(json.dumps(final_reranker_report, indent=2), encoding="utf-8")

    if is_full:
        min_coverage = 99.0
        uniq_cov = final_reranker_report.get("unique_query_coverage_pct", 0.0)
        pos_cov = final_reranker_report.get("positive_query_coverage_pct", 0.0)
        neg_cov = final_reranker_report.get("negative_query_coverage_pct", 0.0)
        actual_cov = final_reranker_report.get("actual_query_coverage_pct", 0.0)

        if uniq_cov < min_coverage:
            raise RuntimeError(
                f"FULL mode requires unique_query_coverage_pct >= {min_coverage}%, got {uniq_cov}% "
                f"({final_reranker_report.get('actual_unique_queries_seen')}/{final_reranker_report.get('eligible_training_queries')} queries seen)."
            )
        if pos_cov < min_coverage:
            raise RuntimeError(
                f"FULL mode requires positive_query_coverage_pct >= {min_coverage}%, got {pos_cov}% "
                f"({final_reranker_report.get('positive_queries_seen')}/{final_reranker_report.get('eligible_training_queries')} queries with positive seen)."
            )
        if neg_cov < min_coverage:
            raise RuntimeError(
                f"FULL mode requires negative_query_coverage_pct >= {min_coverage}%, got {neg_cov}% "
                f"({final_reranker_report.get('queries_with_negative_seen')}/{final_reranker_report.get('eligible_training_queries')} queries with negative seen)."
            )
        if actual_cov < min_coverage:
            raise RuntimeError(
                f"FULL mode requires actual_query_coverage_pct >= {min_coverage}%, got {actual_cov}%."
            )

    # 11. Public Test Inference with Fully Loaded Final Production Pipeline (P0.3, P0.4, P1.8)
    print("\n" + "=" * 70)
    print(f"[*] Executing Public Test Batch Inference with Final Production Pipeline ({len(public_data)} queries)...")
    print("=" * 70)

    # Load fully integrated pipeline
    fusion_model_load_path = (checkpoints_dir / "fusion_final") if use_learned_fusion else None
    reranker_adapter_load_path = final_reranker_dir if final_reranker_dir.exists() else None

    t_pipe0 = time.perf_counter()
    final_audit_json = working_path / "parameter_audit.json"
    pipeline = LegalIRPipeline.load_pipeline(
        data_dir=canonical_data_dir,
        index_dir=index_dir,
        reranker_adapter_path=reranker_adapter_load_path,
        fusion_model_path=fusion_model_load_path,
        use_reranker=True,
        use_learned_fusion=use_learned_fusion,
        dense_device=dense_device,
        reranker_device=reranker_device,
        strict_artifacts=strict_artifacts,
        audit_preflight=True,
        audit_output_json=final_audit_json,
        reranker_model_name="mock" if is_smoke else "BAAI/bge-reranker-v2-m3",
        reranker_batch_size=int(reranker_cfg.get("inference_batch_size") or reranker_cfg.get("batch_size", 16)),
        reranker_max_length=int(reranker_cfg.get("max_length", 384)),
        precision=_prec_norm,
        reranker_revision=(
            None
            if is_smoke
            else (
                final_reranker_report.get("base_model_revision")
                if "final_reranker_report" in locals() and isinstance(final_reranker_report, dict)
                else None
            )
        ),
    )
    final_audit_report = pipeline.audit_parameters(
        output_json=final_audit_json,
        raise_on_violation=True,
        require_loaded_models=(is_full or is_gpu_smoke),
    )
    pipe_load_audit_time = max(0.001, time.perf_counter() - t_pipe0)
    stage_timings.record("final_pipeline_load_audit", elapsed_seconds=pipe_load_audit_time, cache_hit=False)

    # Strict final PEFT adapter audit verification
    if is_full or is_gpu_smoke:
        reranker_audit = next((m for m in final_audit_report.get("models", {}).values() if m.get("role") == "cross_encoder_reranker"), {})
        if reranker_adapter_load_path is not None:
            if not reranker_audit.get("is_peft_lora", False):
                raise RuntimeError("Strict runtime audit failed: loaded reranker is not marked as is_peft_lora")
            if reranker_audit.get("adapter_parameters", 0) <= 0:
                raise RuntimeError(f"Strict runtime audit failed: reranker adapter_parameters must be > 0, got {reranker_audit.get('adapter_parameters')}")
            if final_audit_report.get("total_learned_parameters", 0) <= 702_754_049:
                raise RuntimeError("Strict runtime audit failed: total parameters did not increase after adapter loading")

    t0_infer = time.perf_counter()
    predictions: dict[str, dict[str, list[str]]] = {}
    q_items = list(public_data.items())
    if (is_smoke or is_gpu_smoke) and len(q_items) > 20:
        q_items = q_items[:20]

    # Precompute public query dense embeddings once on GPU0
    public_q_embs: dict[str, np.ndarray] = {}
    dense_ret = getattr(pipeline.hybrid_engine, "dense_retriever", None) or getattr(pipeline.hybrid_engine, "dense", None)
    if dense_ret is not None and len(q_items) > 0:
        print(f"[*] Precomputing public query dense embeddings for {len(q_items)} queries on {dense_device}...")
        try:
            q_texts = [
                (q_val.get("question", "") if isinstance(q_val, dict) else str(q_val))
                for _, q_val in q_items
            ]
            q_embs_array = dense_ret.encode_queries(
                q_texts, batch_size=64 if is_full else 16, stage_name="public_query"
            )
            for (qid, _), emb in zip(q_items, q_embs_array):
                public_q_embs[str(qid)] = emb
            print(f"[+] Precomputed {len(public_q_embs):,} public query embeddings for reuse.")
        except Exception as e:
            if is_full or is_gpu_smoke:
                raise RuntimeError(f"Failed to precompute public query embeddings on {dense_device} in {run_mode_str.upper()} mode: {e}") from e
            print(f"[-] Warning: public query embedding precomputation skipped: {e}")

    public_query_dict = {
        str(qid): (q_val.get("question", "") if isinstance(q_val, dict) else str(q_val))
        for qid, q_val in q_items
    }
    pipeline.candidate_k = 200 if is_full else 20
    pipeline.rerank_k = int(reranker_cfg.get("rerank_k", 100)) if is_full else 10

    predictions = pipeline.predict_batch(
        public_query_dict,
        query_embeddings=public_q_embs,
        show_progress=True,
    )

    public_inference_time = max(0.001, time.perf_counter() - t0_infer)
    stage_timings.record("public_inference", elapsed_seconds=public_inference_time, cache_hit=False)
    print(f"[+] Public inference completed in {public_inference_time:.2f}s ({len(predictions)} queries predicted).")
    rss_peak_mb = get_peak_process_rss_mb()

    # 12. Strict Submission Invariant Validation & Zip Packaging (P0.5, Invariants 1-24)
    t_pkg0 = time.perf_counter()
    sub_json = submissions_dir / "submission.json"
    sub_zip = submissions_dir / "submission.zip"
    package_submission(predictions, sub_json, sub_zip)

    expected_qids = official_public_qids if is_full else set(predictions.keys())
    if is_full:
        assert set(predictions.keys()) == expected_qids, f"Prediction keys mismatch with official public keys: missing {len(expected_qids - set(predictions.keys()))}, extra {len(set(predictions.keys()) - expected_qids)}"

    official_doc_ids = set(df_docs["doc_id"].astype(str)) if "doc_id" in df_docs.columns else None
    val_res = validate_submission(sub_json, expected_qids=expected_qids, corpus_doc_ids=official_doc_ids)
    zip_val_res = validate_submission_zip(sub_zip)

    is_submission_valid = bool(val_res.get("is_valid") and zip_val_res.get("is_valid"))
    pkg_time = max(0.001, time.perf_counter() - t_pkg0)
    stage_timings.record("submission_packaging", elapsed_seconds=pkg_time, cache_hit=False)
    print(f"[+] Submission Validation: JSON = {val_res.get('is_valid')} | ZIP = {zip_val_res.get('is_valid')} | Overall = {is_submission_valid}")
    if is_full and not is_submission_valid:
        raise RuntimeError(
            f"Final official submission failed validation: JSON errors={val_res.get('errors')} | ZIP errors={zip_val_res.get('errors')}"
        )

    # 13. Manifest, Hashes, Reports & Parameter Audits
    git_sha = get_git_commit(root_path)
    manifest_path = submissions_dir / "submission_manifest.json"
    manifest = create_submission_manifest(
        submission_json_path=sub_json,
        submission_zip_path=sub_zip,
        output_path=manifest_path,
        git_commit=git_sha,
        parameter_total=final_audit_report.get("total_learned_parameters", 0),
        model_names_and_revisions=[
            {"name": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", "role": "dense_embedding"},
            {"name": "BAAI/bge-reranker-v2-m3", "role": "cross_encoder_reranker"},
        ],
        metadata={
            "run_mode": run_mode_str,
            "submission_status": "SUBMITTABLE_OFFICIAL" if is_full else f"NON_SUBMITTABLE_{run_mode_str.upper()}",
            "cv_mean_recall@5": cv_report.get("mean_recall@5", 0.0),
            "cv_mean_precision@5": cv_report.get("mean_precision@5", 0.0),
            "candidate_recall@150": cv_report.get("mean_candidate@150", 0.0),
            "fusion_winning_method": winning_fusion_method,
            "devices": {"dense": dense_device, "reranker": reranker_device},
            "submission_valid": is_submission_valid,
            "stage_timings": stage_timings.to_dict(),
            "peak_rss_mb": rss_peak_mb,
        },
    )

    # 13b. Hardware Placement & Peak VRAM Verification (Section 3)
    dense_actual_dev = "cpu"
    if hasattr(pipeline.hybrid_engine, "dense") and pipeline.hybrid_engine.dense is not None:
        dense_m = getattr(pipeline.hybrid_engine.dense, "model", None)
        if dense_m is not None:
            try:
                dense_actual_dev = str(next(dense_m.parameters()).device)
            except Exception:
                dense_actual_dev = str(getattr(pipeline.hybrid_engine.dense, "device", "cpu"))

    reranker_actual_dev = "cpu"
    if pipeline.reranker is not None:
        reranker_m = getattr(pipeline.reranker, "model", None)
        if reranker_m is not None:
            try:
                reranker_actual_dev = str(next(reranker_m.parameters()).device)
            except Exception:
                reranker_actual_dev = str(getattr(pipeline.reranker, "device", "cpu"))

    if (is_gpu_smoke or is_full) and not allow_nonstandard_production_devices and torch.cuda.is_available():
        if not dense_actual_dev.startswith(dense_device):
            raise RuntimeError(f"Dense model device mismatch in {run_mode_str.upper()}: requested {dense_device}, actual {dense_actual_dev}")
        if not reranker_actual_dev.startswith(reranker_device):
            raise RuntimeError(f"Reranker model device mismatch in {run_mode_str.upper()}: requested {reranker_device}, actual {reranker_actual_dev}")

    if is_gpu_smoke or is_full:
        gpu0_alloc = torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 0
        gpu1_alloc = torch.cuda.max_memory_allocated(1) if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 0
        gpu0_res = torch.cuda.max_memory_reserved(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 0
        gpu1_res = torch.cuda.max_memory_reserved(1) if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 0

        oof_reranker_oom = sum(f.get("reranker_oom_events", 0) for f in cv_report.get("folds", []))
        final_reranker_oom = int(getattr(pipeline.reranker, "oom_events", 0)) if pipeline.reranker is not None else 0

        stage_telem = getattr(dense_retriever, "stage_telemetry", None)
        corpus_telem = stage_telem.get("corpus") if isinstance(stage_telem, dict) else None
        train_q_telem = stage_telem.get("train_query") if isinstance(stage_telem, dict) else None

        pipeline_dense = getattr(pipeline.hybrid_engine, "dense_retriever", None) or getattr(pipeline.hybrid_engine, "dense", None)
        pipe_stage_telem = getattr(pipeline_dense, "stage_telemetry", None)
        public_q_telem = pipe_stage_telem.get("public_query") if isinstance(pipe_stage_telem, dict) else None

        if corpus_telem is not None:
            dense_corpus_oom = int(getattr(corpus_telem, "oom_events", 0) or 0)
        else:
            dense_corpus_oom = int(getattr(dense_retriever, "dense_oom_events", 0) or 0)

        if train_q_telem is not None:
            dense_train_q_oom = int(getattr(train_q_telem, "oom_events", 0) or 0)
        else:
            dense_train_q_oom = 0

        if public_q_telem is not None:
            dense_public_q_oom = int(getattr(public_q_telem, "oom_events", 0) or 0)
        elif pipeline_dense is not None and pipeline_dense is not dense_retriever:
            dense_public_q_oom = int(getattr(pipeline_dense, "dense_oom_events", 0) or 0)
        else:
            dense_public_q_oom = 0

        dense_oom = dense_corpus_oom + dense_train_q_oom + dense_public_q_oom
        total_reranker_oom = int(oof_reranker_oom or 0) + int(final_reranker_oom or 0)
        total_oom_events = total_reranker_oom + dense_oom

        min_batch = int(getattr(pipeline.reranker, "min_successful_batch_size", 16) or 16) if pipeline.reranker is not None else 16
        reranker_audit_entry = next((m for m in final_audit_report.get("models", {}).values() if m.get("role") == "cross_encoder_reranker"), {})

        dataset_identity_payload = {
            "dataset": official_identity_report.get("dataset", OFFICIAL_DATASET_NAME),
            "version": official_identity_report.get("version", OFFICIAL_VERSION),
            "schema": official_identity_report.get("schema", OFFICIAL_SCHEMA),
            "documents": int(official_identity_report.get("documents", OFFICIAL_DOCUMENT_COUNT)),
            "chunks": int(official_identity_report.get("chunks", OFFICIAL_TOTAL_CHUNKS)),
            "micro_chunks": int(official_identity_report.get("micro_chunks", OFFICIAL_MICRO_CHUNKS)),
            "macro_chunks": int(official_identity_report.get("macro_chunks", OFFICIAL_MACRO_CHUNKS)),
            "train_queries": int(official_identity_report.get("train_queries", OFFICIAL_TRAIN_QUERIES)),
            "qrels": int(official_identity_report.get("qrels", OFFICIAL_QRELS)),
            "public_queries": int(official_identity_report.get("public_queries", len(public_data))),
            "audit_valid": bool(official_identity_report.get("audit_valid", True)),
            "audit_errors": list(official_identity_report.get("audit_errors", [])),
        }

        gpu_smoke_report = {
            "dataset_identity": dataset_identity_payload,
            "split_provenance": split_provenance_report,
            "duplicate_groups": {
                "source": dup_groups_source,
                "group_count": dup_report.get("group_count", 0),
                "total_duplicate_docs": dup_report.get("total_duplicate_docs", 0),
                "is_valid": dup_report.get("is_valid", False),
            },
            "max_pair_validation_leakage_count": cv_report.get("max_pair_validation_leakage_count", 0),
            "max_pair_unknown_train_count": cv_report.get("max_pair_unknown_train_count", 0),
            "dense_requested": dense_device,
            "dense_actual": dense_actual_dev,
            "reranker_requested": reranker_device,
            "reranker_actual": reranker_actual_dev,
            "dense_search_backend": "faiss_index_flat_ip" if getattr(dense_retriever, "_faiss_index", None) is not None else "numpy",
            "gpu0_peak_allocated_bytes": int(gpu0_alloc),
            "gpu1_peak_allocated_bytes": int(gpu1_alloc),
            "gpu0_peak_reserved_bytes": int(gpu0_res),
            "gpu1_peak_reserved_bytes": int(gpu1_res),
            "dense_corpus": {
                "requested_batch_size": int(getattr(corpus_telem, "requested_batch_size", getattr(dense_retriever, "dense_initial_batch_size", 32)) or 32),
                "min_successful_batch_size": int(getattr(corpus_telem, "min_successful_batch_size", getattr(dense_retriever, "dense_min_successful_batch_size", 32)) or 32),
                "oom_events": dense_corpus_oom,
                "item_count": int(getattr(corpus_telem, "item_count", len(df_chunks)) or len(df_chunks)),
                "elapsed_seconds": round(float(getattr(corpus_telem, "elapsed_seconds", dense_build_time) or dense_build_time), 2),
            },
            "dense_train_query": {
                "requested_batch_size": int(getattr(train_q_telem, "requested_batch_size", 128) or 128),
                "min_successful_batch_size": int(getattr(train_q_telem, "min_successful_batch_size", 128) or 128),
                "oom_events": dense_train_q_oom,
                "item_count": int(getattr(train_q_telem, "item_count", len(df_queries)) or len(df_queries)),
                "elapsed_seconds": round(float(getattr(train_q_telem, "elapsed_seconds", train_query_enc_time) or train_query_enc_time), 2),
            },
            "dense_public_query": {
                "requested_batch_size": int(getattr(public_q_telem, "requested_batch_size", 32) or 32),
                "min_successful_batch_size": int(getattr(public_q_telem, "min_successful_batch_size", 32) or 32),
                "oom_events": dense_public_q_oom,
                "item_count": int(getattr(public_q_telem, "item_count", len(public_data)) or len(public_data)),
                "elapsed_seconds": round(float(getattr(public_q_telem, "elapsed_seconds", public_inference_time) or public_inference_time), 2),
            },
            "dense_total_oom_events": dense_oom,
            "dense_oom_events": dense_oom,
            "oof_reranker_oom_events": oof_reranker_oom,
            "final_reranker_oom_events": final_reranker_oom,
            "total_reranker_oom_events": total_reranker_oom,
            "total_oom_events": total_oom_events,
            "oom": bool(total_oom_events > 0),
            "stable_reranker_batch_size": min_batch,
            "optimizer_steps": int(final_reranker_report.get("optimizer_steps", 0)),
            "param_diff": float(final_reranker_report.get("param_diff", 0.0) or 0.0),
            "actual_unique_queries_seen": int(final_reranker_report.get("actual_unique_queries_seen", 0)),
            "actual_query_coverage_pct": float(final_reranker_report.get("actual_query_coverage_pct", 0.0)),
            "adapter_checksum": str(final_reranker_report.get("adapter_checksum", "")),
            "adapter_parameters": int(reranker_audit_entry.get("adapter_parameters", 0)),
            "is_peft_lora": bool(reranker_audit_entry.get("is_peft_lora", False)),
            "host_rss_mb": {
                "after_canonical_load": rss_after_load_mb,
                "after_dense_index": rss_after_dense_mb,
                "after_oof": rss_after_oof_mb,
                "peak": rss_peak_mb,
            },
            "stage_timings": stage_timings.to_dict(),
            "strict_artifacts": bool(strict_artifacts),
            "fusion_crossfit_folds": int(oof_num_folds),
        }
        report_path = working_path / "gpu_smoke_report.json"
        report_path.write_text(json.dumps(gpu_smoke_report, indent=2), encoding="utf-8")
        print(f"[+] Saved GPU smoke hardware report to {report_path}")

        # Runtime projection calculation
        q_infer_rate = len(q_items) / max(0.001, public_inference_time)
        total_public_count = float(len(public_data))
        projected_public_infer_sec = total_public_count / max(0.1, q_infer_rate)

        # OOF pure inference throughput
        heldout_qps = float(cv_report.get("heldout_inference_queries_per_second", cv_report.get("queries_per_second", 1.0)))
        total_oof_queries = float(len(df_queries)) if not df_queries.empty else 7000.0
        projected_oof_inference_sec = total_oof_queries / max(0.1, heldout_qps)

        # Training rate per optimizer step
        measured_final_steps = int(final_reranker_report.get("optimizer_steps", final_reranker_report.get("global_steps", 1)))
        final_training_sec = float(final_reranker_report.get("training_time_sec", 10.0))
        sec_per_final_step = final_training_sec / max(1, measured_final_steps)

        fold_opt_steps_total = int(cv_report.get("reranker_optimizer_steps_total", 0))
        fold_train_sec_total = float(cv_report.get("reranker_training_seconds_total", 0.0))
        sec_per_fold_step = (fold_train_sec_total / max(1, fold_opt_steps_total)) if fold_opt_steps_total > 0 else sec_per_final_step

        sec_per_train_step = sec_per_final_step if sec_per_final_step > 0 else (sec_per_fold_step if sec_per_fold_step > 0 else 0.5)

        # Production step counts (split-specific coverage requirements)
        # 1. Final training on all 7,000 queries:
        full_final_steps = int(effective_max_steps if is_full else 875)

        # 2. Split-specific 5-fold cross-validation steps
        splits_json_path = canonical_data_dir / "splits/random_5fold.json"
        production_fold_step_list: list[int] = []
        if splits_json_path.exists():
            try:
                splits_info = json.loads(splits_json_path.read_text(encoding="utf-8"))
                for f_info in splits_info.get("folds", []):
                    train_qids_fold = f_info.get("train_query_ids", [])
                    q_count = len(train_qids_fold) if train_qids_fold else int(7000 * 4 / 5)
                    st = compute_coverage_required_steps(
                        eligible_query_count=q_count,
                        batch_size=int(reranker_cfg.get("batch_size", 2)),
                        gradient_accumulation_steps=int(reranker_cfg.get("gradient_accumulation_steps", 8)),
                        target_coverage_pct=1.0,
                        require_pos_and_neg=True,
                    )
                    production_fold_step_list.append(st)
            except Exception:
                pass

        if not production_fold_step_list:
            default_fold_q_count = int((len(df_queries) if not df_queries.empty else 7000) * 4 / 5)
            default_fold_step = compute_coverage_required_steps(
                eligible_query_count=default_fold_q_count,
                batch_size=int(reranker_cfg.get("batch_size", 2)),
                gradient_accumulation_steps=int(reranker_cfg.get("gradient_accumulation_steps", 8)),
                target_coverage_pct=1.0,
                require_pos_and_neg=True,
            )
            production_fold_step_list = [default_fold_step] * 5

        if is_full:
            production_fold_step_list = [max(st, effective_max_steps) for st in production_fold_step_list]

        avg_fold_steps = int(round(float(np.mean(production_fold_step_list)))) if production_fold_step_list else 700

        # 3. Document-disjoint split-specific steps
        dj_splits_path = canonical_data_dir / "splits/doc_disjoint_split.json"
        dj_q_count = int((len(df_queries) if not df_queries.empty else 7000) * 4 / 5)
        if dj_splits_path.exists():
            try:
                dj_info = json.loads(dj_splits_path.read_text(encoding="utf-8"))
                dj_train_qids = dj_info.get("train_query_ids", [])
                if dj_train_qids:
                    dj_q_count = len(dj_train_qids)
            except Exception:
                pass

        projected_doc_disjoint_steps = compute_coverage_required_steps(
            eligible_query_count=dj_q_count,
            batch_size=int(reranker_cfg.get("batch_size", 2)),
            gradient_accumulation_steps=int(reranker_cfg.get("gradient_accumulation_steps", 8)),
            target_coverage_pct=1.0,
            require_pos_and_neg=True,
        )
        if is_full:
            projected_doc_disjoint_steps = max(projected_doc_disjoint_steps, effective_max_steps)

        # Scale pair mining runtime from measured queries to full fold / doc disjoint / final counts (Task 4)
        is_smoke_run = is_smoke or is_gpu_smoke
        fold_measured_queries = 20 if is_smoke_run else default_fold_q_count
        avg_measured_fold_pm_sec = float(cv_report.get("pair_mining_seconds_total", 5.0)) / max(1, len(cv_report.get("folds", [])))
        setup_fold_pm = min(1.0, avg_measured_fold_pm_sec * 0.2)
        loop_fold_pm = max(0.001, avg_measured_fold_pm_sec - setup_fold_pm)
        projected_single_fold_pm_sec = project_pair_mining_seconds(
            setup_seconds=setup_fold_pm,
            query_loop_seconds=loop_fold_pm,
            measured_queries=fold_measured_queries,
            target_queries=default_fold_q_count,
            safety_factor=1.10,
        )

        projected_final_training_sec = full_final_steps * sec_per_train_step
        projected_5fold_training_sec = sum(st * sec_per_train_step + projected_single_fold_pm_sec for st in production_fold_step_list)
        projected_5fold_oof_sec = projected_oof_inference_sec + projected_5fold_training_sec

        # Document disjoint projection with scaled pair mining
        dj_report = doc_disjoint_report
        dj_measured_queries = 20 if is_smoke_run else dj_q_count
        raw_dj_mining_sec = float(dj_report.get("doc_disjoint_pair_mining_seconds", 5.0))
        setup_dj_pm = min(1.0, raw_dj_mining_sec * 0.2)
        loop_dj_pm = max(0.001, raw_dj_mining_sec - setup_dj_pm)
        projected_doc_disjoint_pm_sec = project_pair_mining_seconds(
            setup_seconds=setup_dj_pm,
            query_loop_seconds=loop_dj_pm,
            measured_queries=dj_measured_queries,
            target_queries=dj_q_count,
            safety_factor=1.10,
        )
        dj_training_sec = projected_doc_disjoint_steps * sec_per_train_step
        dj_infer_sec = float(dj_report.get("doc_disjoint_inference_seconds", 5.0))
        projected_doc_disjoint_sec = projected_doc_disjoint_pm_sec + dj_training_sec + dj_infer_sec

        # Final pair mining projection (scaled to 7,000 queries)
        final_measured_queries = 100 if is_smoke_run else 7000
        setup_final_pm = min(1.0, final_pair_mining_time * 0.2)
        loop_final_pm = max(0.001, final_pair_mining_time - setup_final_pm)
        projected_final_pair_mining_sec = project_pair_mining_seconds(
            setup_seconds=setup_final_pm,
            query_loop_seconds=loop_final_pm,
            measured_queries=final_measured_queries,
            target_queries=7000,
            safety_factor=1.10,
        )

        # Component-complete cold-start projection map (Task 5)
        projection_components = {
            "canonical_load": round(canonical_load_time, 2),
            "bm25_legal": round(bm25_legal_time, 2),
            "bm25_pyvi": round(bm25_pyvi_time, 2),
            "dense_index": round(dense_build_time, 2),
            "train_query_encoding": round(train_query_enc_time, 2),
            "projected_oof": round(projected_5fold_oof_sec, 2),
            "fusion_training": round(fusion_time, 2),
            "projected_doc_disjoint": round(projected_doc_disjoint_sec, 2),
            "question_memory": round(qm_time, 2),
            "projected_final_pair_mining": round(projected_final_pair_mining_sec, 2),
            "projected_final_reranker": round(projected_final_training_sec, 2),
            "final_pipeline_load_audit": round(pipe_load_audit_time, 2),
            "projected_public_inference": round(projected_public_infer_sec, 2),
            "submission_packaging": round(pkg_time, 2),
            "safety_overhead": 60.0,
        }
        cold_start_total_sec = round(sum(projection_components.values()), 2)
        warm_cache_total_sec = round(
            projected_5fold_oof_sec
            + fusion_time
            + projected_doc_disjoint_sec
            + qm_time
            + projected_final_pair_mining_sec
            + projected_final_training_sec
            + pipe_load_audit_time
            + projected_public_infer_sec
            + pkg_time
            + 60.0,
            2,
        )

        KAGGLE_MAX_SECONDS = 12 * 3600.0  # 43,200s
        SAFETY_FACTOR = 0.90
        PRODUCTION_RUNTIME_BUDGET = KAGGLE_MAX_SECONDS * SAFETY_FACTOR  # 38,880s (~10.8 hours)

        runtime_proj = {
            "public_queries_per_second": round(q_infer_rate, 2),
            "oof_pure_inference_qps": round(heldout_qps, 2),
            "sec_per_optimizer_step": round(sec_per_train_step, 4),
            "measured_gpu_smoke_fold_steps": fold_opt_steps_total // max(1, len(cv_report.get("folds", []))),
            "projected_full_fold_steps": avg_fold_steps,
            "projected_fold_steps_by_fold": production_fold_step_list,
            "projected_doc_disjoint_steps": projected_doc_disjoint_steps,
            "measured_final_steps": measured_final_steps,
            "projected_final_steps": full_final_steps,
            "projected_num_folds": 5,
            "total_oof_validation_queries": int(total_oof_queries),
            "public_queries": int(total_public_count),
            "includes_doc_disjoint": True,
            "includes_dense_build": True,
            "dense_corpus_build_seconds": round(dense_build_time, 2),
            "train_query_encoding_seconds": round(train_query_enc_time, 2),
            "projected_oof_inference_seconds": round(projected_oof_inference_sec, 2),
            "projected_5fold_training_seconds": round(projected_5fold_training_sec, 2),
            "projected_5fold_oof_seconds": round(projected_5fold_oof_sec, 2),
            "projected_5fold_oof_hours": round(projected_5fold_oof_sec / 3600.0, 3),
            "projected_final_training_seconds": round(projected_final_training_sec, 2),
            "projected_doc_disjoint_seconds": round(projected_doc_disjoint_sec, 2),
            "projected_public_inference_seconds": round(projected_public_infer_sec, 2),
            "projected_final_pair_mining_seconds": round(projected_final_pair_mining_sec, 2),
            "projected_oof_pair_mining_seconds": round(projected_single_fold_pm_sec * 5, 2),
            "projected_doc_disjoint_pair_mining_seconds": round(projected_doc_disjoint_pm_sec, 2),
            "projection_components_seconds": projection_components,
            "cold_start_total_seconds": round(cold_start_total_sec, 2),
            "cold_start_total_hours": round(cold_start_total_sec / 3600.0, 3),
            "warm_cache_total_seconds": round(warm_cache_total_sec, 2),
            "warm_cache_total_hours": round(warm_cache_total_sec / 3600.0, 3),
            "production_runtime_budget_seconds": round(PRODUCTION_RUNTIME_BUDGET, 2),
            "production_runtime_budget_hours": round(PRODUCTION_RUNTIME_BUDGET / 3600.0, 2),
            "fits_kaggle_session_limit": bool(cold_start_total_sec < PRODUCTION_RUNTIME_BUDGET),
        }
        proj_path = working_path / "runtime_projection.json"
        proj_path.write_text(json.dumps(runtime_proj, indent=2), encoding="utf-8")
        print(f"[+] Saved runtime projection to {proj_path}")

    # Copy files to /kaggle/working root if running in Kaggle environment and in full mode AFTER manifest creation
    if is_full and Path("/kaggle/working").exists() and is_submission_valid:
        try:
            import shutil
            shutil.copy2(sub_json, Path("/kaggle/working/submission.json"))
            shutil.copy2(sub_zip, Path("/kaggle/working/submission.zip"))
            shutil.copy2(manifest_path, Path("/kaggle/working/submission_manifest.json"))
            print("[+] Copied submission.json, submission.zip, and submission_manifest.json to /kaggle/working/ root.")
        except Exception as e:
            print(f"[-] Warning: Failed to copy submission to /kaggle/working: {e}")

    # Save ablation row
    ablation_csv = working_path / "ablation_report.csv"
    ablation_df = pd.DataFrame([
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_mode": run_mode_str,
            "mean_recall@5": cv_report.get("mean_recall@5", 0.0),
            "mean_precision@5": cv_report.get("mean_precision@5", 0.0),
            "mean_mrr": cv_report.get("mean_mrr", 0.0),
            "mean_ndcg@5": cv_report.get("mean_ndcg@5", 0.0),
            "candidate_recall@50": cv_report.get("mean_candidate@50", 0.0),
            "candidate_recall@150": cv_report.get("mean_candidate@150", 0.0),
            "candidate_recall@200": cv_report.get("mean_candidate@200", 0.0),
            "runtime_per_query_ms": cv_report.get("runtime_per_query_ms", 0.0),
            "fusion_winner": winning_fusion_method,
            "total_learned_parameters": final_audit_report.get("total_learned_parameters", 0),
            "public_predictions_count": len(predictions),
            "git_commit": git_sha,
        }
    ])
    ablation_df.to_csv(ablation_csv, index=False)
    print(f"[+] Saved ablation report to {ablation_csv}")

    fusion_winner_rec5 = fusion_report.get("winner_mean_recall@5", cv_report.get("mean_recall@5", 0.0))
    fusion_winner_prec5 = fusion_report.get("comparison", {}).get("winner_mean_precision@5", cv_report.get("mean_precision@5", 0.0))
    doc_disjoint_rec5 = (
        doc_disjoint_report.get("trained_reranker_system", {}).get("recall@5", 0.0)
        if doc_disjoint_report
        else 0.0
    )

    total_runtime = time.time() - t_start
    supervisor_end_utc = datetime.now(timezone.utc).isoformat()

    # 14. Attempt-bound acceptance receipt (FULL only; fix.md section 12).
    # Built from this attempt's own artifacts and verified immediately; a
    # builder failure must never mask training results, so it degrades to an
    # INCOMPLETE receipt instead of raising.
    acceptance_verdict = "SKIPPED_NON_FULL"
    acceptance_receipt_path = None
    acceptance_reasons: list[str] = []
    if is_full:
        from src.release.acceptance import not_run_receipt as _not_run

        try:
            _qrels_dict: dict[str, list[str]] = defaultdict(list)
            for _r in df_qrels.to_dict("records"):
                _qrels_dict[str(_r["query_id"])].append(str(_r["doc_id"]))
            _splits = json.loads(Path(split_res.random_5fold_path).read_text(encoding="utf-8"))
            _corpus_ids = set(df_docs["doc_id"].astype(str)) if "doc_id" in df_docs.columns else set()
            _final_rev = final_reranker_report.get("base_model_revision")
            _receipt, _verification = build_attempt_acceptance_receipt(
                working_path=working_path,
                cv_report=cv_report,
                fusion_report=fusion_report,
                doc_disjoint_report=doc_disjoint_report,
                predictions=predictions,
                qrels_dict=dict(_qrels_dict),
                splits=_splits,
                corpus_doc_ids=_corpus_ids,
                final_reranker_dir=final_reranker_dir,
                final_reranker_report=final_reranker_report,
                reranker_model="BAAI/bge-reranker-v2-m3",
                reranker_revision=_final_rev,
                git_sha=git_sha,
                backend=str(resolved_backend),
                supervisor_start_utc=supervisor_start_utc,
                supervisor_end_utc=supervisor_end_utc,
                elapsed_seconds=total_runtime,
                submission_valid=is_submission_valid,
            )
            acceptance_receipt_path = working_path / "acceptance_receipt.json"
            acceptance_receipt_path.write_text(
                json.dumps({"receipt": _receipt, "verification": _verification}, indent=2),
                encoding="utf-8",
            )
            acceptance_verdict = str(_verification.get("verdict", "INCOMPLETE"))
            acceptance_reasons = list(_verification.get("reasons", []))
            print(f"[+] Attempt acceptance receipt: {acceptance_receipt_path} verdict={acceptance_verdict}")
        except Exception as _acc_exc:
            _fallback = _not_run(attempt_id=working_path.name)
            _fallback["reasons"] = [f"receipt builder failed: {type(_acc_exc).__name__}"]
            acceptance_receipt_path = working_path / "acceptance_receipt.json"
            try:
                acceptance_receipt_path.write_text(
                    json.dumps({"receipt": _fallback, "verification": {"verdict": "INCOMPLETE"}}, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                acceptance_receipt_path = None
            acceptance_verdict = "INCOMPLETE"
            acceptance_reasons = list(_fallback["reasons"])
            print(f"[!] Attempt acceptance receipt INCOMPLETE ({type(_acc_exc).__name__}); training results preserved.")

    print("\n" + "=" * 80)
    print(f"LEGALIR TASK 1 PIPELINE RUN COMPLETE in {total_runtime:.2f}s")
    print(f"  - Validation Status                      : {'PASS' if is_submission_valid else 'FAIL'}")
    print(f"  - Submission JSON                        : {sub_json}")
    print(f"  - Submission ZIP                         : {sub_zip} ({sub_zip.stat().st_size:,} bytes)")
    print(f"  - Manifest JSON                          : {manifest_path}")
    print(f"  - Reranker OOF Recall@5                  : {cv_report.get('mean_recall@5', 0.0) * 100:.4f}%")
    print(f"  - Fusion Winner                          : {winning_fusion_method}")
    print(f"  - Fusion Winner Cross-Fitted Recall@5    : {fusion_winner_rec5 * 100:.4f}%")
    print(f"  - Fusion Winner Precision@5              : {fusion_winner_prec5 * 100:.4f}%")
    print(f"  - Candidate Recall@150                   : {cv_report.get('mean_candidate@150', 0.0) * 100:.4f}%")
    if doc_disjoint_rec5 > 0:
        print(f"  - Doc-Disjoint Trained Reranker Recall@5 : {doc_disjoint_rec5 * 100:.4f}%")
    print(f"  - Parameter Utilization                  : {final_audit_report.get('total_learned_parameters', 0):,} / 4,000,000,000 ({final_audit_report.get('budget_utilization_pct', 0.0):.2f}%)")
    print(f"  - Submission Status                      : {'SUBMITTABLE_OFFICIAL' if is_full else f'NON_SUBMITTABLE_{run_mode_str.upper()}'}")
    print("=" * 80)

    return KaggleRunResult(
        is_valid=is_submission_valid,
        submission_path=sub_json,
        submission_zip_path=sub_zip,
        manifest_path=manifest_path,
        audit_report=final_audit_report,
        cv_report=cv_report,
        fusion_report=fusion_report,
        public_predictions_count=len(predictions),
        run_mode=run_mode_str,
        execution_time_seconds=total_runtime,
        artifacts_dir=working_path,
        metadata={
            "git_commit": git_sha,
            "dense_device": dense_device,
            "reranker_device": reranker_device,
            "fusion_winner": winning_fusion_method,
            "supervisor_start_utc": supervisor_start_utc,
            "supervisor_end_utc": supervisor_end_utc,
            "acceptance_verdict": acceptance_verdict,
            "acceptance_receipt_path": str(acceptance_receipt_path) if acceptance_receipt_path else None,
            "acceptance_reasons": acceptance_reasons,
        },
    )

"""Out-of-Fold (OOF) 5-fold cross-validation runner for LegalIR Task 1.

Provides leakage-safe cross-validation orchestration, feature generation,
official Codabench scorer parity, and full metrics reporting.
"""

from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.core.paths import ProjectPaths
from src.evaluation.codabench_compat import assert_official_equivalence
from src.evaluation.evaluator import (
    DEFAULT_CANDIDATE_CUTOFFS,
    FINAL_RANKING_METRICS,
    compute_candidate_cutoffs,
    compute_candidate_recall,
    evaluate_predictions,
    normalize_candidate_cutoffs,
)
from src.models.parameter_audit import audit_system_parameters, MAX_PARAMETER_BUDGET
from src.evaluation.splits import (
    generate_document_disjoint_split,
    generate_random_5fold_split,
    verify_document_disjoint_isolation,
    verify_fold_isolation,
)
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.oof_features import compute_training_doc_frequencies, extract_candidate_features
from src.ranking.reranker import CrossEncoderReranker
from src.ranking.selector import TopKSelector
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import TrainQuestionMemory
from src.retrieval.types import CandidateRecord
from src.ranking.fusion import ReciprocalRankFusion


# Fixed predeclared fusion policy shared by OOF folds, the document-disjoint
# trained pass, and public inference (predict.py default). Stateless; the
# confirmatory score must evaluate one identical scoring policy everywhere.
_OOF_FIXED_RRF = ReciprocalRankFusion()


def _sha256_sorted_ids(ids) -> str:
    """Stable SHA-256 over sorted string IDs for resume-identity checks."""
    joined = ",".join(sorted(str(x) for x in ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _fold_expected_identity(
    fold_idx: int,
    fold_info: dict[str, Any],
    *,
    smoke: bool,
    smoke_sample_size: int,
    reranker_model: str,
    candidate_k: int,
    rerank_k: int,
    precision: str,
    split_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_train = [str(x) for x in fold_info.get("train_query_ids", fold_info.get("train", []))]
    raw_val = [str(x) for x in fold_info.get("val_query_ids", fold_info.get("val", []))]
    if smoke:
        raw_val = raw_val[: int(smoke_sample_size)]
    identity: dict[str, Any] = {
        "fold": int(fold_idx),
        "train_count": len(raw_train),
        "val_count": len(raw_val),
        "train_ids_sha256": _sha256_sorted_ids(raw_train),
        "val_ids_sha256": _sha256_sorted_ids(raw_val),
        "reranker_model": str(reranker_model),
        "candidate_k": int(candidate_k),
        "rerank_k": int(rerank_k),
        "precision": str(precision),
        "smoke": bool(smoke),
    }
    try:
        if isinstance(split_provenance, dict):
            rnd = (split_provenance.get("random_5fold") or {})
            if isinstance(rnd, dict) and rnd.get("sha256"):
                identity["split_random_5fold_sha256"] = str(rnd.get("sha256"))
    except Exception:
        pass
    return identity


class OOFRunner:
    """Orchestrates leakage-safe 5-fold OOF cross-validation and feature extraction."""

    def __init__(
        self,
        data_dir: str | Path = "artifacts/task1/data",
        index_dir: str | Path = "artifacts/task1/indexes",
        splits_path: str | Path | None = None,
        output_dir: str | Path = "artifacts/local/cv",
        config_path: str | Path | None = "configs/pipeline.yaml",
        num_folds: int = 5,
        candidate_k: int = 150,
        rerank_k: int = 50,
        use_reranker: bool = False,
        reranker_model: str = "mock",
        train_reranker_per_fold: bool = False,
        dense_device: str | None = None,
        reranker_device: str | None = None,
        device: str | None = None,
        smoke: bool = False,
        smoke_sample_size: int = 20,
        doc_disjoint: bool = False,
        doc_disjoint_splits_path: str | Path | None = None,
        train_query_embeddings: Any | None = None,
        reranker_config_path: str | Path | None = None,
        duplicate_groups_path: str | Path | None = None,
        split_provenance: dict[str, Any] | None = None,
        precision: str | None = None,
        num_workers: int | None = None,
        reranker_batch_size: int | None = None,
        reranker_max_length: int | None = None,
        allow_stage_reuse: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.duplicate_groups_path = Path(duplicate_groups_path) if duplicate_groups_path else None
        self.split_provenance = split_provenance or {}
        if splits_path is not None and Path(splits_path).exists():
            self.splits_path = Path(splits_path)
        elif (self.data_dir / "splits" / "random_5fold.json").exists():
            self.splits_path = self.data_dir / "splits" / "random_5fold.json"
        elif splits_path is not None:
            self.splits_path = Path(splits_path)
        else:
            self.splits_path = Path("artifacts/shared/canonical/v2/splits/random_5fold.json")

        self.output_dir = Path(output_dir)
        self.config_path = Path(config_path) if config_path else None
        self.reranker_config_path = Path(reranker_config_path) if reranker_config_path else self.config_path
        _prec = str(precision or "bf16").lower().strip()
        if _prec == "bfloat16":
            _prec = "bf16"
        self.precision = _prec if _prec in ("bf16", "fp16", "fp32") else "bf16"
        self.num_folds = int(num_folds)
        self.candidate_k = int(candidate_k)
        self.rerank_k = int(rerank_k)
        self.use_reranker = bool(use_reranker)
        self.reranker_model = str(reranker_model)
        self.train_reranker_per_fold = bool(train_reranker_per_fold)
        self.dense_device = dense_device if dense_device is not None else device
        self.reranker_device = reranker_device if reranker_device is not None else device
        self.device = self.reranker_device
        self.num_workers = None if num_workers is None else max(0, int(num_workers))

        # Resolve inference batch size and max length from config/overrides
        rcfg: dict[str, Any] = {}
        if self.reranker_config_path and Path(self.reranker_config_path).is_file():
            try:
                import yaml
                rcfg = yaml.safe_load(Path(self.reranker_config_path).read_text(encoding="utf-8")) or {}
            except Exception:
                rcfg = {}
        self.reranker_batch_size = int(reranker_batch_size or rcfg.get("inference_batch_size") or rcfg.get("batch_size") or 16)
        self.reranker_max_length = int(reranker_max_length or rcfg.get("max_length") or 384)
        self._shared_memory_dense_encoder: DenseMacroRetriever | None = None
        self.smoke = bool(smoke)
        self.smoke_sample_size = int(smoke_sample_size)
        # F4: completed-stage reuse is DISABLED by default. The F4 identity/
        # completeness contract is incomplete, so FULL runs must recompute.
        # Explicit opt-in only for bounded development; never cross-attempt resume
        # (Modal uses a fresh UUID attempt dir per invocation).
        self.allow_stage_reuse = bool(allow_stage_reuse)
        self.doc_disjoint = bool(doc_disjoint)

        if doc_disjoint_splits_path is not None and Path(doc_disjoint_splits_path).exists():
            self.doc_disjoint_splits_path = Path(doc_disjoint_splits_path)
        elif (self.data_dir / "splits" / "doc_disjoint_split.json").exists():
            self.doc_disjoint_splits_path = self.data_dir / "splits" / "doc_disjoint_split.json"
        elif doc_disjoint_splits_path is not None:
            self.doc_disjoint_splits_path = Path(doc_disjoint_splits_path)
        else:
            self.doc_disjoint_splits_path = Path("artifacts/shared/canonical/v2/splits/doc_disjoint_split.json")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded attributes
        self.docs_df: pd.DataFrame | None = None
        self.queries_df: pd.DataFrame | None = None
        self.qrels_df: pd.DataFrame | None = None
        self.doc_map: dict[str, dict[str, Any]] = {}
        self.queries_map: dict[str, str] = {}
        self.qrels_map: dict[str, list[str]] = defaultdict(list)
        self.train_query_embeddings: dict[str, np.ndarray] = (
            {str(k): np.asarray(v) for k, v in train_query_embeddings.items()}
            if train_query_embeddings is not None
            else {}
        )
        self.evidence_builder: EvidencePackBuilder | None = None
        self.bm25: BM25MicroRetriever | None = None
        self.bm25_pyvi: BM25PyViRetriever | None = None
        self.dense: DenseMacroRetriever | None = None
        self.exact: ExactMatcher | None = None
        self.selector = TopKSelector(max_k=5)
        self.doc_disjoint_report: dict[str, Any] = {}
        self._static_branch_cache: dict[str, Any] = {}

    def load_data(self) -> None:
        """Load canonical dataset tables and build lookups."""
        if self.docs_df is not None:
            return

        docs_path = self.data_dir / "documents.parquet"
        queries_path = self.data_dir / "queries_train.parquet"
        qrels_path = self.data_dir / "qrels_train.parquet"

        if not (docs_path.exists() and queries_path.exists() and qrels_path.exists()):
            raise FileNotFoundError(f"Canonical data files missing in {self.data_dir}")

        self.docs_df = pd.read_parquet(docs_path)
        self.queries_df = pd.read_parquet(queries_path)
        self.qrels_df = pd.read_parquet(qrels_path)

        for r in self.docs_df.to_dict("records"):
            self.doc_map[str(r["doc_id"])] = r

        for r in self.queries_df.to_dict("records"):
            # Prefer normalized question text, fallback to raw
            q_text = r.get("question_norm") or r.get("question_raw") or r.get("question") or ""
            self.queries_map[str(r["query_id"])] = str(q_text)

        self.qrels_map = defaultdict(list)
        for r in self.qrels_df.to_dict("records"):
            self.qrels_map[str(r["query_id"])].append(str(r["doc_id"]))

        # Build evidence pack builder if chunks exist
        chunks_path = self.data_dir / "chunks.parquet"
        if chunks_path.exists():
            self.evidence_builder = EvidencePackBuilder(
                chunks_path=chunks_path,
                doc_metadata=self.doc_map,
                max_chunks=3,
            )

    def load_retrievers(self) -> None:
        """Load or initialize retrieval components."""
        if self.exact is None:
            chunks_path = self.data_dir / "chunks.parquet"
            self.exact = ExactMatcher(
                documents=list(self.doc_map.values()),
                chunks=chunks_path if chunks_path.exists() else None,
            )

        if self.bm25 is None:
            bm25_index_dir = self.index_dir / "bm25"
            bm25_file = bm25_index_dir / "bm25_micro_index.pkl"
            if bm25_file.exists():
                self.bm25 = BM25MicroRetriever.load(bm25_file)
            elif bm25_index_dir.exists() and list(bm25_index_dir.glob("*.pkl")):
                self.bm25 = BM25MicroRetriever.load(bm25_index_dir)
            else:
                # Fit on micro chunks
                chunks_path = self.data_dir / "chunks.parquet"
                if chunks_path.exists():
                    chunks_df = pd.read_parquet(chunks_path)
                    micro_chunks = chunks_df[chunks_df["granularity"] == "micro"] if "granularity" in chunks_df.columns else chunks_df
                    if self.docs_df is not None:
                        from src.retrieval.build_indexes import enrich_chunks_with_doc_metadata
                        micro_chunks = enrich_chunks_with_doc_metadata(micro_chunks, self.docs_df)
                    self.bm25 = BM25MicroRetriever()
                    self.bm25.fit(micro_chunks.to_dict("records"), show_progress=False)

        if self.bm25_pyvi is None:
            bm25_pyvi_index_dir = self.index_dir / "bm25_pyvi"
            bm25_pyvi_file = bm25_pyvi_index_dir / "bm25_pyvi_index.pkl"
            if bm25_pyvi_file.exists():
                self.bm25_pyvi = BM25PyViRetriever.load(bm25_pyvi_file)
            elif bm25_pyvi_index_dir.exists() and list(bm25_pyvi_index_dir.glob("*.pkl")):
                self.bm25_pyvi = BM25PyViRetriever.load(bm25_pyvi_index_dir)
            else:
                chunks_path = self.data_dir / "chunks.parquet"
                if chunks_path.exists():
                    chunks_df = pd.read_parquet(chunks_path)
                    micro_chunks = chunks_df[chunks_df["granularity"] == "micro"] if "granularity" in chunks_df.columns else chunks_df
                    if self.docs_df is not None:
                        from src.retrieval.build_indexes import enrich_chunks_with_doc_metadata
                        micro_chunks = enrich_chunks_with_doc_metadata(micro_chunks, self.docs_df)
                    self.bm25_pyvi = BM25PyViRetriever()
                    self.bm25_pyvi.fit(micro_chunks.to_dict("records"), show_progress=False)

        if self.dense is None:
            dense_dek21 = self.index_dir / "dense_dek21"
            dense_std = self.index_dir / "dense"
            dense_path = dense_dek21 if dense_dek21.exists() else dense_std
            if dense_path.exists() and (dense_path / "embeddings.npy").exists():
                try:
                    from src.retrieval.dense_macro import pinned_dense_revision

                    self.dense = DenseMacroRetriever.load(
                        dense_path,
                        model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
                        revision=pinned_dense_revision("CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"),
                        device=self.dense_device,
                    )
                except Exception as e:
                    print(f"Warning: Dense retriever could not be loaded from {dense_path}: {e}")
                    self.dense = None

    def _build_question_memory(self, **kwargs: Any) -> TrainQuestionMemory:
        """Build fold-local memory while reusing one dense encoder when no index exists."""
        dense_encoder = self.dense
        has_dense = self.dense is not None or bool(self.train_query_embeddings)
        if not has_dense:
            return TrainQuestionMemory(
                dense_encoder=None,
                use_dense=False,
                dense_device=self.dense_device,
                **kwargs,
            )

        if dense_encoder is None:
            if self._shared_memory_dense_encoder is None:
                self._shared_memory_dense_encoder = DenseMacroRetriever(
                    model_name="mock" if self.smoke else DEFAULT_MODEL_NAME,
                    dimension=DEFAULT_DIMENSION,
                    use_pyvi=not self.smoke,
                    device=self.dense_device or "cpu",
                )
            dense_encoder = self._shared_memory_dense_encoder
        return TrainQuestionMemory(
            dense_encoder=dense_encoder,
            use_dense=True,
            dense_device=self.dense_device,
            **kwargs,
        )

    def precompute_train_query_embeddings(self) -> dict[str, np.ndarray]:
        """Precompute normalized dense query embeddings once on GPU 0 and index by query_id."""
        if self.train_query_embeddings:
            return self.train_query_embeddings
        if self.dense is not None and self.queries_map:
            qids = list(self.queries_map.keys())
            texts = [self.queries_map[qid] for qid in qids]
            try:
                embs = self.dense.encode_queries(texts, batch_size=64)
                for qid, emb in zip(qids, embs):
                    self.train_query_embeddings[str(qid)] = emb
                print(f"[+] Precomputed and cached {len(self.train_query_embeddings):,} train query dense embeddings on GPU.")
            except Exception as e:
                print(f"[-] Warning: query embedding precomputation skipped: {e}")
        return self.train_query_embeddings

    def get_splits(self) -> list[dict[str, Any]]:
        """Load or generate 5-fold cross-validation splits and verify isolation."""
        if self.splits_path.exists():
            with open(self.splits_path, "r", encoding="utf-8") as f:
                folds = json.load(f)
        elif (self.data_dir / "splits/random_5fold.json").exists():
            with open(self.data_dir / "splits/random_5fold.json", "r", encoding="utf-8") as f:
                folds = json.load(f)
        else:
            print(f"Splits not found at {self.splits_path}; generating fresh split...")
            queries_list = [{"query_id": qid} for qid in self.queries_map.keys()]
            folds = generate_random_5fold_split(queries_list, seed=42, num_folds=self.num_folds)
            self.splits_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.splits_path, "w", encoding="utf-8") as f:
                json.dump(folds, f, indent=2)

        # Strict fold isolation verification
        verify_fold_isolation(folds, self.qrels_map)
        return folds

    def _expected_fold_identity(self, fold_idx: int, fold_info: dict[str, Any]) -> dict[str, Any]:
        return _fold_expected_identity(
            fold_idx,
            fold_info,
            smoke=self.smoke,
            smoke_sample_size=self.smoke_sample_size,
            reranker_model=self.reranker_model,
            candidate_k=self.candidate_k,
            rerank_k=self.rerank_k,
            precision=self.precision,
            split_provenance=self.split_provenance,
        )

    def resolved_run_config(self) -> dict[str, Any]:
        """Effective configuration actually used (never budget from YAML alone).

        FULL orchestration uses 150/50 while algorithm YAML specifies 100/30;
        record the resolved values plus model pins and a stable hash so every
        receipt binds the same contract.
        """
        from src.retrieval.dense_macro import pinned_dense_revision as _pinned_dense

        dense_model = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
        try:
            from src.models.bootstrap import MODEL_REGISTRY as _REG

            reranker_rev = (_REG.get(str(self.reranker_model), {}) or {}).get("revision")
        except Exception:
            reranker_rev = None
        cfg: dict[str, Any] = {
            "candidate_k": int(self.candidate_k),
            "rerank_k": int(self.rerank_k),
            "num_folds": int(self.num_folds),
            "precision": str(self.precision),
            "reranker_model": str(self.reranker_model),
            "reranker_revision": reranker_rev,
            "dense_model": dense_model,
            "dense_revision": _pinned_dense(dense_model),
            "reranker_batch_size": int(self.reranker_batch_size),
            "reranker_max_length": int(self.reranker_max_length),
            "smoke": bool(self.smoke),
            "smoke_sample_size": int(self.smoke_sample_size),
            "train_reranker_per_fold": bool(self.train_reranker_per_fold),
            "use_reranker": bool(self.use_reranker),
            "doc_disjoint": bool(self.doc_disjoint),
        }
        try:
            if isinstance(self.split_provenance, dict):
                for key in ("random_5fold", "doc_disjoint"):
                    part = self.split_provenance.get(key) or {}
                    if isinstance(part, dict) and part.get("sha256"):
                        cfg[f"split_{key}_sha256"] = str(part["sha256"])
        except Exception:
            pass
        cfg["config_sha256"] = hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return cfg

    def _try_reuse_completed_fold(
        self,
        fold_idx: int,
        fold_info: dict[str, Any],
        fold_dir: Path,
    ) -> tuple[dict[str, list[str]] | None, dict[str, list[str]] | None, list, dict | None]:
        """Validate persisted fold artifacts against current identity.

        Returns ``(preds, candidates, feature_dfs, metrics)`` on success, else
        ``(None, None, [], None)`` to force recomputation. Reuse is only valid
        when ``output_dir`` is explicitly reused (local reruns); Modal launches
        create a fresh UUID attempt directory per invocation and therefore
        never hit this path across attempts.
        """
        complete_marker = fold_dir / "complete.json"
        predictions_path = fold_dir / "predictions.parquet"
        metrics_path = fold_dir / "metrics.json"
        if not (complete_marker.is_file() and predictions_path.is_file() and metrics_path.is_file()):
            return None, None, [], None
        try:
            complete_data = json.loads(complete_marker.read_text(encoding="utf-8"))
            if complete_data.get("status") != "COMPLETED":
                print(f"[-] Fold {fold_idx} marker status is {complete_data.get('status')}; recomputing.")
                return None, None, [], None
            if int(complete_data.get("fold", -1)) != int(fold_idx):
                print(f"[-] Fold {fold_idx} marker fold mismatch; recomputing.")
                return None, None, [], None
            expected = self._expected_fold_identity(fold_idx, fold_info)
            for key in (
                "train_ids_sha256",
                "val_ids_sha256",
                "train_count",
                "val_count",
                "reranker_model",
                "candidate_k",
                "rerank_k",
                "precision",
                "smoke",
            ):
                if complete_data.get(key) != expected.get(key):
                    print(
                        f"[-] Fold {fold_idx} identity mismatch on '{key}' "
                        f"(stored={complete_data.get(key)} expected={expected.get(key)}); recomputing."
                    )
                    return None, None, [], None
            exp_split_sha = expected.get("split_random_5fold_sha256")
            if exp_split_sha and complete_data.get("split_random_5fold_sha256") not in (None, exp_split_sha):
                if complete_data.get("split_random_5fold_sha256") != exp_split_sha:
                    print(f"[-] Fold {fold_idx} split SHA mismatch; recomputing.")
                    return None, None, [], None

            f_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if int(f_metrics.get("fold", fold_idx)) != int(fold_idx):
                print(f"[-] Fold {fold_idx} metrics fold mismatch; recomputing.")
                return None, None, [], None

            f_preds_df = pd.read_parquet(predictions_path)
            f_preds = {
                str(r["query_id"]): list(r["predicted_doc_ids"]) for r in f_preds_df.to_dict("records")
            }
            raw_val = [str(x) for x in fold_info.get("val_query_ids", fold_info.get("val", []))]
            if self.smoke:
                raw_val = raw_val[: self.smoke_sample_size]
            expected_val_set = set(raw_val)
            if set(f_preds.keys()) != expected_val_set:
                missing = sorted(expected_val_set - set(f_preds.keys()))[:5]
                extra = sorted(set(f_preds.keys()) - expected_val_set)[:5]
                print(
                    f"[-] Fold {fold_idx} predictions ID mismatch "
                    f"(missing={missing} extra={extra}); recomputing."
                )
                return None, None, [], None

            f_cands: dict[str, list[str]] = {}
            cands_path = fold_dir / "candidates.parquet"
            if cands_path.is_file():
                f_cands_df = pd.read_parquet(cands_path)
                f_cands = {
                    str(r["query_id"]): list(r["candidate_doc_ids"])
                    for r in f_cands_df.to_dict("records")
                }
                if set(f_cands.keys()) and set(f_cands.keys()) != expected_val_set:
                    print(f"[-] Fold {fold_idx} candidates ID mismatch; recomputing.")
                    return None, None, [], None

            features_path = fold_dir / "features.parquet"
            f_feat_dfs = [pd.read_parquet(features_path)] if features_path.is_file() else []

            # When fold adapters are trained, require the adapter artifacts to
            # exist and match the recorded checksum; otherwise recompute.
            if self.train_reranker_per_fold and self.reranker_model != "mock":
                adapter_dir = fold_dir / "reranker_adapter"
                adapter_cfg = adapter_dir / "adapter_config.json"
                weights = adapter_dir / "adapter_model.safetensors"
                if not weights.exists():
                    weights = adapter_dir / "adapter_model.bin"
                manifest_file = adapter_dir / "training_manifest.json"
                if not (adapter_cfg.is_file() and weights.is_file() and manifest_file.is_file()):
                    print(f"[-] Fold {fold_idx} adapter artifacts incomplete; recomputing.")
                    return None, None, [], None
                try:
                    m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    recorded = f_metrics.get("adapter_checksum") or complete_data.get("adapter_checksum")
                    if recorded:
                        actual = hashlib.sha256(weights.read_bytes()).hexdigest()
                        if actual != recorded:
                            print(f"[-] Fold {fold_idx} adapter checksum mismatch; recomputing.")
                            return None, None, [], None
                    # Base-model identity must also match; checksums alone cannot
                    # detect an upstream weight change.
                    manifest_base = str(m_data.get("base_model") or "")
                    if manifest_base and manifest_base != "mock" and not Path(manifest_base).is_dir():
                        if manifest_base != str(self.reranker_model):
                            print(f"[-] Fold {fold_idx} adapter base_model mismatch; recomputing.")
                            return None, None, [], None
                except Exception as exc:
                    print(f"[-] Fold {fold_idx} adapter validation failed ({type(exc).__name__}); recomputing.")
                    return None, None, [], None

            print(
                f"[+] Reusing completed Fold {fold_idx} from {fold_dir} "
                f"(Recall@5: {f_metrics.get('recall@5', 0.0):.4f}; identity-verified)"
            )
            return f_preds, f_cands, f_feat_dfs, f_metrics
        except Exception as e:
            print(f"[-] Warning: Failed loading completed fold {fold_idx} cache, recomputing: {e}")
            return None, None, [], None

    def run_fold(
        self,
        fold_idx: int,
        fold_info: dict[str, Any],
        reranker: CrossEncoderReranker | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], list[pd.DataFrame], dict[str, Any], dict[str, float]]:
        """
        Execute OOF evaluation on a single fold with strict isolation guarantees:
        1. Construct TrainQuestionMemory on training queries only.
        2. Retrieve top candidates for validation queries.
        3. Extract candidate features.
        4. Optional cross-encoder reranking.
        5. Top-5 selection and metric computation.
        6. Assert official scorer equivalence.
        """
        train_ids = set(str(x) for x in fold_info.get("train_query_ids", fold_info.get("train", [])))
        val_ids = [str(x) for x in fold_info.get("val_query_ids", fold_info.get("val", []))]

        if self.smoke:
            val_ids = val_ids[: self.smoke_sample_size]

        fold_train_queries = [
            (qid, self.queries_map[qid], self.train_query_embeddings.get(qid))
            if qid in self.train_query_embeddings
            else (qid, self.queries_map[qid], None)
            for qid in train_ids if qid in self.queries_map
        ]
        fold_train_qrels = {qid: self.qrels_map[qid] for qid in train_ids if qid in self.qrels_map}

        # Build fold-isolated question memory
        memory = self._build_question_memory(min_similarity=0.82)
        memory.fit(fold_train_queries, fold_train_qrels)

        # Strict validation: memory must not contain any validation query
        leaked_val = set(val_ids) & memory.training_query_ids
        if leaked_val:
            raise AssertionError(f"Fold {fold_idx} Question Memory contains validation queries: {leaked_val}")

        hybrid_engine = HybridSearchEngine(
            bm25_retriever=self.bm25,
            bm25_pyvi_retriever=self.bm25_pyvi,
            dense_retriever=self.dense,
            question_memory=memory,
            exact_matcher=self.exact,
        )

        # Compute fold-specific training doc frequencies (zero validation label leakage)
        fold_train_doc_freq = compute_training_doc_frequencies(fold_train_qrels)

        fold_preds: dict[str, list[str]] = {}
        fold_candidates: dict[str, list[str]] = {}
        fold_feature_dfs: list[pd.DataFrame] = []
        fold_runtimes: dict[str, float] = {}

        t0 = time.time()
        window_size = min(32, max(1, self.reranker_batch_size))
        val_id_batches = [val_ids[i : i + window_size] for i in range(0, len(val_ids), window_size)]

        for batch_qids in tqdm(val_id_batches, desc=f"Fold {fold_idx} OOF Inference", leave=False):
            window_items: list[tuple[str, str, list[CandidateRecord], float]] = []
            for qid in batch_qids:
                q_text = self.queries_map.get(qid, "")
                t_q0 = time.time()
                q_emb = self.train_query_embeddings.get(qid)

                branch_cands = None
                if self._static_branch_cache is not None:
                    if qid not in self._static_branch_cache:
                        exact_cands = self.exact.search(q_text, top_k=10) if self.exact else []
                        bm25_cands = self.bm25.search(q_text, top_k=max(80, self.candidate_k)) if self.bm25 else []
                        pyvi_cands = self.bm25_pyvi.search(q_text, top_k=max(80, self.candidate_k)) if self.bm25_pyvi else []
                        dense_cands = self.dense.retrieve(q_text, top_k=max(80, self.candidate_k), q_emb=q_emb) if self.dense else []
                        self._static_branch_cache[qid] = {
                            "exact": exact_cands,
                            "bm25": bm25_cands,
                            "bm25_pyvi": pyvi_cands,
                            "dense": dense_cands,
                            "bm25_80": bm25_cands[:80],
                            "pyvi_80": pyvi_cands[:80],
                            "dense_80": dense_cands[:80],
                        }
                    cached_s = self._static_branch_cache[qid]
                    branch_cands = {
                        "exact": cached_s.get("exact", []),
                        "bm25": cached_s.get("bm25", []),
                        "bm25_pyvi": cached_s.get("bm25_pyvi", []),
                        "dense": cached_s.get("dense", []),
                    }

                candidates: list[CandidateRecord] = hybrid_engine.search_candidates(
                    query=q_text,
                    top_k=self.candidate_k,
                    exclude_qid=str(qid),
                    q_emb=q_emb,
                    branch_candidates=branch_cands,
                )
                cand_ids = [str(c["doc_id"]) for c in candidates]
                fold_candidates[qid] = cand_ids
                window_items.append((qid, q_text, candidates, t_q0))

            # Rerank batch across the query window
            if reranker is not None and self.evidence_builder is not None:
                q_cands = [(item[1], item[2]) for item in window_items]
                if hasattr(reranker, "rerank_batch"):
                    reranked_list = reranker.rerank_batch(
                        q_cands,
                        evidence_builder=self.evidence_builder,
                        top_k=self.rerank_k,
                        batch_size=self.reranker_batch_size,
                        max_length=self.reranker_max_length,
                    )
                else:
                    reranked_list = [
                        reranker.rerank(
                            query=q,
                            candidates=c,
                            evidence_builder=self.evidence_builder,
                            top_k=self.rerank_k,
                            batch_size=self.reranker_batch_size,
                            max_length=self.reranker_max_length,
                        )
                        for q, c in q_cands
                    ]
            else:
                reranked_list = [item[2] for item in window_items]

            for (qid, q_text, _, t_q0), candidates in zip(window_items, reranked_list):
                # Extract features for candidate union AFTER reranking
                feat_df = extract_candidate_features(
                    query_id=qid,
                    candidate_records=candidates,
                    query_text=q_text,
                    doc_freq_map=fold_train_doc_freq,
                    qrels=self.qrels_map,
                )
                if not feat_df.empty:
                    feat_df["fold"] = fold_idx
                    fold_feature_dfs.append(feat_df)

                # Fixed predeclared RRF before selection, mirroring public
                # inference and the disjoint trained pass: the confirmatory OOF
                # score must evaluate the submission scoring policy, not
                # reranker-order top-5. Fold-local doc frequencies only.
                ranked = _OOF_FIXED_RRF.predict(
                    candidates,
                    query_id=qid,
                    query_text=q_text,
                    doc_freq_map=fold_train_doc_freq,
                )
                top5 = self.selector.select(ranked)
                fold_preds[qid] = top5
                fold_runtimes[qid] = time.time() - t_q0

        elapsed_total = time.time() - t0

        # Evaluate fold metrics
        fold_gold = {qid: self.qrels_map[qid] for qid in val_ids}
        fold_metrics = evaluate_predictions(
            y_pred=fold_preds,
            y_true=fold_gold,
            candidate_pools=fold_candidates,
            runtimes=fold_runtimes,
            cutoffs=DEFAULT_CANDIDATE_CUTOFFS,
        )
        fold_metrics["fold"] = fold_idx
        fold_metrics["val_queries"] = len(val_ids)
        fold_metrics["elapsed_seconds"] = elapsed_total

        # Verify exact Codabench official scorer equivalence
        assert_official_equivalence(fold_preds, fold_gold)

        return fold_preds, fold_candidates, fold_feature_dfs, fold_metrics, fold_runtimes

    def run(self) -> dict[str, Any]:
        """
        Execute full 5-fold out-of-fold cross-validation pipeline:
        - Orchestrates 5 folds with strict isolation
        - Gathers OOF predictions, OOF candidate pools, and OOF candidate features
        - Exports oof_predictions.parquet, oof_features.parquet, and cv_report.json
        """
        print("=" * 70)
        mode_str = "SMOKE / FAST" if self.smoke else "FULL"
        print(f"LegalIR Task 1: 5-Fold OOF Validation [{mode_str} MODE]")
        print("=" * 70)

        # Preflight parameter budget audit
        audit_path = self.output_dir / "parameter_audit.json"
        audit_report = audit_system_parameters(
            config_path=self.config_path,
            output_json=audit_path,
            raise_on_violation=True,
            offline_fallback=True,
        )
        print(
            f"[Preflight] Parameter Budget Audit: {audit_report['total_learned_parameters']:,} params "
            f"({audit_report['total_parameters_billions']:.4f}B / 4.0B, "
            f"{audit_report['budget_utilization_pct']:.2f}% utilization). PASS\n"
        )

        self.load_data()
        self.load_retrievers()
        self.precompute_train_query_embeddings()
        folds = self.get_splits()

        global_reranker: CrossEncoderReranker | None = None
        if self.use_reranker and not self.train_reranker_per_fold:
            print(f"Initializing CrossEncoderReranker: {self.reranker_model}...")
            global_reranker = CrossEncoderReranker(
                model_name=self.reranker_model,
                device=self.device,
                batch_size=self.reranker_batch_size,
                max_length=self.reranker_max_length,
                precision=self.precision,
            )

        all_oof_predictions: dict[str, list[str]] = {}
        all_candidate_pools: dict[str, list[str]] = {}
        all_feature_dfs: list[pd.DataFrame] = []
        all_runtimes: dict[str, float] = {}
        fold_records: list[dict[str, Any]] = []

        total_t0 = time.time()
        active_folds = folds[: self.num_folds]

        for f_idx, fold_info in enumerate(active_folds):
            print(f"\n>>> Running Fold {f_idx + 1}/{len(active_folds)} (Fold {f_idx})...")

            fold_dir = self.output_dir / f"fold_{f_idx}"
            predictions_path = fold_dir / "predictions.parquet"
            features_path = fold_dir / "features.parquet"
            metrics_path = fold_dir / "metrics.json"

            # F4: reuse DISABLED by default. Modal uses a fresh UUID attempt dir
            # per invocation (no cross-attempt resume); even with an explicitly
            # reused output_dir, reuse requires opt-in until the F4 contract
            # (versioned identity + every downstream-required artifact) is complete.
            if self.allow_stage_reuse:
                reused = self._try_reuse_completed_fold(f_idx, fold_info, fold_dir)
                if reused[3] is not None:
                    f_preds, f_cands, f_feat_dfs, f_metrics = reused
                    assert f_preds is not None and f_metrics is not None
                    all_oof_predictions.update(f_preds)
                    all_candidate_pools.update(f_cands or {})
                    all_feature_dfs.extend(f_feat_dfs)
                    fold_records.append(f_metrics)
                    continue

            fold_reranker: CrossEncoderReranker | None = None
            pair_mining_sec = 0.0
            train_sec = 0.0
            opt_steps = 0
            train_report = {}

            if self.train_reranker_per_fold:
                print(f"--- Training fold-specific LoRA reranker for Fold {f_idx} ---")
                fold_dir = self.output_dir / f"fold_{f_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                pairs_dir = fold_dir / "pairs"

                from src.training.build_pairs import build_training_pairs
                from src.training.train_reranker import train_reranker

                train_ids = sorted(str(x) for x in fold_info.get("train_query_ids", fold_info.get("train", [])))
                val_ids = sorted(str(x) for x in fold_info.get("val_query_ids", fold_info.get("val", [])))

                t_pm0 = time.time()
                _, pairs_df = build_training_pairs(
                    data_dir=self.data_dir,
                    index_dir=self.index_dir,
                    output_dir=pairs_dir,
                    fold=f_idx,
                    train_query_ids=train_ids,
                    use_all_queries=False,
                    limit=self.smoke_sample_size if self.smoke else None,
                    query_embeddings=self.train_query_embeddings,
                    duplicate_groups_path=self.duplicate_groups_path,
                    static_branch_cache=self._static_branch_cache,
                )
                pair_mining_sec = time.time() - t_pm0

                pair_qids = set(pairs_df["query_id"].astype(str)) if not pairs_df.empty else set()
                train_set = set(map(str, train_ids))
                val_set = set(map(str, val_ids))

                unknown = pair_qids - train_set
                leaked = pair_qids & val_set
                if unknown or leaked:
                    raise AssertionError(
                        f"Fold {f_idx} pair isolation failed: "
                        f"unknown={sorted(unknown)[:10]}, leaked={sorted(leaked)[:10]}"
                    )
                # Missing-query check: isolation alone cannot detect silent
                # coverage loss (pairs covering only a subset of train queries).
                # Run the same expected-query audit used for final training.
                from src.training.trainer import audit_pair_coverage as _audit_fold_pairs

                _fold_audit = _audit_fold_pairs(pairs_df, expected_qids=train_set)
                _missing = sorted(train_set - pair_qids)[:10]
                if not self.smoke:
                    if _fold_audit.get("missing_queries_count", 0) > 0:
                        raise AssertionError(
                            f"Fold {f_idx} pair coverage failed: "
                            f"{_fold_audit.get('missing_queries_count')} expected training queries "
                            f"have no pairs (e.g. {sorted(train_set - pair_qids)[:10]}). "
                            f"pos_cov={_fold_audit.get('positive_coverage_pct')}% "
                            f"neg_cov={_fold_audit.get('negative_coverage_pct')}%"
                        )
                    if float(_fold_audit.get("positive_coverage_pct", 0.0)) < 100.0:
                        raise AssertionError(
                            f"Fold {f_idx} requires 100% positive pair coverage, "
                            f"got {_fold_audit.get('positive_coverage_pct')}%"
                        )
                    if float(_fold_audit.get("negative_coverage_pct", 0.0)) < 99.0:
                        raise AssertionError(
                            f"Fold {f_idx} requires >=99% negative pair coverage, "
                            f"got {_fold_audit.get('negative_coverage_pct')}%"
                        )
                elif _missing and len(train_set) <= 200:
                    # Smoke runs use bounded subsets; still surface gaps loudly.
                    print(
                        f"[!] Fold {f_idx} smoke pair coverage gap: "
                        f"{len(train_set - pair_qids)} expected queries missing pairs."
                    )

                adapter_dir = fold_dir / "reranker_adapter"
                reranker_cfg = self.reranker_config_path or self.config_path or "configs/experiments/reranker_lora.yaml"
                base_m_name = self.reranker_model if self.reranker_model != "mock" else None

                t_tr0 = time.time()
                train_report = train_reranker(
                    pairs_file=pairs_dir / "reranker_pairs.parquet",
                    config_path=reranker_cfg,
                    output_dir=adapter_dir,
                    fold=f_idx,
                    base_model_name=base_m_name,
                    max_steps=5 if self.smoke else None,
                    device=self.reranker_device,
                    precision=self.precision,
                    num_workers=self.num_workers,
                    enforce_full_coverage_steps=not self.smoke,
                )
                train_sec = time.time() - t_tr0
                opt_steps = int(train_report.get("optimizer_steps", train_report.get("global_steps", 0)))

                fold_reranker = CrossEncoderReranker(
                    model_name=self.reranker_model,
                    adapter_path=adapter_dir,
                    device=self.reranker_device,
                    batch_size=self.reranker_batch_size,
                    max_length=self.reranker_max_length,
                    precision=self.precision,
                    revision=train_report.get("base_model_revision"),
                )
            elif self.use_reranker:
                fold_reranker = global_reranker

            t_inf0 = time.time()
            f_preds, f_cands, f_feat_dfs, f_metrics, f_runtimes = self.run_fold(
                fold_idx=f_idx,
                fold_info=fold_info,
                reranker=fold_reranker,
            )
            infer_sec = time.time() - t_inf0
            val_q_count = len(f_preds)
            f_metrics["pair_mining_seconds"] = round(pair_mining_sec, 3)
            f_metrics["reranker_training_seconds"] = round(train_sec, 3)
            f_metrics["reranker_optimizer_steps"] = opt_steps
            f_metrics["heldout_inference_seconds"] = round(infer_sec, 3)
            f_metrics["heldout_queries"] = val_q_count
            f_metrics["heldout_queries_per_second"] = round(val_q_count / max(0.001, infer_sec), 2)
            f_metrics["training_query_count"] = len(train_ids) if 'train_ids' in locals() else len(fold_info.get("train_query_ids", []))
            f_metrics["pair_query_count"] = len(pair_qids) if 'pair_qids' in locals() else 0
            f_metrics["validation_query_count"] = len(val_ids) if 'val_ids' in locals() else len(fold_info.get("val_query_ids", []))
            f_metrics["pair_unknown_train_count"] = len(unknown) if 'unknown' in locals() else 0
            f_metrics["pair_validation_leakage_count"] = len(leaked) if 'leaked' in locals() else 0
            f_metrics["pair_validation_leakage_ids"] = sorted(leaked) if 'leaked' in locals() else []

            if fold_reranker is not None:
                f_metrics["reranker_oom_events"] = int(getattr(fold_reranker, "oom_events", 0))
                f_metrics["min_successful_batch_size"] = int(getattr(fold_reranker, "min_successful_batch_size", 16))

            if self.train_reranker_per_fold and fold_reranker is not None:
                f_metrics["training_queries"] = len(train_ids) if 'train_ids' in locals() else len(fold_info.get("train_query_ids", []))
                f_metrics["training_pairs"] = len(pairs_df) if 'pairs_df' in locals() else 0
                f_metrics["adapter_path"] = str(adapter_dir)
                f_metrics["adapter_checksum"] = train_report.get("adapter_checksum")
                f_metrics["param_diff"] = train_report.get("param_diff")

                del fold_reranker
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_oof_predictions.update(f_preds)
            all_candidate_pools.update(f_cands)
            all_feature_dfs.extend(f_feat_dfs)
            all_runtimes.update(f_runtimes)
            fold_records.append(f_metrics)

            rec5 = f_metrics.get("recall@5", 0.0)
            prec5 = f_metrics.get("precision@5", 0.0)
            cand50 = f_metrics.get("candidate_recall@50", 0.0)
            cand150 = f_metrics.get("candidate_recall@150", 0.0)
            mrr = f_metrics.get("mrr", 0.0)
            map_score = f_metrics.get("map", 0.0)
            ndcg5 = f_metrics.get("ndcg@5", 0.0)
            elapsed = f_metrics.get("elapsed_seconds", 0.0)

            print(
                f"Fold {f_idx} Results: "
                f"Recall@5 = {rec5 * 100:.2f}% | "
                f"Prec@5 = {prec5 * 100:.2f}% | "
                f"MRR = {mrr:.4f} | "
                f"MAP = {map_score:.4f} | "
                f"nDCG@5 = {ndcg5:.4f} | "
                f"Cand@50 = {cand50 * 100:.2f}% | "
                f"Cand@150 = {cand150 * 100:.2f}% "
                f"({elapsed:.1f}s)"
            )

            # Persist durable fold outputs for completed-stage recovery
            fold_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                {"query_id": qid, "predicted_doc_ids": docs}
                for qid, docs in f_preds.items()
            ]).to_parquet(predictions_path, index=False)
            pd.DataFrame([
                {"query_id": qid, "candidate_doc_ids": docs}
                for qid, docs in f_cands.items()
            ]).to_parquet(fold_dir / "candidates.parquet", index=False)
            if f_feat_dfs:
                pd.concat(f_feat_dfs, ignore_index=True).to_parquet(features_path, index=False)
            metrics_path.write_text(json.dumps(f_metrics, indent=2), encoding="utf-8")
            complete_marker = fold_dir / "complete.json"
            fold_identity = self._expected_fold_identity(f_idx, fold_info)
            fold_identity.update({
                "status": "COMPLETED",
                "queries_count": len(f_preds),
                "recall@5": f_metrics.get("recall@5", 0.0),
                "precision@5": f_metrics.get("precision@5", 0.0),
                "adapter_checksum": f_metrics.get("adapter_checksum"),
                "timestamp": time.time(),
            })
            complete_marker.write_text(
                json.dumps(fold_identity, indent=2),
                encoding="utf-8",
            )

        overall_elapsed = time.time() - total_t0

        # Exact expected-query coverage: scoring population must equal the full
        # expected held-out set (never just whichever predictions were returned).
        _expected_oof_ids: set[str] = set()
        for _fi, _fold_info in enumerate(active_folds):
            _raw = [str(x) for x in _fold_info.get("val_query_ids", _fold_info.get("val", []))]
            if self.smoke:
                _raw = _raw[: self.smoke_sample_size]
            _expected_oof_ids.update(_raw)
        if set(all_oof_predictions.keys()) != _expected_oof_ids:
            _missing = sorted(_expected_oof_ids - set(all_oof_predictions.keys()))[:10]
            _extra = sorted(set(all_oof_predictions.keys()) - _expected_oof_ids)[:10]
            raise AssertionError(
                f"OOF prediction coverage failed: expected {len(_expected_oof_ids)} queries, "
                f"got {len(all_oof_predictions)} (missing={_missing} extra={_extra})."
            )

        # Global aggregate evaluation across all OOF queries
        all_gold = {qid: self.qrels_map[qid] for qid in all_oof_predictions.keys()}
        overall_metrics = evaluate_predictions(
            y_pred=all_oof_predictions,
            y_true=all_gold,
            candidate_pools=all_candidate_pools,
            runtimes=all_runtimes,
            cutoffs=DEFAULT_CANDIDATE_CUTOFFS,
        )

        # Cross-fold mean and std calculations
        rec5_scores = [f["recall@5"] for f in fold_records]
        prec5_scores = [f["precision@5"] for f in fold_records]
        rec1_scores = [f["recall@1"] for f in fold_records]
        rec3_scores = [f["recall@3"] for f in fold_records]
        mrr_scores = [f.get("mrr", 0.0) for f in fold_records]
        map_scores = [f.get("map", 0.0) for f in fold_records]
        ndcg5_scores = [f.get("ndcg@5", 0.0) for f in fold_records]

        cand_means = {
            f"mean_candidate@{k}": float(np.mean([f.get(f"candidate_recall@{k}", 0.0) for f in fold_records]))
            for k in DEFAULT_CANDIDATE_CUTOFFS
        }

        heldout_inf_sec_total = sum(f.get("heldout_inference_seconds", 0.0) for f in fold_records)
        heldout_q_total = sum(f.get("heldout_queries", 0) for f in fold_records)
        heldout_inf_qps = round(heldout_q_total / max(0.001, heldout_inf_sec_total), 2)
        rerank_train_sec_total = sum(f.get("reranker_training_seconds", 0.0) for f in fold_records)
        rerank_opt_steps_total = sum(f.get("reranker_optimizer_steps", 0) for f in fold_records)
        pair_mining_sec_total = sum(f.get("pair_mining_seconds", 0.0) for f in fold_records)

        resolved_cfg = self.resolved_run_config()
        print(
            f"[Config] FULL effective: candidate_k={resolved_cfg['candidate_k']} "
            f"rerank_k={resolved_cfg['rerank_k']} folds={resolved_cfg['num_folds']} "
            f"precision={resolved_cfg['precision']} sha={resolved_cfg['config_sha256'][:12]}..."
        )
        cv_report = {
            "resolved_config": resolved_cfg,
            "resolved_config_sha256": resolved_cfg["config_sha256"],
            "mean_recall@5": float(np.mean(rec5_scores)),
            "std_recall@5": float(np.std(rec5_scores)),
            "mean_precision@5": float(np.mean(prec5_scores)),
            "std_precision@5": float(np.std(prec5_scores)),
            "mean_recall@1": float(np.mean(rec1_scores)),
            "mean_recall@3": float(np.mean(rec3_scores)),
            "mean_mrr": float(np.mean(mrr_scores)),
            "mean_map": float(np.mean(map_scores)),
            "mean_ndcg@5": float(np.mean(ndcg5_scores)),
            **cand_means,
            "overall_aggregate_metrics": overall_metrics,
            "total_evaluated_queries": len(all_oof_predictions),
            "total_runtime_seconds": overall_elapsed,
            "heldout_inference_seconds_total": round(heldout_inf_sec_total, 3),
            "heldout_queries_total": heldout_q_total,
            "heldout_inference_queries_per_second": heldout_inf_qps,
            "reranker_training_seconds_total": round(rerank_train_sec_total, 3),
            "reranker_optimizer_steps_total": rerank_opt_steps_total,
            "pair_mining_seconds_total": round(pair_mining_sec_total, 3),
            "runtime_per_query_ms": float(np.mean(list(all_runtimes.values()))) * 1000.0 if all_runtimes else 0.0,
            "queries_per_second": (len(all_oof_predictions) / overall_elapsed) if overall_elapsed > 0 else 0.0,
            "official_scorer_parity_verified": True,
            "is_smoke_mode": self.smoke,
            "num_folds": len(active_folds),
            "folds": fold_records,
            "split_provenance": self.split_provenance,
            "max_pair_validation_leakage_count": max((f.get("pair_validation_leakage_count", 0) for f in fold_records), default=0),
            "max_pair_unknown_train_count": max((f.get("pair_unknown_train_count", 0) for f in fold_records), default=0),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        # 1. Save cv_report.json
        report_path = self.output_dir / "cv_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(cv_report, f, indent=2)

        # 2. Save oof_predictions.parquet & oof_predictions.json
        pred_records = []
        for f_idx, fold_info in enumerate(active_folds):
            val_ids = [str(x) for x in fold_info.get("val_query_ids", fold_info.get("val", []))]
            if self.smoke:
                val_ids = val_ids[: self.smoke_sample_size]
            for qid in val_ids:
                pred_records.append({
                    "query_id": str(qid),
                    "answer": all_oof_predictions.get(qid, []),
                    "fold": f_idx,
                })

        pred_df = pd.DataFrame(pred_records)
        pred_parquet_path = self.output_dir / "oof_predictions.parquet"
        pred_df.to_parquet(pred_parquet_path, index=False)

        pred_json_path = self.output_dir / "oof_predictions.json"
        codabench_preds = {r["query_id"]: {"answer": r["answer"]} for r in pred_records}
        with open(pred_json_path, "w", encoding="utf-8") as f:
            json.dump(codabench_preds, f, indent=2)

        # 3. Save oof_features.parquet
        feat_parquet_path = self.output_dir / "oof_features.parquet"
        if all_feature_dfs:
            combined_features_df = pd.concat(all_feature_dfs, ignore_index=True)
            combined_features_df.to_parquet(feat_parquet_path, index=False)
            print(f"Saved OOF Features: {feat_parquet_path} ({len(combined_features_df)} candidate rows)")
        else:
            # Create empty placeholder DataFrame with schema
            empty_df = pd.DataFrame(columns=["query_id", "doc_id", "label", "fold"])
            empty_df.to_parquet(feat_parquet_path, index=False)

        # 4. Optional Document-Disjoint Robustness Split Evaluation
        if self.doc_disjoint:
            print("\n>>> Running Document-Disjoint Robustness Split Evaluation...")
            self.run_document_disjoint_evaluation(reranker=global_reranker)

        # Print final summary
        print("\n" + "=" * 70)
        print(">> 5-FOLD OUT-OF-FOLD (OOF) CV SUMMARY:")
        print(f"   Mean Recall@5       : {cv_report['mean_recall@5'] * 100:.4f}% (+/- {cv_report['std_recall@5'] * 100:.4f}%)")
        print(f"   Mean Precision@5    : {cv_report['mean_precision@5'] * 100:.4f}% (+/- {cv_report['std_precision@5'] * 100:.4f}%)")
        print(f"   Mean MRR            : {cv_report['mean_mrr']:.4f}")
        print(f"   Mean MAP            : {cv_report['mean_map']:.4f}")
        print(f"   Mean nDCG@5         : {cv_report['mean_ndcg@5']:.4f}")
        print(f"   Candidate Recall@50 : {cv_report.get('mean_candidate@50', 0.0) * 100:.4f}%")
        print(f"   Candidate Recall@150: {cv_report.get('mean_candidate@150', 0.0) * 100:.4f}%")
        print(f"   Candidate Recall@200: {cv_report.get('mean_candidate@200', 0.0) * 100:.4f}%")
        print(f"   Total Queries       : {cv_report['total_evaluated_queries']}")
        print(f"   Runtime / Query     : {cv_report['runtime_per_query_ms']:.2f} ms")
        print(f"   Artifacts Exported  : {self.output_dir}")
        print("=" * 70)

        return cv_report

    def run_document_disjoint_evaluation(
        self,
        reranker: CrossEncoderReranker | None = None,
    ) -> dict[str, Any]:
        """Evaluate document-disjoint split to test generalization to unseen documents."""
        report_path = self.output_dir / "doc_disjoint_report.json"
        doc_disjoint_dir = self.output_dir / "doc_disjoint"
        dj_complete_marker = doc_disjoint_dir / "complete.json"
        # F4: disjoint reuse also gated on explicit opt-in.
        if self.allow_stage_reuse and dj_complete_marker.is_file() and report_path.is_file():
            try:
                _dj_complete = json.loads(dj_complete_marker.read_text(encoding="utf-8"))
                if _dj_complete.get("status") != "COMPLETED":
                    raise ValueError(f"status={_dj_complete.get('status')}")
                # Identity check: split SHAs and config must match current run.
                _exp_split_sha = None
                try:
                    if isinstance(self.split_provenance, dict):
                        _dj_prov = (self.split_provenance.get("doc_disjoint") or {})
                        if isinstance(_dj_prov, dict) and _dj_prov.get("sha256"):
                            _exp_split_sha = str(_dj_prov.get("sha256"))
                except Exception:
                    pass
                if _exp_split_sha and _dj_complete.get("split_doc_disjoint_sha256") not in (None, _exp_split_sha):
                    if _dj_complete.get("split_doc_disjoint_sha256") != _exp_split_sha:
                        raise ValueError("split SHA mismatch")
                if _dj_complete.get("reranker_model") not in (None, str(self.reranker_model)):
                    if _dj_complete.get("reranker_model") != str(self.reranker_model):
                        raise ValueError("reranker_model mismatch")
                if _dj_complete.get("smoke") not in (None, bool(self.smoke)):
                    if bool(_dj_complete.get("smoke")) != bool(self.smoke):
                        raise ValueError("smoke-mode mismatch")
                final_report = json.loads(report_path.read_text(encoding="utf-8"))
                self.doc_disjoint_report = final_report
                print(f"[+] Reusing completed doc-disjoint evaluation from {report_path} (identity-verified)")
                return final_report
            except Exception as e:
                print(f"[-] Warning: Failed loading doc-disjoint report cache, recomputing: {e}")

        if self.doc_disjoint_splits_path.exists():
            with open(self.doc_disjoint_splits_path, "r", encoding="utf-8") as f:
                disjoint_split = json.load(f)
        elif (self.data_dir / "splits/doc_disjoint_split.json").exists():
            with open(self.data_dir / "splits/doc_disjoint_split.json", "r", encoding="utf-8") as f:
                disjoint_split = json.load(f)
        else:
            print("Generating fresh document-disjoint split...")
            queries_list = [{"query_id": qid} for qid in self.queries_map.keys()]
            qrels_list = self.qrels_df.to_dict("records") if self.qrels_df is not None else []
            disjoint_split = generate_document_disjoint_split(queries_list, qrels_list, val_ratio=0.2, seed=42)
            self.doc_disjoint_splits_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.doc_disjoint_splits_path, "w", encoding="utf-8") as f:
                json.dump(disjoint_split, f, indent=2)

        # Verify strict document disjoint isolation
        verify_document_disjoint_isolation(disjoint_split, self.qrels_map)

        train_ids = set(str(x) for x in disjoint_split.get("train_query_ids", disjoint_split.get("train", [])))
        val_ids = [str(x) for x in disjoint_split.get("val_query_ids", disjoint_split.get("val", []))]

        if self.smoke:
            val_ids = val_ids[: self.smoke_sample_size]

        fold_train_queries = {qid: self.queries_map[qid] for qid in train_ids if qid in self.queries_map}
        fold_train_qrels = {qid: self.qrels_map[qid] for qid in train_ids if qid in self.qrels_map}

        memory = self._build_question_memory(min_similarity=0.82)
        memory.fit(fold_train_queries, fold_train_qrels)

        hybrid_engine = HybridSearchEngine(
            bm25_retriever=self.bm25,
            bm25_pyvi_retriever=self.bm25_pyvi,
            dense_retriever=self.dense,
            question_memory=memory,
            exact_matcher=self.exact,
        )

        # 1. Retrieval-only pass
        preds_retrieval: dict[str, list[str]] = {}
        candidates_map: dict[str, list[str]] = {}
        runtimes_retrieval: dict[str, float] = {}

        t0 = time.time()
        for qid in tqdm(val_ids, desc="Doc-Disjoint Retrieval Eval", leave=False):
            q_text = self.queries_map.get(qid, "")
            t_q0 = time.time()
            q_emb = self.train_query_embeddings.get(qid)

            cands = hybrid_engine.search_candidates(
                query=q_text,
                top_k=self.candidate_k,
                exclude_qid=str(qid),
                q_emb=q_emb,
            )
            candidates_map[qid] = [str(c["doc_id"]) for c in cands]
            preds_retrieval[qid] = self.selector.select(cands)
            runtimes_retrieval[qid] = time.time() - t_q0

        gold = {qid: self.qrels_map[qid] for qid in val_ids}
        retrieval_metrics = evaluate_predictions(
            y_pred=preds_retrieval,
            y_true=gold,
            candidate_pools=candidates_map,
            runtimes=runtimes_retrieval,
            cutoffs=DEFAULT_CANDIDATE_CUTOFFS,
        )
        retrieval_metrics["elapsed_seconds"] = time.time() - t0
        retrieval_metrics["val_queries"] = len(val_ids)

        dj_pair_mining_sec = 0.0
        dj_train_sec = 0.0
        dj_opt_steps = 0

        active_reranker = reranker
        doc_disjoint_adapter_dir: Path | None = None
        if active_reranker is None and self.train_reranker_per_fold:
            print("--- Training dedicated fold-safe LoRA reranker for Document-Disjoint split ---")
            doc_disjoint_dir = self.output_dir / "doc_disjoint"
            doc_disjoint_dir.mkdir(parents=True, exist_ok=True)
            pairs_dir = doc_disjoint_dir / "pairs"

            from src.training.build_pairs import build_training_pairs
            from src.training.train_reranker import train_reranker

            t_dj_pm0 = time.time()
            dj_train_list = sorted(str(x) for x in train_ids)
            dj_val_list = sorted(str(x) for x in val_ids)
            _, pairs_df = build_training_pairs(
                data_dir=self.data_dir,
                index_dir=self.index_dir,
                output_dir=pairs_dir,
                train_query_ids=dj_train_list,
                use_all_queries=False,
                limit=self.smoke_sample_size if self.smoke else None,
                query_embeddings=self.train_query_embeddings,
                duplicate_groups_path=self.duplicate_groups_path,
                static_branch_cache=self._static_branch_cache,
            )
            dj_pair_mining_sec = time.time() - t_dj_pm0

            dj_pair_qids = set(pairs_df["query_id"].astype(str)) if not pairs_df.empty else set()
            dj_train_set = set(map(str, dj_train_list))
            dj_val_set = set(map(str, dj_val_list))
            dj_unknown = dj_pair_qids - dj_train_set
            dj_leaked = dj_pair_qids & dj_val_set
            if dj_unknown or dj_leaked:
                raise AssertionError(
                    f"Doc-disjoint pair isolation failed: "
                    f"unknown={sorted(dj_unknown)[:10]}, leaked={sorted(dj_leaked)[:10]}"
                )
            from src.training.trainer import audit_pair_coverage as _audit_dj_pairs

            _dj_audit = _audit_dj_pairs(pairs_df, expected_qids=dj_train_set)
            if not self.smoke:
                if _dj_audit.get("missing_queries_count", 0) > 0:
                    raise AssertionError(
                        f"Doc-disjoint pair coverage failed: "
                        f"{_dj_audit.get('missing_queries_count')} expected training queries have no pairs."
                    )
                if float(_dj_audit.get("positive_coverage_pct", 0.0)) < 100.0:
                    raise AssertionError(
                        "Doc-disjoint requires 100% positive pair coverage, "
                        f"got {_dj_audit.get('positive_coverage_pct')}%"
                    )
                if float(_dj_audit.get("negative_coverage_pct", 0.0)) < 99.0:
                    raise AssertionError(
                        "Doc-disjoint requires >=99% negative pair coverage, "
                        f"got {_dj_audit.get('negative_coverage_pct')}%"
                    )

            # Strict doc-disjoint: exclude all held-out validation documents (and duplicate equivalents) from training pairs
            val_gold_docs: set[str] = set()
            for v_qid in val_ids:
                val_gold_docs.update(str(d) for d in self.qrels_map.get(str(v_qid), []))
            if self.duplicate_groups_path and self.duplicate_groups_path.exists():
                try:
                    dup_data = json.loads(self.duplicate_groups_path.read_text(encoding="utf-8"))
                    dup_groups = dup_data.get("duplicate_groups", [])
                    expanded_val_docs = set(val_gold_docs)
                    for grp in dup_groups:
                        grp_set = {str(x) for x in grp}
                        if grp_set & val_gold_docs:
                            expanded_val_docs.update(grp_set)
                    val_gold_docs = expanded_val_docs
                except Exception:
                    pass

            if not pairs_df.empty and "doc_id" in pairs_df.columns and val_gold_docs:
                leaked_doc_mask = pairs_df["doc_id"].astype(str).isin(val_gold_docs)
                if leaked_doc_mask.any():
                    print(f"[*] Removing {int(leaked_doc_mask.sum())} pairs exposing held-out documents in doc-disjoint training.")
                    pairs_df = pairs_df[~leaked_doc_mask].reset_index(drop=True)
                    pairs_df.to_parquet(pairs_dir / "reranker_pairs.parquet", index=False)

            doc_disjoint_adapter_dir = doc_disjoint_dir / "reranker_adapter"
            reranker_cfg = self.reranker_config_path or self.config_path or "configs/experiments/reranker_lora.yaml"
            base_m_name = self.reranker_model if self.reranker_model != "mock" else None

            t_dj_tr0 = time.time()
            dj_train_report = train_reranker(
                pairs_file=pairs_dir / "reranker_pairs.parquet",
                config_path=reranker_cfg,
                output_dir=doc_disjoint_adapter_dir,
                base_model_name=base_m_name,
                max_steps=5 if self.smoke else None,
                device=self.reranker_device,
                precision=self.precision,
                num_workers=self.num_workers,
                enforce_full_coverage_steps=not self.smoke,
            )
            dj_train_sec = time.time() - t_dj_tr0
            dj_opt_steps = int(dj_train_report.get("optimizer_steps", dj_train_report.get("global_steps", 0)))

            active_reranker = CrossEncoderReranker(
                model_name=self.reranker_model,
                adapter_path=doc_disjoint_adapter_dir,
                device=self.reranker_device,
                batch_size=self.reranker_batch_size,
                max_length=self.reranker_max_length,
                precision=self.precision,
                revision=dj_train_report.get("base_model_revision"),
            )

        # 2. Reranked pass under the fixed predeclared RRF policy, mirroring
        # public inference (predict.py): reranked candidates are fused with the
        # same fixed RRF weights before top-5 selection, so the disjoint score
        # evaluates the submission scoring policy instead of selector-only order.
        dj_ranker = _OOF_FIXED_RRF
        dj_train_doc_freq = compute_training_doc_frequencies(
            {qid: self.qrels_map[qid] for qid in train_ids if qid in self.qrels_map}
        )
        preds_system: dict[str, list[str]] = {}
        runtimes_system: dict[str, float] = {}

        t1 = time.time()
        window_size = min(32, max(1, self.reranker_batch_size))
        val_id_batches = [val_ids[i : i + window_size] for i in range(0, len(val_ids), window_size)]
        for batch_qids in tqdm(val_id_batches, desc="Doc-Disjoint System Eval", leave=False):
            window_items = []
            for qid in batch_qids:
                q_text = self.queries_map.get(qid, "")
                t_q0 = time.time()
                q_emb = self.train_query_embeddings.get(qid)

                branch_cands = None
                if self._static_branch_cache is not None:
                    if qid not in self._static_branch_cache:
                        exact_cands = self.exact.search(q_text, top_k=10) if self.exact else []
                        bm25_cands = self.bm25.search(q_text, top_k=max(80, self.candidate_k)) if self.bm25 else []
                        pyvi_cands = self.bm25_pyvi.search(q_text, top_k=max(80, self.candidate_k)) if self.bm25_pyvi else []
                        dense_cands = self.dense.retrieve(q_text, top_k=max(80, self.candidate_k), q_emb=q_emb) if self.dense else []
                        self._static_branch_cache[qid] = {
                            "exact": exact_cands,
                            "bm25": bm25_cands,
                            "bm25_pyvi": pyvi_cands,
                            "dense": dense_cands,
                            "bm25_80": bm25_cands[:80],
                            "pyvi_80": pyvi_cands[:80],
                            "dense_80": dense_cands[:80],
                        }
                    cached_s = self._static_branch_cache[qid]
                    branch_cands = {
                        "exact": cached_s.get("exact", []),
                        "bm25": cached_s.get("bm25", []),
                        "bm25_pyvi": cached_s.get("bm25_pyvi", []),
                        "dense": cached_s.get("dense", []),
                    }

                cands = hybrid_engine.search_candidates(
                    query=q_text,
                    top_k=self.candidate_k,
                    exclude_qid=str(qid),
                    q_emb=q_emb,
                    branch_candidates=branch_cands,
                )
                window_items.append((qid, q_text, cands, t_q0))

            if active_reranker is not None and self.evidence_builder is not None:
                q_cands = [(item[1], item[2]) for item in window_items]
                if hasattr(active_reranker, "rerank_batch"):
                    reranked_list = active_reranker.rerank_batch(
                        q_cands,
                        evidence_builder=self.evidence_builder,
                        top_k=self.rerank_k,
                        batch_size=self.reranker_batch_size,
                        max_length=self.reranker_max_length,
                    )
                else:
                    reranked_list = [
                        active_reranker.rerank(
                            query=q,
                            candidates=c,
                            evidence_builder=self.evidence_builder,
                            top_k=self.rerank_k,
                            batch_size=self.reranker_batch_size,
                            max_length=self.reranker_max_length,
                        )
                        for q, c in q_cands
                    ]
            else:
                reranked_list = [item[2] for item in window_items]

            for (qid, q_text, _, t_q0), cands in zip(window_items, reranked_list):
                ranked = dj_ranker.predict(
                    cands,
                    query_id=qid,
                    query_text=q_text,
                    doc_freq_map=dj_train_doc_freq,
                )
                top5 = self.selector.select(ranked)
                preds_system[qid] = top5
                runtimes_system[qid] = time.time() - t_q0

        dj_infer_sec = time.time() - t1

        trained_system_metrics = evaluate_predictions(
            y_pred=preds_system,
            y_true=gold,
            candidate_pools=candidates_map,
            runtimes=runtimes_system,
            cutoffs=DEFAULT_CANDIDATE_CUTOFFS,
        )
        trained_system_metrics["elapsed_seconds"] = time.time() - t1
        trained_system_metrics["val_queries"] = len(val_ids)

        assert_official_equivalence(preds_system, gold)

        final_report = {
            "retrieval_only": retrieval_metrics,
            "trained_reranker_system": trained_system_metrics,
            "recall@5": trained_system_metrics["recall@5"],
            "precision@5": trained_system_metrics["precision@5"],
            "mrr": trained_system_metrics.get("mrr", 0.0),
            "map": trained_system_metrics.get("map", 0.0),
            "ndcg@5": trained_system_metrics.get("ndcg@5", 0.0),
            "doc_disjoint_pair_mining_seconds": round(dj_pair_mining_sec, 3),
            "doc_disjoint_training_seconds": round(dj_train_sec, 3),
            "doc_disjoint_optimizer_steps": dj_opt_steps,
            "doc_disjoint_inference_seconds": round(dj_infer_sec, 3),
            "adapter_path": str(doc_disjoint_adapter_dir) if doc_disjoint_adapter_dir else None,
            "fusion_policy": "predeclared_rrf",
            "is_smoke_mode": self.smoke,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        report_path = self.output_dir / "doc_disjoint_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

        doc_disjoint_dir.mkdir(parents=True, exist_ok=True)
        _dj_split_sha = None
        try:
            if isinstance(self.split_provenance, dict):
                _dj_prov = (self.split_provenance.get("doc_disjoint") or {})
                if isinstance(_dj_prov, dict) and _dj_prov.get("sha256"):
                    _dj_split_sha = str(_dj_prov.get("sha256"))
        except Exception:
            pass
        dj_complete_marker.write_text(
            json.dumps({
                "status": "COMPLETED",
                "recall@5": trained_system_metrics["recall@5"],
                "precision@5": trained_system_metrics["precision@5"],
                "reranker_model": str(self.reranker_model),
                "smoke": bool(self.smoke),
                "split_doc_disjoint_sha256": _dj_split_sha,
                "timestamp": time.time(),
            }, indent=2),
            encoding="utf-8",
        )

        self.doc_disjoint_report = final_report
        print(
            f"Document-Disjoint Split: Retrieval Recall@5 = {retrieval_metrics['recall@5'] * 100:.2f}% | "
            f"Trained System Recall@5 = {trained_system_metrics['recall@5'] * 100:.2f}% | "
            f"Precision@5 = {trained_system_metrics['precision@5'] * 100:.2f}%"
        )
        return final_report

    def run_fusion_evaluation(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        """Run cross-fitted fusion evaluation on generated oof_features.parquet."""
        from src.ranking.train_fusion import train_and_evaluate_fusion_cv

        oof_feat_path = self.output_dir / "oof_features.parquet"
        if not oof_feat_path.exists():
            raise FileNotFoundError(f"OOF features not found: {oof_feat_path}")
        oof_df = pd.read_parquet(oof_feat_path)
        if oof_df.empty or "fold" not in oof_df.columns:
            raise ValueError(f"OOF features DataFrame at {oof_feat_path} is empty or missing 'fold' column.")
        qrels_dict = self.qrels_map
        fusion_out = Path(output_dir) if output_dir else self.output_dir / "fusion"
        return train_and_evaluate_fusion_cv(oof_df=oof_df, qrels_dict=qrels_dict, output_dir=fusion_out)

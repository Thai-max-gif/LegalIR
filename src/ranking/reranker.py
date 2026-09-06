from collections import defaultdict
from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.core.paths import ProjectPaths
from src.models.device import resolve_device
from src.ranking.evidence_pack import EvidencePackBuilder
from src.retrieval.types import CandidateRecord


class CrossEncoderReranker:
    """Batch cross-encoder reranker for candidate legal documents."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        score_fn: Callable[..., list[float]] | None = None,
        *,
        model_path: str | Path | None = None,
        adapter_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        local_files_only: bool | None = None,
    ):
        self.model_name = str(model_name)
        self.adapter_path = Path(adapter_path).expanduser() if adapter_path is not None else None
        self.model_path = self._resolve_model_path(model_path, manifest_path)
        self.local_files_only = (
            self.model_path is not None
            if local_files_only is None
            else bool(local_files_only)
        )
        self.device = (
            resolve_device(device or "auto")
            if device != "cpu" and self.model_name != "mock"
            else "cpu"
        )
        self.tokenizer = None
        self.model = None
        self.score_fn = score_fn
        self.oom_events: int = 0
        self.initial_batch_size: int = 16
        self.min_successful_batch_size: int = 16
        self.last_successful_batch_size: int = 16

    def _resolve_model_path(
        self,
        model_path: str | Path | None,
        manifest_path: str | Path | None,
    ) -> Path | None:
        """Resolve an explicit path or bootstrap manifest entry if present."""
        if model_path is not None:
            path = Path(model_path).expanduser()
            return path if path.is_dir() else None

        requested_path = Path(self.model_name).expanduser()
        if requested_path.is_dir():
            return requested_path

        manifest = Path(manifest_path).expanduser() if manifest_path else (
            ProjectPaths.from_repo().local_models / "huggingface" / "manifest.json"
        )
        if not manifest.is_file():
            return None
        try:
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            entry = manifest_data.get(self.model_name, {})
            manifest_model_path = entry.get("path") if isinstance(entry, Mapping) else None
        except (OSError, ValueError, TypeError):
            return None
        if not manifest_model_path:
            return None
        path = Path(manifest_model_path).expanduser()
        if not path.is_absolute():
            path = manifest.parent / path
        return path if path.is_dir() else None

    def ensure_loaded(self) -> None:
        """Explicitly instantiate and load tokenizer and model onto device."""
        self._load_model()

    def _load_model(self) -> None:
        if self.model is not None or self.score_fn is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Check if an adapter checkpoint is specified or present in model_path
        adapter_dir = None
        if self.adapter_path is not None and Path(self.adapter_path).is_dir():
            adapter_dir = Path(self.adapter_path)
        elif self.model_path is not None and (Path(self.model_path) / "adapter_config.json").is_file():
            adapter_dir = Path(self.model_path)

        load_kwargs = {"local_files_only": self.local_files_only}

        if adapter_dir is not None:
            # PEFT LoRA adapter checkpoint loading
            adapter_config_file = adapter_dir / "adapter_config.json"
            base_model_source = self.model_name
            if adapter_config_file.is_file():
                try:
                    cfg_data = json.loads(adapter_config_file.read_text(encoding="utf-8"))
                    if "base_model_name_or_path" in cfg_data and cfg_data["base_model_name_or_path"]:
                        # Use base model from adapter config unless user provided custom model_name
                        if self.model_name == "BAAI/bge-reranker-v2-m3" or self.model_name == "":
                            base_model_source = cfg_data["base_model_name_or_path"]
                except Exception:
                    pass

            manifest_file = adapter_dir / "training_manifest.json"
            if manifest_file.is_file():
                try:
                    m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    if "base_model" in m_data and m_data["base_model"]:
                        base_model_source = m_data["base_model"]
                except Exception:
                    pass

            if base_model_source == "mock":
                import tempfile
                from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
                config = BertConfig(
                    vocab_size=300,
                    hidden_size=32,
                    num_attention_heads=2,
                    num_hidden_layers=2,
                    intermediate_size=64,
                    max_position_embeddings=128,
                    num_labels=1,
                )
                self.model = BertForSequenceClassification(config)
                tmp_vocab = Path(tempfile.gettempdir()) / "mock_vocab.txt"
                if not tmp_vocab.exists():
                    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(295)]
                    tmp_vocab.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
                self.tokenizer = BertTokenizerFast(vocab_file=str(tmp_vocab))
            else:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), **load_kwargs)
                except Exception:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(base_model_source, **load_kwargs)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load tokenizer for real base model '{base_model_source}': {e}") from e

                try:
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_source,
                        num_labels=1,
                        **load_kwargs,
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to load real base model '{base_model_source}': {e}") from e

                from peft import PeftModel

                self.model = PeftModel.from_pretrained(base_model, str(adapter_dir), **load_kwargs)
        elif self.model_name == "mock":
            import tempfile
            from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
            config = BertConfig(
                vocab_size=300,
                hidden_size=32,
                num_attention_heads=2,
                num_hidden_layers=2,
                intermediate_size=64,
                max_position_embeddings=128,
                num_labels=1,
            )
            self.model = BertForSequenceClassification(config)
            tmp_vocab = Path(tempfile.gettempdir()) / "mock_vocab.txt"
            if not tmp_vocab.exists():
                vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(295)]
                tmp_vocab.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
            self.tokenizer = BertTokenizerFast(vocab_file=str(tmp_vocab))
        else:
            model_source = str(self.model_path) if self.model_path is not None else self.model_name
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer for real model '{model_source}': {e}") from e

            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_source,
                    num_labels=1,
                    **load_kwargs,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load real model '{model_source}': {e}") from e

        self.model.to(self.device)
        self.model.eval()

    def _score_with_callback(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int,
        max_length: int,
    ) -> list[float]:
        if self.score_fn is None:
            raise RuntimeError("score callback is not configured")
        try:
            result = self.score_fn(pairs, batch_size=batch_size, max_length=max_length)
        except TypeError as exc:
            # Preserve compatibility with simple test doubles accepting only
            # the pair list, without swallowing errors from the callback body.
            message = str(exc)
            if "batch_size" not in message and "max_length" not in message and "argument" not in message:
                raise
            result = self.score_fn(pairs)
        scores = [float(score) for score in np.asarray(result).reshape(-1).tolist()]
        if len(scores) != len(pairs):
            raise ValueError(
                f"score callback returned {len(scores)} scores for {len(pairs)} pairs"
            )
        return scores

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        if isinstance(inputs, Mapping):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        return inputs

    def score_pairs(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 16,
        max_length: int = 512,
    ) -> list[float]:
        """Score ``(query, passage)`` pairs using deterministic mini-batches."""
        if not pairs:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")

        if self.score_fn is not None:
            return self._score_with_callback(pairs, batch_size, max_length)

        self._load_model()
        import torch

        self.initial_batch_size = batch_size
        current_batch = batch_size
        all_scores: list[float] = []
        idx = 0
        n_pairs = len(pairs)

        while idx < n_pairs:
            batch = pairs[idx : idx + current_batch]
            queries = [str(pair[0]) for pair in batch]
            passages = [str(pair[1]) for pair in batch]
            try:
                inputs = self.tokenizer(
                    queries,
                    passages,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = self._move_inputs_to_device(inputs)
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                batch_scores = logits.reshape(-1).float().cpu().tolist()
                if len(batch_scores) != len(batch):
                    raise ValueError(
                        f"model returned {len(batch_scores)} scores for {len(batch)} pairs"
                    )
                all_scores.extend(float(score) for score in batch_scores)
                self.min_successful_batch_size = min(self.min_successful_batch_size, len(batch))
                self.last_successful_batch_size = len(batch)
                idx += len(batch)
            except RuntimeError as exc:
                message = str(exc).lower()
                if ("out of memory" in message or "cuda error: out of memory" in message or "mps" in message):
                    self.oom_events += 1
                    if current_batch == 1:
                        raise
                    current_batch = max(1, current_batch // 2)
                    self.min_successful_batch_size = min(self.min_successful_batch_size, current_batch)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
        return all_scores

    def aggregate_document(
        self,
        doc_id: str,
        chunk_records: list[dict[str, Any]],
        chunk_scores: list[float],
    ) -> dict[str, Any]:
        """Aggregate chunk-level scores into document-level ranking features."""
        doc_id = str(doc_id)
        if not chunk_scores:
            return {
                "doc_id": doc_id,
                "reranker_score": 0.0,
                "reranker_best_score": 0.0,
                "reranker_second_score": 0.0,
                "reranker_margin": 0.0,
                "reranker_best_chunk_id": None,
                "evidence_chunk_count": 0,
            }

        scored_chunks = list(zip(chunk_records, (float(score) for score in chunk_scores)))
        scored_chunks.sort(key=lambda item: item[1], reverse=True)
        best_chunk, best_score = scored_chunks[0]
        second_score = scored_chunks[1][1] if len(scored_chunks) > 1 else best_score

        return {
            "doc_id": doc_id,
            "reranker_score": float(best_score),
            "reranker_best_score": float(best_score),
            "reranker_second_score": float(second_score),
            "reranker_margin": float(best_score - second_score),
            "reranker_best_chunk_id": str(best_chunk.get("chunk_id", "")),
            "evidence_chunk_count": len(chunk_scores),
        }

    @staticmethod
    def _candidate_record(candidate: Any) -> dict[str, Any]:
        if isinstance(candidate, Mapping):
            if "doc_id" not in candidate:
                raise ValueError("candidate record must contain doc_id")
            record = dict(candidate)
            record["doc_id"] = str(record["doc_id"])
            return record
        if isinstance(candidate, (tuple, list)):
            if not candidate:
                raise ValueError("candidate tuple must contain a document ID")
            record = {"doc_id": str(candidate[0])}
            if len(candidate) > 1:
                record["candidate_score"] = float(candidate[1])
            return record
        if candidate is None:
            raise ValueError("candidate document ID cannot be null")
        return {"doc_id": str(candidate)}

    def rerank(
        self,
        query: str,
        candidates: list[CandidateRecord] | list[tuple[Any, float]] | list[Any],
        evidence_builder: EvidencePackBuilder | None = None,
        top_k: int = 50,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> list[CandidateRecord]:
        """Rerank the first ``top_k`` candidates by BGE cross-encoder score."""
        if not candidates or not query:
            return candidates
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        normalized_candidates = [self._candidate_record(candidate) for candidate in candidates]
        target_candidates = normalized_candidates[:top_k]
        remaining_candidates = normalized_candidates[top_k:]

        all_pairs: list[tuple[str, str]] = []
        pair_to_doc: list[tuple[str, dict[str, Any]]] = []
        for candidate in target_candidates:
            doc_id = str(candidate["doc_id"])
            if evidence_builder is None:
                records = [{
                    "chunk_id": f"{doc_id}_fallback",
                    "reranker_text": f"[DOCUMENT] {doc_id} [EVIDENCE 1] {doc_id}",
                }]
            else:
                records = evidence_builder.build(
                    query,
                    doc_id,
                    candidate_record=candidate,
                )
            for index, record in enumerate(records):
                # The first pair is the complete multi-evidence document pack;
                # additional pairs retain chunk-level evidence for aggregation.
                passage = (
                    record.get("pack")
                    if index == 0 and record.get("pack")
                    else record.get("reranker_text")
                    or record.get("text")
                    or record.get("chunk_text", "")
                )
                all_pairs.append((str(query), str(passage)))
                pair_to_doc.append((doc_id, record))

        all_scores = self.score_pairs(all_pairs, batch_size=batch_size, max_length=max_length)

        doc_scores_map: dict[str, list[float]] = defaultdict(list)
        doc_records_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (doc_id, record), score in zip(pair_to_doc, all_scores):
            doc_scores_map[doc_id].append(float(score))
            doc_records_map[doc_id].append(record)

        reranked_target: list[CandidateRecord] = []
        for candidate in target_candidates:
            doc_id = str(candidate["doc_id"])
            updated_candidate = dict(candidate)
            updated_candidate.update(
                self.aggregate_document(
                    doc_id,
                    doc_records_map.get(doc_id, []),
                    doc_scores_map.get(doc_id, []),
                )
            )
            reranked_target.append(updated_candidate)

        reranked_target.sort(
            key=lambda candidate: (
                -float(candidate["reranker_best_score"]),
                str(candidate["doc_id"]),
            )
        )

        # Candidates outside the reranker budget retain their input order and
        # receive explicit low scores so they cannot outrank scored records.
        for candidate in remaining_candidates:
            candidate.update({
                "reranker_score": -999.0,
                "reranker_best_score": -999.0,
                "reranker_second_score": -999.0,
                "reranker_margin": 0.0,
                "reranker_best_chunk_id": None,
                "evidence_chunk_count": 0,
            })
        return reranked_target + remaining_candidates

    def rerank_pairs(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int | None = None,
        max_length: int = 256,
    ) -> np.ndarray:
        """Score pairs and return a NumPy array of float32 scores."""
        if not pairs:
            return np.array([], dtype=np.float32)
        if self.model_name == "mock" and self.score_fn is None:
            scores = []
            for q, doc in pairs:
                q_words = set(str(q).lower().split())
                doc_words = set(str(doc).lower().split())
                overlap = len(q_words & doc_words)
                scores.append(float(overlap))
            return np.array(scores, dtype=np.float32)
        scores = self.score_pairs(pairs, batch_size=batch_size or 16, max_length=max_length)
        return np.array(scores, dtype=np.float32)

    def rerank_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        evidence_texts: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank candidate list using pairs constructed from evidence texts."""
        if not candidates:
            return []
        if evidence_texts is None:
            evidence_texts = [c.get("evidence_text") or c.get("text_raw", "") for c in candidates]
        pairs = [(query, str(text)[:1000]) for text in evidence_texts]
        scores = self.rerank_pairs(pairs)
        scored = []
        for item, sc in zip(candidates, scores):
            entry = dict(item)
            entry["reranker_score"] = float(sc)
            scored.append(entry)
        scored.sort(key=lambda x: -x["reranker_score"])
        for rank, item in enumerate(scored[:top_k], start=1):
            item["final_rank"] = rank
        return scored[:top_k]


# Backward-compatibility alias
BGEReranker = CrossEncoderReranker


class DocumentReranker:
    """Convenience wrapper for document-level reranking with evidence builder."""

    def __init__(
        self,
        reranker: CrossEncoderReranker | None = None,
        evidence_builder: EvidencePackBuilder | None = None,
        doc_map: dict[str, Any] | None = None,
        chunk_map: dict[str, list[dict[str, Any]]] | None = None,
    ):
        self.reranker = reranker
        self.evidence_builder = evidence_builder or EvidencePackBuilder()
        self.doc_map = doc_map or {}
        self.chunk_map = chunk_map or {}

    def rerank_documents(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not candidates or self.reranker is None:
            return candidates[:top_k]

        evidence_texts = []
        valid_candidates = []

        for c in candidates:
            doc_id = str(c.get("doc_id", ""))
            doc_info = self.doc_map.get(doc_id, {"doc_id": doc_id})
            chunks = self.chunk_map.get(doc_id, [])

            if not chunks and "best_chunk" in c:
                chunks = [c["best_chunk"]]

            if hasattr(self.evidence_builder, "build_pack") and (doc_id in getattr(self.evidence_builder, "chunks_by_doc", {}) or not chunks):
                ev_text = self.evidence_builder.build_pack(query, doc_id, candidate_record=c)
            else:
                ev_text = self.evidence_builder.build_evidence_text(query, doc_info, chunks)
            evidence_texts.append(ev_text)
            valid_candidates.append(c)

        reranked = self.reranker.rerank_candidates(
            query, valid_candidates, evidence_texts=evidence_texts, top_k=top_k
        )
        return reranked



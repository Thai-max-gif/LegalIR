"""Dense semantic retrieval over macro legal-document chunks."""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
import unicodedata
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.device import resolve_device

DEFAULT_MODEL_NAME = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
DEFAULT_DIMENSION = 768

_MUTABLE_REVISIONS = {"", "main", "master", "latest", "default"}


def is_mutable_revision(rev: Any) -> bool:
    """Mutable refs (``main``/missing) never establish immutable provenance."""
    if rev is None:
        return True
    s = str(rev).strip()
    if not s:
        return True
    if s.lower() in _MUTABLE_REVISIONS:
        return True
    # Immutable pins are 40-char lowercase hex SHAs.
    if len(s) != 40:
        return True
    return not all(c in "0123456789abcdef" for c in s.lower())


def pinned_dense_revision(model_name: str) -> str | None:
    """Registry pin for a dense model id, or ``None`` when unknown."""
    try:
        from src.models.bootstrap import MODEL_REGISTRY

        entry = MODEL_REGISTRY.get(str(model_name), {})
        rev = entry.get("revision") if isinstance(entry, dict) else None
        rev_s = str(rev).strip() if rev else None
        if rev_s and not is_mutable_revision(rev_s):
            return rev_s
        return None
    except Exception:
        return None


def resolve_dense_revision(model_name: str, explicit: Any = None) -> str | None:
    """Resolve the revision to record/use.

    ``mock`` needs no revision. An explicit value (even mutable) is returned
    as-is so provenance checks fail closed instead of silently substituting the
    registry pin and relabeling unknown weights. When explicit is ``None``,
    the registry pin is used; unknown models yield ``None``.
    """
    if str(model_name) == "mock":
        return None
    if explicit is not None:
        s = str(explicit).strip()
        return s or None
    return pinned_dense_revision(str(model_name))


def chunk_ids_digest(chunk_ids: Sequence[Any]) -> str:
    import hashlib

    h = hashlib.sha256()
    for cid in chunk_ids:
        h.update(str(cid).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass
class DenseEncodeTelemetry:
    """Detailed telemetry for a single encode_texts/encode_queries execution."""
    requested_batch_size: int
    min_successful_batch_size: int | None
    last_successful_batch_size: int | None
    oom_events: int
    item_count: int
    elapsed_seconds: float


class DenseMacroRetriever:
    """Encode macro chunks and aggregate dense similarities by document.

    The DEk21 v2 model is used by default. Encoded vectors are mean-pooled over
    non-padding tokens and L2-normalized, so inner products are cosine scores.
    FAISS is used when installed; NumPy provides the same search semantics as a
    dependency-free fallback.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dimension: int = DEFAULT_DIMENSION,
        use_pyvi: bool = True,
        device: str | None = None,
        model_name_or_path: str | None = None,
        revision: str | None = None,
    ):
        if model_name_or_path is not None:
            model_name = model_name_or_path

        self.model_name = str(model_name)
        # Keep the old attribute for callers using the pre-DEk21 interface.
        self.model_name_or_path = self.model_name
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        self.use_pyvi = bool(use_pyvi)
        self.device = resolve_device(device or "auto")
        self.revision = str(revision) if revision is not None else None

        self.tokenizer = None
        self.model = None
        self.chunk_ids: list[str] = []
        self.doc_ids: list[str] = []
        self.corpus: list[dict[str, Any]] = []
        self.query_ids: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.query_encoder: Callable[[list[str]], np.ndarray] | None = None
        self._faiss_index = None
        self.dense_oom_events: int = 0
        self.dense_initial_batch_size: int = 32
        self.dense_min_successful_batch_size: int | None = None
        self.last_encode_telemetry: DenseEncodeTelemetry | None = None
        self.stage_telemetry: dict[str, DenseEncodeTelemetry] = {}

    @classmethod
    def from_arrays(
        cls,
        embeddings_path: str | Path | None = None,
        chunk_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
        query_encoder: Callable[[list[str]], np.ndarray] | None = None,
        embeddings: np.ndarray | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        use_pyvi: bool = False,
    ) -> "DenseMacroRetriever":
        """Create a retriever from a precomputed embedding matrix.

        This is useful for loading a local index or supplying a deterministic
        encoder in tests without downloading a model.
        """
        if embeddings is not None and embeddings_path is not None:
            raise ValueError("provide embeddings or embeddings_path, not both")

        loaded_embeddings: np.ndarray | None = None
        if embeddings is not None:
            loaded_embeddings = np.asarray(embeddings)
        elif embeddings_path is not None:
            loaded_embeddings = np.load(str(embeddings_path), mmap_mode="r")

        dimension = (
            int(loaded_embeddings.shape[1])
            if loaded_embeddings is not None and loaded_embeddings.ndim == 2
            else DEFAULT_DIMENSION
        )
        retriever = cls(
            model_name=model_name,
            dimension=dimension,
            use_pyvi=use_pyvi,
            device="cpu",
        )
        if loaded_embeddings is not None:
            retriever._set_embeddings(loaded_embeddings)
        retriever.chunk_ids = [str(x) for x in (chunk_ids or [])]
        retriever.doc_ids = [str(x) for x in (doc_ids or [])]
        retriever.query_encoder = query_encoder
        retriever._validate_metadata_lengths()
        return retriever

    @staticmethod
    def _normalize_text(text: Any) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        text = unicodedata.normalize("NFC", str(text))
        text = re.sub(r"[ \t\r\n]+", " ", text)
        return text.strip()

    def preprocess_text(self, text: Any) -> str:
        """Normalize text and optionally apply PyVi Vietnamese segmentation."""
        normalized = self._normalize_text(text)
        if not normalized or not self.use_pyvi:
            return normalized

        # If text is already PyVi word-segmented (e.g. from canonical text_norm), skip redundant re-tokenization
        if "_" in normalized:
            return normalized

        # Pathological inputs hang ViTokenizer (observed: 5.7M-char anomalous
        # chunk stalls segmentation indefinitely). Return unsegmented text;
        # the encoder truncates to max_length downstream.
        from src.retrieval.bm25_pyvi import PYVI_MAX_CHARS
        if len(normalized) > PYVI_MAX_CHARS:
            return normalized

        try:
            from pyvi import ViTokenizer
        except ImportError as exc:
            raise ImportError(
                "PyVi is required when use_pyvi=True; install it with `pip install pyvi`."
            ) from exc
        return ViTokenizer.tokenize(normalized)

    def ensure_loaded(self) -> None:
        """Explicitly load tokenizer and model onto device."""
        self._load_model()

    def _load_model(self) -> None:
        if self.model_name == "mock":
            if self.tokenizer is None:
                import tempfile
                from transformers import BertTokenizerFast
                tmp_vocab = Path(tempfile.gettempdir()) / "mock_dense_vocab.txt"
                if not tmp_vocab.exists():
                    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(295)]
                    tmp_vocab.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
                self.tokenizer = BertTokenizerFast(vocab_file=str(tmp_vocab))
            if self.model is None:
                from transformers import BertConfig, BertModel
                config = BertConfig(
                    vocab_size=300,
                    hidden_size=self.dimension,
                    num_attention_heads=2,
                    num_hidden_layers=2,
                    intermediate_size=64,
                    max_position_embeddings=512,
                )
                self.model = BertModel(config)
            self.model.to(self.device)
            self.model.eval()
            return
        if self.tokenizer is not None and self.model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading dense model {self.model_name} on {self.device}...")
        revision = getattr(self, "revision", None)
        if revision is None:
            try:
                from src.models.bootstrap import MODEL_REGISTRY
                if self.model_name in MODEL_REGISTRY:
                    revision = MODEL_REGISTRY[self.model_name].get("revision")
            except Exception:
                pass

        kwargs: dict[str, Any] = {}
        if revision:
            kwargs["revision"] = revision

        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load tokenizer for real dense model '{self.model_name}' (revision={revision}): {exc}"
                ) from exc

        if self.model is None:
            try:
                self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load real dense model '{self.model_name}' (revision={revision}): {exc}"
                ) from exc

        self.model.to(self.device)
        self.model.eval()

    def unload_model(self) -> None:
        """Explicitly release dense transformer model and free device memory."""
        import gc
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        if isinstance(inputs, Mapping):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        return inputs

    def _coerce_embedding_matrix(self, values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("encoder output must be a two-dimensional matrix")
        if matrix.shape[1] != self.dimension:
            # A supplied encoder is intentionally allowed to define its own
            # dimensionality for in-memory/test indexes.
            if self.query_encoder is not None and self.model is None:
                self.dimension = int(matrix.shape[1])
            else:
                raise ValueError(
                    f"encoder dimension {matrix.shape[1]} does not match configured "
                    f"dimension {self.dimension}"
                )
        return self._normalize_embeddings(matrix)

    @staticmethod
    def _normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)

    def encode_texts(
        self,
        texts: Iterable[Any],
        batch_size: int = 32,
        max_length: int = 512,
        stage_name: str | None = None,
    ) -> np.ndarray:
        """Encode texts with mean pooling and normalized embeddings."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        t_call_0 = time.time()
        if isinstance(texts, str):
            texts = [texts]
        normalized_texts = [self.preprocess_text(text) for text in texts]
        if not normalized_texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.query_encoder is not None:
            res = self._coerce_embedding_matrix(self.query_encoder(normalized_texts))
            telemetry = DenseEncodeTelemetry(
                requested_batch_size=batch_size,
                min_successful_batch_size=batch_size,
                last_successful_batch_size=batch_size,
                oom_events=0,
                item_count=len(normalized_texts),
                elapsed_seconds=time.time() - t_call_0,
            )
            self.last_encode_telemetry = telemetry
            if stage_name:
                self.stage_telemetry[stage_name] = telemetry
            return res

        if self.model_name == "mock":
            np.random.seed(42)
            emb = np.random.randn(len(normalized_texts), self.dimension).astype(np.float32)
            telemetry = DenseEncodeTelemetry(
                requested_batch_size=batch_size,
                min_successful_batch_size=batch_size,
                last_successful_batch_size=batch_size,
                oom_events=0,
                item_count=len(normalized_texts),
                elapsed_seconds=time.time() - t_call_0,
            )
            self.last_encode_telemetry = telemetry
            if stage_name:
                self.stage_telemetry[stage_name] = telemetry
            return self._coerce_embedding_matrix(emb)

        self._load_model()

        import torch
        import torch.nn.functional as F

        is_cuda = str(self.device).startswith("cuda")
        all_vectors: list[np.ndarray] = []
        self.dense_initial_batch_size = batch_size
        curr_batch_size = batch_size
        call_oom_events = 0
        min_successful: int | None = None
        last_successful: int | None = None

        # Ensure max_length never exceeds model positional embedding table
        pos_emb_val = getattr(getattr(self.model, "config", None), "max_position_embeddings", 512)
        try:
            pos_emb = int(pos_emb_val)
        except (TypeError, ValueError):
            pos_emb = 512
        effective_max_length = min(max_length, pos_emb - 2 if pos_emb <= 514 else pos_emb)
        if effective_max_length <= 0:
            effective_max_length = 256

        idx = 0
        while idx < len(normalized_texts):
            curr_chunk = normalized_texts[idx : idx + curr_batch_size]
            try:
                inputs = self.tokenizer(
                    curr_chunk,
                    padding=True,
                    truncation=True,
                    max_length=effective_max_length,
                    return_tensors="pt",
                )
                inputs = self._move_inputs_to_device(inputs)

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                        outputs = self.model(**inputs)
                        hidden_state = getattr(outputs, "last_hidden_state", None)
                        if hidden_state is None:
                            hidden_state = outputs[0]
                        attention_mask = inputs.get("attention_mask")
                        if attention_mask is None:
                            attention_mask = torch.ones(
                                hidden_state.shape[:2], dtype=hidden_state.dtype, device=hidden_state.device
                            )
                        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
                        pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
                        normalized = F.normalize(pooled, p=2, dim=1)
                        vectors = normalized.cpu().to(torch.float32).numpy()

                all_vectors.append(vectors)
                min_successful = min(min_successful, len(curr_chunk)) if min_successful is not None else len(curr_chunk)
                last_successful = len(curr_chunk)
                idx += len(curr_chunk)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if ("out of memory" in msg or "cuda error: out of memory" in msg or "mps" in msg):
                    self.dense_oom_events += 1
                    call_oom_events += 1
                    if is_cuda and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if curr_batch_size <= 1:
                        raise RuntimeError(f"Dense encoding failed with OOM even at batch_size=1: {exc}") from exc
                    curr_batch_size = max(1, curr_batch_size // 2)
                    print(f"[-] Dense CUDA OOM event #{self.dense_oom_events}: adapting batch size to {curr_batch_size}")
                else:
                    raise

        self.dense_min_successful_batch_size = min_successful
        telemetry = DenseEncodeTelemetry(
            requested_batch_size=batch_size,
            min_successful_batch_size=min_successful,
            last_successful_batch_size=last_successful,
            oom_events=call_oom_events,
            item_count=len(normalized_texts),
            elapsed_seconds=time.time() - t_call_0,
        )
        self.last_encode_telemetry = telemetry
        if stage_name:
            self.stage_telemetry[stage_name] = telemetry

        return self._coerce_embedding_matrix(np.vstack(all_vectors))

    @staticmethod
    def _records(corpus: Any) -> list[dict[str, Any]]:
        if isinstance(corpus, Path) or (
            isinstance(corpus, str) and Path(corpus).exists()
        ):
            corpus = pd.read_parquet(corpus)
        elif isinstance(corpus, str):
            corpus = [{"text": corpus}]
        if isinstance(corpus, pd.DataFrame):
            if "granularity" in corpus.columns:
                corpus = corpus[corpus["granularity"] == "macro"]
            records = corpus.to_dict(orient="records")
        elif isinstance(corpus, Mapping):
            records = [dict(corpus)]
        else:
            records = [dict(record) if isinstance(record, Mapping) else {"text": record} for record in corpus]

        if any("granularity" in record for record in records):
            records = [record for record in records if record.get("granularity", "macro") == "macro"]
        return records

    @staticmethod
    def _record_text(record: Mapping[str, Any]) -> str:
        for key in ("text_norm", "text", "text_raw", "body"):
            value = record.get(key)
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                return str(value)
        return ""

    def _set_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")
        if matrix.shape[1] != self.dimension:
            self.dimension = int(matrix.shape[1])
        self.embeddings = self._normalize_embeddings(matrix)
        self._build_search_index()
        return self.embeddings

    def _build_search_index(self) -> None:
        self._faiss_index = None
        if self.embeddings is None or len(self.embeddings) == 0:
            return
        try:
            import faiss
            index = faiss.IndexFlatIP(self.dimension)
            index.add(self.embeddings.astype(np.float32, copy=False))
            self._faiss_index = index
        except ImportError:
            self._faiss_index = None

    def _validate_metadata_lengths(self) -> None:
        if self.embeddings is None:
            return
        count = len(self.embeddings)
        if self.chunk_ids and len(self.chunk_ids) != count:
            raise ValueError("chunk_ids length must match embeddings")
        if self.doc_ids and len(self.doc_ids) != count:
            raise ValueError("doc_ids length must match embeddings")

    def encode_corpus(
        self,
        corpus: Any,
        batch_size: int = 32,
        max_length: int = 512,
        stage_name: str = "corpus",
    ) -> np.ndarray:
        """Encode macro records and retain chunk/document metadata for search."""
        records = self._records(corpus)
        texts = [self._record_text(record) for record in records]
        embeddings = self.encode_texts(texts, batch_size=batch_size, max_length=max_length, stage_name=stage_name)

        self.chunk_ids = [str(record.get("chunk_id", index)) for index, record in enumerate(records)]
        self.doc_ids = [
            str(record.get("doc_id", record.get("document_id", self.chunk_ids[index])))
            for index, record in enumerate(records)
        ]
        self.corpus = [
            {
                "chunk_id": self.chunk_ids[index],
                "doc_id": self.doc_ids[index],
                "article": record.get("article", record.get("article_id")),
            }
            for index, record in enumerate(records)
        ]
        self._set_embeddings(embeddings)
        return self.embeddings

    def encode_queries(
        self,
        queries: Any,
        batch_size: int = 32,
        max_length: int = 512,
        stage_name: str = "train_query",
    ) -> np.ndarray:
        """Encode query strings or query records from ``queries_train.parquet``."""
        if isinstance(queries, Path) or (
            isinstance(queries, str) and Path(queries).exists()
        ):
            queries = pd.read_parquet(queries)
        elif isinstance(queries, str):
            queries = [queries]
        if isinstance(queries, pd.DataFrame):
            records = queries.to_dict(orient="records")
        elif isinstance(queries, Mapping):
            records = [dict(queries)]
        else:
            records = list(queries)

        texts: list[Any] = []
        query_ids: list[str] = []
        for index, query in enumerate(records):
            if isinstance(query, Mapping):
                texts.append(
                    query.get("question_norm")
                    or query.get("question_raw")
                    or query.get("text")
                    or query.get("query", "")
                )
                query_ids.append(str(query.get("query_id", index)))
            else:
                texts.append(query)
                query_ids.append(str(index))

        self.query_ids = query_ids
        return self.encode_texts(texts, batch_size=batch_size, max_length=max_length, stage_name=stage_name)

    def encode_and_cache_queries(
        self,
        queries: Any,
        cache_path: str | Path,
        batch_size: int = 32,
        max_length: int = 512,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """Encode queries and cache to disk, or load from cache if already present."""
        cache_path = Path(cache_path)
        if cache_path.exists():
            embeddings = np.load(str(cache_path))
            return self._normalize_embeddings(embeddings)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = self.encode_queries(queries, batch_size=batch_size, max_length=max_length)
        np.save(str(cache_path), embeddings.astype(dtype))
        return embeddings

    def search(
        self,
        query: str,
        top_k: int = 50,
        q_emb: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Return top documents with scores aggregated from their best chunks.

        Accepts either a raw query string or a precomputed query vector ``q_emb``.
        """
        if top_k <= 0 or self.embeddings is None or len(self.embeddings) == 0:
            return []
        self._validate_metadata_lengths()

        if q_emb is not None:
            q_vec = np.asarray(q_emb, dtype=np.float32)
            if q_vec.ndim == 2 and q_vec.shape[0] == 1:
                q_vec = q_vec[0]
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
        else:
            if not query:
                return []
            q_vec = self.encode_queries([query], batch_size=1)[0].astype(np.float32, copy=False)

        candidate_count = min(max(top_k * 6, top_k), len(self.embeddings))
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_vec.reshape(1, -1), candidate_count)
            chunk_indices = indices[0]
            similarities = scores[0]
        else:
            similarities_all = np.dot(self.embeddings.astype(np.float32, copy=False), q_vec)
            chunk_indices = np.argsort(-similarities_all, kind="stable")[:candidate_count]
            similarities = similarities_all[chunk_indices]

        doc_chunk_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for chunk_index, similarity in zip(chunk_indices, similarities):
            chunk_index = int(chunk_index)
            if chunk_index < 0:
                continue
            doc_id = self.doc_ids[chunk_index] if self.doc_ids else str(chunk_index)
            chunk_id = self.chunk_ids[chunk_index] if self.chunk_ids else str(chunk_index)
            doc_chunk_scores[doc_id].append((chunk_id, float(similarity)))

        document_records: list[dict[str, Any]] = []
        for doc_id, chunk_scores in doc_chunk_scores.items():
            chunk_scores.sort(key=lambda item: (-item[1], item[0]))
            best_chunk_id, best_score = chunk_scores[0]
            second_score = chunk_scores[1][1] if len(chunk_scores) > 1 else 0.0
            mean_score = sum(score for _, score in chunk_scores) / len(chunk_scores)
            aggregate_score = best_score + 0.1 * second_score
            document_records.append(
                {
                    "doc_id": doc_id,
                    "score": float(aggregate_score),
                    "dense_score": float(aggregate_score),
                    "dense_best_score": float(best_score),
                    "dense_second_score": float(second_score),
                    "dense_mean_score": float(mean_score),
                    "dense_best_chunk_id": best_chunk_id,
                }
            )

        document_records.sort(key=lambda item: (-item["score"], str(item["doc_id"])))
        return document_records[:top_k]

    def retrieve(self, query: str, top_k: int = 50, q_emb: np.ndarray | None = None) -> list[dict[str, Any]]:
        """Backward-compatible alias for :meth:`search`."""
        return self.search(query, top_k=top_k, q_emb=q_emb)

    def fit(
        self,
        corpus: Any,
        batch_size: int = 32,
        max_length: int = 512,
        stage_name: str = "corpus",
    ):
        """Fit / encode corpus for retrieval."""
        self.encode_corpus(
            corpus,
            batch_size=batch_size,
            max_length=max_length,
            stage_name=stage_name,
        )
        return self

    def save(self, output_dir: str | Path) -> Path:
        """Save embeddings and chunk metadata with immutable provenance.

        Fails closed for real models when the revision is missing/mutable:
        never relabels unknown weights as newly pinned. ``mock`` needs no pin.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_rev = resolve_dense_revision(self.model_name, getattr(self, "revision", None))
        if str(self.model_name) != "mock" and (resolved_rev is None or is_mutable_revision(resolved_rev)):
            raise ValueError(
                f"Refusing to save dense index for '{self.model_name}' without an immutable "
                f"revision (got {getattr(self, 'revision', None)!r}; registry pin missing/mutable). "
                "Rebuild with a pinned revision instead of relabeling old vectors."
            )
        # Record the resolved pin on the live object so query-cache identity
        # uses the same validated contract.
        self.revision = resolved_rev
        if self.embeddings is not None:
            np.save(str(output_dir / "embeddings.npy"), self.embeddings.astype(np.float16))
        meta_df = pd.DataFrame({"chunk_id": self.chunk_ids, "doc_id": self.doc_ids})
        meta_df.to_parquet(output_dir / "chunks_meta.parquet", index=False)
        emb_arr = np.asarray(self.embeddings) if self.embeddings is not None else None
        manifest = {
            "model_name": self.model_name,
            "model_revision": resolved_rev,
            # Legacy alias read by older loaders; always the immutable pin.
            "revision": resolved_rev,
            "total_chunks": len(self.chunk_ids),
            "dimension": self.dimension,
            "embedding_dim": int(emb_arr.shape[1]) if emb_arr is not None and emb_arr.ndim > 1 else int(self.dimension),
            "embedding_rows": int(emb_arr.shape[0]) if emb_arr is not None and emb_arr.ndim > 1 else len(self.chunk_ids),
            "embedding_dtype": str(emb_arr.dtype) if emb_arr is not None else "float16",
            "chunk_ids_sha256": chunk_ids_digest(self.chunk_ids),
            "use_pyvi": bool(self.use_pyvi),
            "max_length": 512,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return output_dir

    @classmethod
    def build(
        cls,
        chunks: Any,
        output_dir: str | Path,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        dimension: int = DEFAULT_DIMENSION,
        use_pyvi: bool = True,
        model_name_or_path: str | None = None,
        revision: str | None = None,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = model_name_or_path or model_name

        records = cls._records(chunks)
        resolved_rev = resolve_dense_revision(model_name, revision)
        if str(model_name) != "mock" and (resolved_rev is None or is_mutable_revision(resolved_rev)):
            raise ValueError(
                f"Refusing to build dense index for '{model_name}' without an immutable "
                f"revision (got {revision!r}). Pass the pinned revision; never relabel old vectors."
            )
        encoder = cls(
            model_name=model_name,
            dimension=dimension,
            use_pyvi=use_pyvi,
            device=device,
            revision=resolved_rev,
        )
        embeddings = encoder.encode_corpus(records, batch_size=batch_size, max_length=max_length)
        np.save(str(output_dir / "embeddings.npy"), embeddings.astype(np.float16))

        meta_df = pd.DataFrame({"chunk_id": encoder.chunk_ids, "doc_id": encoder.doc_ids})
        meta_df.to_parquet(output_dir / "chunks_meta.parquet", index=False)
        manifest = {
            "model_name": model_name,
            "model_name_or_path": model_name,
            "model_revision": resolved_rev,
            "revision": resolved_rev,
            "total_macro_chunks": len(records),
            "total_chunks": len(records),
            "embedding_dimension": int(embeddings.shape[1]),
            "dimension": int(embeddings.shape[1]),
            "embedding_rows": int(embeddings.shape[0]),
            "embedding_dtype": "float16",
            "chunk_ids_sha256": chunk_ids_digest(encoder.chunk_ids),
            "dtype": "float16",
            "max_length": max_length,
            "use_pyvi": bool(use_pyvi),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return output_dir

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        use_pyvi: bool = True,
        model_name_or_path: str | None = None,
        revision: str | None = None,
    ) -> "DenseMacroRetriever":
        """Load a dense index only when immutable provenance validates.

        Never accepts a missing/mutable (e.g. ``main``) or mismatched revision,
        and never relabels old vectors with today's registry pin. Callers must
        pass the expected immutable revision (or rely on the registry pin for
        the expected model); legacy caches fail closed for rebuild via a fresh
        output directory.
        """
        index_dir = Path(index_dir)
        expected_model = str(model_name_or_path or model_name)
        if expected_model == "mock":
            expected_rev: str | None = None
        elif revision is not None and not is_mutable_revision(revision):
            expected_rev = str(revision).strip()
        else:
            if revision is not None:
                raise ValueError(
                    f"Refusing to load dense index for '{expected_model}' with mutable "
                    f"revision {revision!r}; pass the immutable pin."
                )
            expected_rev = pinned_dense_revision(expected_model)
            if expected_rev is None:
                raise ValueError(
                    f"Refusing to load dense index for '{expected_model}' without an immutable "
                    "expected revision (registry pin missing). Pass revision explicitly."
                )
        embeddings = np.load(str(index_dir / "embeddings.npy"), mmap_mode="r")
        meta_df = pd.read_parquet(index_dir / "chunks_meta.parquet")
        manifest_path = index_dir / "manifest.json"
        manifest_data: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"Dense index manifest unreadable at {manifest_path}: {exc}") from exc
        else:
            raise ValueError(f"Dense index manifest missing at {manifest_path}; rebuild with pinned revision.")
        manifest_model = str(manifest_data.get("model_name") or manifest_data.get("model_name_or_path") or "")
        if manifest_model and manifest_model != expected_model:
            raise ValueError(
                f"Dense index model mismatch (manifest={manifest_model!r} expected={expected_model!r}); rebuild."
            )
        manifest_rev = manifest_data.get("model_revision", manifest_data.get("revision"))
        if expected_model != "mock":
            if manifest_rev is None or is_mutable_revision(manifest_rev):
                raise ValueError(
                    f"Dense index at {index_dir} has missing/mutable revision {manifest_rev!r}; "
                    "rebuild with immutable provenance instead of relabeling."
                )
            if str(manifest_rev).strip() != str(expected_rev).strip():
                raise ValueError(
                    f"Dense index revision mismatch (manifest={manifest_rev!r} expected={expected_rev!r}); rebuild."
                )
        # Dimension / preprocessing / corpus identity.
        manifest_dim = manifest_data.get("dimension", manifest_data.get("embedding_dimension"))
        if manifest_dim is not None and int(manifest_dim) != int(embeddings.shape[1]):
            raise ValueError(
                f"Dense index dimension mismatch (manifest={manifest_dim} npy={embeddings.shape[1]}); rebuild."
            )
        if "use_pyvi" in manifest_data and bool(manifest_data["use_pyvi"]) != bool(use_pyvi):
            raise ValueError(
                f"Dense index preprocessing mismatch (manifest use_pyvi={manifest_data['use_pyvi']!r} "
                f"requested={use_pyvi!r}); rebuild."
            )
        loaded_chunk_ids = meta_df["chunk_id"].astype(str).tolist()
        loaded_doc_ids = meta_df["doc_id"].astype(str).tolist()
        manifest_count = manifest_data.get("total_chunks", manifest_data.get("total_macro_chunks"))
        if manifest_count is not None and int(manifest_count) != len(loaded_chunk_ids):
            raise ValueError(
                f"Dense index row-count mismatch (manifest={manifest_count} meta={len(loaded_chunk_ids)}); rebuild."
            )
        if int(embeddings.shape[0]) != len(loaded_chunk_ids):
            raise ValueError(
                f"Dense index row mismatch (npy rows={embeddings.shape[0]} meta rows={len(loaded_chunk_ids)}); rebuild."
            )
        manifest_sha = manifest_data.get("chunk_ids_sha256")
        if manifest_sha and chunk_ids_digest(loaded_chunk_ids) != str(manifest_sha):
            raise ValueError("Dense index chunk-identity mismatch; rebuild.")
        if embeddings.ndim != 2:
            raise ValueError(f"Dense index embeddings must be 2-D, got ndim={embeddings.ndim}.")
        try:
            if not bool(np.isfinite(np.asarray(embeddings)).all()):
                raise ValueError("Dense index embeddings contain non-finite values.")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Dense index integrity check failed: {exc}") from exc
        retriever = cls(
            model_name=expected_model,
            dimension=int(embeddings.shape[1]),
            use_pyvi=use_pyvi,
            device=device,
            revision=expected_rev,
        )
        retriever._set_embeddings(embeddings)
        retriever.chunk_ids = loaded_chunk_ids
        retriever.doc_ids = loaded_doc_ids
        retriever._validate_metadata_lengths()
        return retriever

"""Fold-safe Question Memory Retriever using Lexical TF-IDF and Dense Embedding Matching."""

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.dense_macro import DenseMacroRetriever


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip().lower()


class TrainQuestionMemory:
    """Fold-local memory over labelled training questions.

    The index is built by :meth:`fit` strictly from training questions and their
    gold qrels.  Validation queries and self-query IDs during training are isolated
    to prevent any target leakage.

    Supports character n-gram TF-IDF and DEk21-compatible question embeddings.
    """

    DEFAULT_MODEL_NAME = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"

    def __init__(
        self,
        min_similarity: float = 0.82,
        dense_encoder: Callable[[list[str]], Any] | Any | None = None,
        dense_min_similarity: float | None = None,
        dense_embeddings: Mapping[str, Any] | Any | None = None,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        model_name: str = DEFAULT_MODEL_NAME,
        use_dense: bool = True,
        dense_dimension: int = 768,
        dense_use_pyvi: bool = True,
        dense_device: str | None = None,
    ):
        if min_similarity < -1.0 or min_similarity > 1.0:
            raise ValueError("min_similarity must be between -1 and 1")
        if dense_min_similarity is not None and not -1.0 <= dense_min_similarity <= 1.0:
            raise ValueError("dense_min_similarity must be between -1 and 1")
        if lexical_weight < 0.0 or dense_weight < 0.0:
            raise ValueError("signal weights must be non-negative")
        if dense_dimension <= 0:
            raise ValueError("dense_dimension must be positive")

        self.min_similarity = float(min_similarity)
        self.dense_min_similarity = float(
            dense_min_similarity if dense_min_similarity is not None else min_similarity
        )
        self.dense_encoder = dense_encoder
        self.dense_embeddings_input = dense_embeddings
        self.use_dense = bool(use_dense or dense_encoder is not None or dense_embeddings is not None)
        self.dense_dimension = int(dense_dimension)
        self.dense_use_pyvi = bool(dense_use_pyvi)
        self.dense_device = dense_device
        self.lexical_weight = float(lexical_weight)
        self.dense_weight = float(dense_weight)
        self.model_name = str(model_name)
        self._clear_index()

    def _clear_index(self) -> None:
        self.qids: list[str] = []
        self.texts: list[str] = []
        self.qid_to_docs: dict[str, list[str]] = {}
        self.training_query_ids: frozenset[str] = frozenset()
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        self.tfidf_matrix = None
        self.dense_embeddings: np.ndarray | None = None

    @staticmethod
    def _read_table(value: Any) -> Any:
        if isinstance(value, Path) or (isinstance(value, str) and Path(value).exists()):
            return pd.read_parquet(value)
        return value

    @staticmethod
    def _query_text(record: Mapping[str, Any]) -> str:
        for key in ("question_norm", "question_raw", "text", "query", "question"):
            value = record.get(key)
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                return normalize_text(value)
        return ""

    @staticmethod
    def _query_embedding(record: Mapping[str, Any]) -> Any | None:
        for key in ("question_embedding", "query_embedding", "embedding", "q_emb"):
            if key in record and record[key] is not None:
                return record[key]
        return None

    @classmethod
    def _query_records(cls, train_queries: Any) -> list[tuple[str, str, Any | None]]:
        train_queries = cls._read_table(train_queries)
        if isinstance(train_queries, pd.DataFrame):
            records = train_queries.to_dict(orient="records")
        elif isinstance(train_queries, Mapping):
            query_keys = {"query_id", "qid", "question_norm", "question_raw", "text", "query"}
            if query_keys.intersection(train_queries):
                records = [dict(train_queries)]
            else:
                records = []
                for qid, value in train_queries.items():
                    if isinstance(value, Mapping):
                        record = dict(value)
                        record.setdefault("query_id", qid)
                    else:
                        record = {"query_id": qid, "question_norm": value}
                    records.append(record)
        elif isinstance(train_queries, str):
            records = [{"query_id": "0", "question_norm": train_queries}]
        else:
            records = list(train_queries or [])

        parsed: list[tuple[str, str, Any | None]] = []
        for index, record in enumerate(records):
            if isinstance(record, Mapping):
                qid = record.get("query_id", record.get("qid", index))
                text = cls._query_text(record)
                embedding = cls._query_embedding(record)
            elif isinstance(record, (tuple, list)):
                if len(record) == 3:
                    qid, text, embedding = record
                elif len(record) == 2:
                    qid, text = record
                    embedding = None
                else:
                    raise ValueError(
                        "question tuple/list must be (qid, text) or (qid, text, embedding)"
                    )
                text = normalize_text(text)
            else:
                qid, text, embedding = index, normalize_text(record), None
            parsed.append((str(qid), text, embedding))
        return parsed

    @staticmethod
    def _doc_ids(value: Any) -> list[str]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, (str, bytes)):
            values = [value]
        else:
            try:
                values = list(value)
            except TypeError:
                values = [value]
        result: list[str] = []
        for doc_id in values:
            if doc_id is None or (isinstance(doc_id, float) and pd.isna(doc_id)):
                continue
            doc_id = str(doc_id)
            if doc_id not in result:
                result.append(doc_id)
        return result

    @classmethod
    def _qrel_records(cls, qrels: Any) -> dict[str, list[str]]:
        qrels = cls._read_table(qrels)
        grouped: dict[str, list[str]] = defaultdict(list)

        if isinstance(qrels, pd.DataFrame):
            query_col = "query_id" if "query_id" in qrels.columns else "qid"
            doc_col = "doc_id" if "doc_id" in qrels.columns else "document_id"
            for record in qrels[[query_col, doc_col]].to_dict(orient="records"):
                grouped[str(record[query_col])].extend(cls._doc_ids(record[doc_col]))
        elif isinstance(qrels, Mapping):
            qrel_keys = {"query_id", "qid", "doc_id", "document_id"}
            if qrel_keys.intersection(qrels):
                qid = qrels.get("query_id", qrels.get("qid"))
                doc_id = qrels.get("doc_id", qrels.get("document_id"))
                if qid is not None:
                    grouped[str(qid)].extend(cls._doc_ids(doc_id))
            else:
                for qid, doc_ids in qrels.items():
                    grouped[str(qid)].extend(cls._doc_ids(doc_ids))
        else:
            for record in list(qrels or []):
                if not isinstance(record, Mapping):
                    continue
                qid = record.get("query_id", record.get("qid"))
                doc_id = record.get("doc_id", record.get("document_id"))
                if qid is not None:
                    grouped[str(qid)].extend(cls._doc_ids(doc_id))

        return {
            qid: list(dict.fromkeys(doc_ids))
            for qid, doc_ids in grouped.items()
            if doc_ids
        }

    @staticmethod
    def _normalize_embeddings(values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("question embeddings must be a two-dimensional matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)

    def _ensure_dense_encoder(self) -> Any | None:
        if self.dense_encoder is None and self.use_dense:
            self.dense_encoder = DenseMacroRetriever(
                model_name=self.model_name,
                dimension=self.dense_dimension,
                use_pyvi=self.dense_use_pyvi,
                device=self.dense_device,
            )
        return self.dense_encoder

    def _encode(self, texts: list[str]) -> np.ndarray:
        encoder = self._ensure_dense_encoder()
        if encoder is None:
            raise ValueError("a dense_encoder is required to encode question text")
        if callable(encoder):
            values = encoder(texts)
        elif hasattr(encoder, "encode_texts"):
            values = encoder.encode_texts(texts)
        elif hasattr(encoder, "encode_queries"):
            values = encoder.encode_queries(texts)
        elif hasattr(encoder, "encode"):
            values = encoder.encode(texts)
        else:
            raise TypeError("dense_encoder must be callable or expose an encode method")
        return self._normalize_embeddings(values)

    def _provided_embeddings(
        self,
        qids: list[str],
        record_embeddings: list[Any | None],
    ) -> np.ndarray | None:
        if not self.use_dense:
            return None

        values = self.dense_embeddings_input
        if values is not None:
            if isinstance(values, Mapping):
                rows = [values.get(qid) for qid in qids]
                if any(row is None for row in rows):
                    return None
                return self._normalize_embeddings(rows)
            matrix = self._normalize_embeddings(values)
            if len(matrix) != len(qids):
                raise ValueError("dense_embeddings length must match indexed training questions")
            return matrix

        if any(value is not None for value in record_embeddings):
            if any(value is None for value in record_embeddings):
                raise ValueError("embeddings must be provided for every indexed question")
            return self._normalize_embeddings(record_embeddings)
        return None

    def fit(
        self,
        train_queries: Any,
        qrels: Any = None,
        dense_embeddings: Any = None,
        encode_dense: bool = True,
    ) -> "TrainQuestionMemory":
        """Index only the supplied training questions and their gold qrels."""
        self._clear_index()
        if dense_embeddings is not None:
            self.dense_embeddings_input = dense_embeddings
        qrels_by_qid = self._qrel_records(qrels) if qrels is not None else {}
        query_records = self._query_records(train_queries)

        for qid, text, embedding in query_records:
            # Leakage guard: only queries with gold labels in the training set enter memory
            if qrels is not None and qid not in qrels_by_qid:
                continue
            self.qids.append(qid)
            self.texts.append(text)
            self.qid_to_docs[qid] = qrels_by_qid.get(qid, [])

        self.training_query_ids = frozenset(self.qids)
        if self.texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

            record_embeddings = [
                embedding
                for qid, _, embedding in query_records
                if qid in self.training_query_ids
            ]
            self.dense_embeddings = self._provided_embeddings(self.qids, record_embeddings)
            if self.dense_embeddings is None and self.use_dense and encode_dense:
                self._ensure_dense_encoder()
                if self.dense_encoder is not None:
                    self.dense_embeddings = self._encode(self.texts)
                    if len(self.dense_embeddings) != len(self.texts):
                        raise ValueError("dense_encoder output length must match indexed questions")
        return self

    @staticmethod
    def _query_vector(q_emb: Any) -> np.ndarray:
        matrix = TrainQuestionMemory._normalize_embeddings(q_emb)
        if len(matrix) != 1:
            raise ValueError("q_emb must contain exactly one question embedding")
        return matrix[0]

    def search(
        self,
        q_text: str,
        top_k: int = 5,
        q_emb: Any | None = None,
        exclude_qid: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return gold documents voted for by similar indexed questions."""
        if top_k <= 0 or not q_text or not self.texts:
            return []

        normalized_query = normalize_text(q_text)
        lexical_sims = cosine_similarity(
            self.vectorizer.transform([normalized_query]), self.tfidf_matrix
        )[0]
        dense_sims: np.ndarray | None = None
        if self.dense_embeddings is not None:
            query_embedding = self._query_vector(q_emb) if q_emb is not None else None
            if query_embedding is None and self.use_dense:
                self._ensure_dense_encoder()
            if query_embedding is None and self.dense_encoder is not None:
                query_embedding = self._encode([normalized_query])[0]
            if query_embedding is not None:
                if query_embedding.shape[0] != self.dense_embeddings.shape[1]:
                    raise ValueError("q_emb dimension must match indexed question embeddings")
                dense_sims = np.dot(self.dense_embeddings, query_embedding)

        votes: dict[str, dict[str, Any]] = {}
        best_contributions: dict[str, float] = {}
        for index, lexical_sim in enumerate(lexical_sims):
            qid = self.qids[index]
            if exclude_qid is not None and str(qid) == str(exclude_qid):
                continue

            lexical_sim = float(lexical_sim)
            dense_sim = float(dense_sims[index]) if dense_sims is not None else 0.0
            lexical_match = lexical_sim >= self.min_similarity
            dense_match = dense_sims is not None and dense_sim >= self.dense_min_similarity
            if not lexical_match and not dense_match:
                continue

            lexical_score = lexical_sim if lexical_match else 0.0
            dense_score = dense_sim if dense_match else 0.0
            contribution = self.lexical_weight * lexical_score + self.dense_weight * dense_score
            for doc_id in self.qid_to_docs.get(qid, []):
                vote = votes.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "score": 0.0,
                        "lexical_similarity": 0.0,
                        "dense_similarity": 0.0,
                        "matched_qid": None,
                        "vote_count": 0,
                    },
                )
                vote["score"] += contribution
                vote["lexical_similarity"] = max(vote["lexical_similarity"], lexical_score)
                vote["dense_similarity"] = max(vote["dense_similarity"], dense_score)
                vote["vote_count"] += 1
                if contribution > best_contributions.get(doc_id, float("-inf")):
                    best_contributions[doc_id] = contribution
                    vote["matched_qid"] = qid

        for vote in votes.values():
            vote["positive_frequency"] = vote["vote_count"] / max(len(self.qids), 1)
        return sorted(
            votes.values(),
            key=lambda vote: (-float(vote["score"]), str(vote["doc_id"])),
        )[:top_k]

    def query(
        self,
        q_text: str,
        top_k: int = 5,
        q_emb: Any | None = None,
        exclude_qid: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.search(q_text, top_k=top_k, q_emb=q_emb, exclude_qid=exclude_qid)

    def retrieve(
        self,
        q_text: str,
        top_k: int = 5,
        q_emb: Any | None = None,
        exclude_qid: str | None = None,
        min_similarity: float | None = None,
        min_sim: float | None = None,
        dense_min_similarity: float | None = None,
    ) -> Any:
        # If called by legacy callers expecting dict return, support it
        hits = self.search(q_text, top_k=top_k, q_emb=q_emb, exclude_qid=exclude_qid)
        return {h["doc_id"]: h for h in hits}

    def save(self, index_dir: str | Path) -> Path:
        """Save question memory index to disk."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "qids": self.qids,
            "queries": self.texts,
            "qrels": self.qid_to_docs,
            "min_similarity": self.min_similarity,
            "dense_min_similarity": self.dense_min_similarity,
        }
        with open(index_dir / "train_qa.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if self.dense_embeddings is not None:
            np.save(str(index_dir / "train_embeddings.npy"), self.dense_embeddings.astype(np.float16))
        return index_dir

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        dense_retriever: Any | None = None,
        min_similarity: float = 0.82,
    ) -> "TrainQuestionMemory":
        """Load question memory index from disk without re-encoding if embeddings exist."""
        index_dir = Path(index_dir)
        qa_path = index_dir / "train_qa.json"
        emb_path = index_dir / "train_embeddings.npy"
        has_dense = (dense_retriever is not None) or emb_path.is_file()
        mem = cls(min_similarity=min_similarity, dense_encoder=dense_retriever, use_dense=has_dense)
        if qa_path.exists():
            with open(qa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            queries = {qid: q for qid, q in zip(data["qids"], data["queries"])}
            qrels = data["qrels"]
            saved_emb = np.load(str(emb_path)) if emb_path.exists() else None
            if saved_emb is not None:
                mem.fit(queries, qrels, dense_embeddings=saved_emb, encode_dense=False)
            else:
                mem.fit(queries, qrels, encode_dense=has_dense)
        return mem


class QuestionMemory(TrainQuestionMemory):
    """Compatibility subclass of TrainQuestionMemory with dict-oriented helper methods."""

    def __init__(
        self,
        train_queries: list[dict[str, Any]] | None = None,
        min_similarity: float = 0.82,
        dense_encoder: Callable[[list[str]], np.ndarray] | None = None,
        dense_min_similarity: float = 0.85,
    ):
        super().__init__(
            min_similarity=min_similarity,
            dense_encoder=dense_encoder,
            dense_min_similarity=dense_min_similarity,
            use_dense=dense_encoder is not None,
        )
        if train_queries is not None:
            self.fit_records(train_queries)

    def fit_records(self, train_queries: list[dict[str, Any]]) -> "QuestionMemory":
        qrels_dict = {}
        queries_dict = {}
        for q in train_queries:
            qid = str(q.get("query_id", q.get("qid", "")))
            text = q.get("question_norm") or q.get("question_raw") or q.get("text", "")
            doc_ids = q.get("doc_ids", [])
            queries_dict[qid] = text
            qrels_dict[qid] = [str(x) for x in doc_ids]

        self.fit(queries_dict, qrels_dict)
        return self

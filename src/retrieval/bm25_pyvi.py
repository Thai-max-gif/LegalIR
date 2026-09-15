"""PyVi Segmented BM25 Retriever for natural-language semantic lexical retrieval."""

from collections import Counter, defaultdict
import functools
import math
from pathlib import Path
import pickle
import re
import time
from typing import Any, Mapping
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset.normalize import clean_legal_text

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

PYVI_TOKEN_PATTERN = re.compile(r'\b[a-zà-ỹ0-9_]+\b', re.IGNORECASE | re.UNICODE)


@functools.lru_cache(maxsize=32768)
def _tokenize_pyvi_cached(text: str) -> tuple[str, ...]:
    """Tokenize and memoize Vietnamese text segmentation for repeated strings."""
    if not isinstance(text, str) or not text:
        return ()
    cleaned = clean_legal_text(text)
    if ViTokenizer is not None:
        segmented = ViTokenizer.tokenize(cleaned)
    else:
        segmented = cleaned
    segmented = unicodedata.normalize("NFC", segmented).lower()
    return tuple(PYVI_TOKEN_PATTERN.findall(segmented))


def tokenize_pyvi(text: str) -> list[str]:
    """Tokenize Vietnamese text with PyVi word segmentation consistently."""
    return list(_tokenize_pyvi_cached(text))


def _normalize_str(val: Any) -> str:
    if val is None or pd.isna(val):
        return ""
    return unicodedata.normalize("NFC", str(val)).strip()


class BM25PyViRetriever:
    """Branch B: Lexical BM25 retriever indexed with PyVi word segmentation."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        field_weights: dict[str, float] | None = None,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.field_weights = field_weights or {
            "legal_number": 4.0,
            "title": 3.0,
            "article": 2.0,
            "clause": 1.0,
            "body": 1.0,
            "url_slug": 1.0,
        }
        self.chunk_ids: list[str] = []
        self.doc_ids: list[str] = []
        self.chunk_lens: np.ndarray | None = None
        self.avg_len: float = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def corpus(self) -> list[str]:
        """Backward-compatibility property returning indexed chunk IDs."""
        return self.chunk_ids

    def fit(self, chunks: Any, show_progress: bool = False) -> "BM25PyViRetriever":
        """Fit BM25 index on micro chunks using PyVi tokenization."""
        if isinstance(chunks, pd.DataFrame):
            n_rows = len(chunks)
            cids = (
                chunks["chunk_id"].fillna("").astype(str).tolist()
                if "chunk_id" in chunks
                else [str(i) for i in range(n_rows)]
            )
            cids = [c if c else str(i) for i, c in enumerate(cids)]
            if "doc_id" in chunks and "document_id" in chunks:
                raw_dids = chunks["doc_id"].fillna(chunks["document_id"]).fillna("").astype(str).tolist()
            elif "doc_id" in chunks:
                raw_dids = chunks["doc_id"].fillna("").astype(str).tolist()
            elif "document_id" in chunks:
                raw_dids = chunks["document_id"].fillna("").astype(str).tolist()
            else:
                raw_dids = cids
            dids = [d if d else c for d, c in zip(raw_dids, cids)]

            body_col = "text_norm" if "text_norm" in chunks else ("text_raw" if "text_raw" in chunks else "text")
            if "text_norm" in chunks or "text_raw" in chunks or "text" in chunks:
                # Mirror list-path fallback: text_norm or text_raw or text.
                norms = chunks["text_norm"].fillna("").astype(str).tolist() if "text_norm" in chunks else [""] * n_rows
                raws = chunks["text_raw"].fillna("").astype(str).tolist() if "text_raw" in chunks else [""] * n_rows
                texts = chunks["text"].fillna("").astype(str).tolist() if "text" in chunks else [""] * n_rows
                bodies = [(n.strip() or r.strip() or t.strip()) for n, r, t in zip(norms, raws, texts)]
            else:
                bodies = [""] * n_rows
            titles = chunks["title"].fillna("").astype(str).tolist() if "title" in chunks else [""] * n_rows
            legal_nums = chunks["legal_number"].fillna("").astype(str).tolist() if "legal_number" in chunks else [""] * n_rows
            articles = chunks["article"].fillna("").astype(str).tolist() if "article" in chunks else [""] * n_rows
            clauses = chunks["clause"].fillna("").astype(str).tolist() if "clause" in chunks else [""] * n_rows
            links = chunks["link"].fillna("").astype(str).tolist() if "link" in chunks else [""] * n_rows
            is_df = True
            N = n_rows
        elif isinstance(chunks, Mapping):
            records = [dict(chunks)]
            is_df = False
            N = len(records)
        else:
            records = list(chunks)
            is_df = False
            N = len(records)

        self.chunk_ids = cids if is_df else []
        self.doc_ids = dids if is_df else []
        lens = np.empty(N, dtype=np.float32)
        term_df = Counter()
        term_docs = defaultdict(list)
        term_freqs = defaultdict(list)

        w_body = int(self.field_weights.get("body", 1.0))
        w_legal = int(self.field_weights.get("legal_number", 4.0))
        w_title = int(self.field_weights.get("title", 3.0))
        w_art = int(self.field_weights.get("article", 2.0))
        w_clause = int(self.field_weights.get("clause", 1.0))
        w_slug = int(self.field_weights.get("url_slug", 1.0))

        t_start = time.time()
        last_log = t_start

        iterator = range(N) if is_df else enumerate(records)
        if show_progress:
            iterator = tqdm(iterator, total=N, desc="Indexing PyVi BM25 chunks")

        for idx_entry in iterator:
            idx = idx_entry if is_df else idx_entry[0]

            if not is_df:
                c = idx_entry[1]
                cid_val = c.get("chunk_id")
                cid = str(cid_val) if cid_val is not None and not pd.isna(cid_val) and str(cid_val) != "" else str(idx)
                did_val = c.get("doc_id")
                if did_val is None or pd.isna(did_val) or str(did_val) == "":
                    did_val = c.get("document_id")
                did = str(did_val) if did_val is not None and not pd.isna(did_val) and str(did_val) != "" else cid
                self.chunk_ids.append(cid)
                self.doc_ids.append(did)
                body_text = (
                    _normalize_str(c.get("text_norm"))
                    or _normalize_str(c.get("text_raw"))
                    or _normalize_str(c.get("text", ""))
                )
                legal_num = _normalize_str(c.get("legal_number", ""))
                title = _normalize_str(c.get("title", ""))
                article = _normalize_str(c.get("article", ""))
                clause = _normalize_str(c.get("clause", ""))
                link = _normalize_str(c.get("link", ""))
            else:
                body_text = _normalize_str(bodies[idx])
                legal_num = _normalize_str(legal_nums[idx])
                title = _normalize_str(titles[idx])
                article = _normalize_str(articles[idx])
                clause = _normalize_str(clauses[idx])
                link = _normalize_str(links[idx])

            weighted_tokens: list[str] = []

            if body_text:
                b_toks = _tokenize_pyvi_cached(body_text)
                if b_toks:
                    weighted_tokens.extend(b_toks if w_body == 1 else b_toks * w_body)

            if legal_num:
                ln_toks = _tokenize_pyvi_cached(legal_num)
                if ln_toks:
                    weighted_tokens.extend(ln_toks if w_legal == 1 else ln_toks * w_legal)

            if title:
                ti_toks = _tokenize_pyvi_cached(title)
                if ti_toks:
                    weighted_tokens.extend(ti_toks if w_title == 1 else ti_toks * w_title)

            if article:
                ar_toks = _tokenize_pyvi_cached(article)
                if ar_toks:
                    weighted_tokens.extend(ar_toks if w_art == 1 else ar_toks * w_art)

            if clause:
                cl_toks = _tokenize_pyvi_cached(clause)
                if cl_toks:
                    weighted_tokens.extend(cl_toks if w_clause == 1 else cl_toks * w_clause)

            if link:
                slug = link.rstrip("/").split("/")[-1].replace("-", " ")
                sl_toks = _tokenize_pyvi_cached(_normalize_str(slug))
                if sl_toks:
                    weighted_tokens.extend(sl_toks if w_slug == 1 else sl_toks * w_slug)

            doc_len = len(weighted_tokens)
            lens[idx] = doc_len

            tf = Counter(weighted_tokens)
            for term, freq in tf.items():
                term_df[term] += 1
                term_docs[term].append(idx)
                term_freqs[term].append(freq)

            now = time.time()
            if (now - last_log) >= 30.0:
                elapsed = now - t_start
                pct = (idx + 1) / max(1, N) * 100.0
                rate = (idx + 1) / max(0.001, elapsed)
                eta = (N - idx - 1) / max(0.001, rate)
                print(
                    f"[*] PyVi BM25 indexing: {idx + 1:,}/{N:,} chunks ({pct:.1f}%) | "
                    f"rate: {rate:.1f} chunks/s | elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
                    flush=True,
                )
                last_log = now

        self.chunk_lens = lens
        self.avg_len = float(np.mean(self.chunk_lens)) if N > 0 else 1.0

        # Calculate BM25 IDF
        self.idf = {
            term: float(math.log((N - df + 0.5) / (df + 0.5) + 1.0))
            for term, df in term_df.items()
        }

        # Convert postings to numpy arrays
        self.postings = {
            term: (np.array(term_docs[term], dtype=np.int32), np.array(term_freqs[term], dtype=np.float32))
            for term in term_df
        }

        elapsed_total = time.time() - t_start
        print(
            f"[+] PyVi BM25 indexing complete: {N:,} chunks in {elapsed_total:.1f}s "
            f"({N / max(0.001, elapsed_total):.1f} chunks/s) | vocabulary: {len(self.idf):,} terms.",
            flush=True,
        )

        return self

    def retrieve(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """Retrieve top candidate documents with PyVi tokenized BM25."""
        if not query or self.chunk_lens is None or len(self.chunk_ids) == 0:
            return []

        tokens = tokenize_pyvi(query)
        if not tokens:
            return []

        q_tf = Counter(tokens)
        scores_arr = np.zeros(len(self.chunk_ids), dtype=np.float32)
        has_matches = False

        for term, qf in q_tf.items():
            if term not in self.postings:
                continue

            c_indices, t_freqs = self.postings[term]
            idf_val = self.idf[term]
            lens = self.chunk_lens[c_indices]

            num = t_freqs * (self.k1 + 1.0)
            den = t_freqs + self.k1 * (1.0 - self.b + self.b * (lens / self.avg_len))
            term_scores = (idf_val * qf) * (num / den)

            np.add.at(scores_arr, c_indices, term_scores)
            has_matches = True

        if not has_matches:
            return []

        candidate_count = min(max(top_k * 10, 100), len(scores_arr))
        top_chunk_indices = np.argpartition(scores_arr, -candidate_count)[-candidate_count:]
        top_chunk_indices = top_chunk_indices[scores_arr[top_chunk_indices] > 0]

        # Aggregate to document level
        doc_chunk_scores = defaultdict(list)
        for c_idx in top_chunk_indices:
            did = self.doc_ids[c_idx]
            cid = self.chunk_ids[c_idx]
            doc_chunk_scores[did].append((cid, float(scores_arr[c_idx])))

        doc_records = []
        for did, items in doc_chunk_scores.items():
            sorted_items = sorted(items, key=lambda x: -x[1])
            best_cid, best_s = sorted_items[0]
            second_s = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
            mean_s = sum(x[1] for x in items) / len(items)
            agg_score = best_s + 0.1 * second_s

            doc_records.append({
                "doc_id": did,
                "score": float(agg_score),
                "bm25_score": float(agg_score),
                "bm25_pyvi_score": float(agg_score),
                "bm25_pyvi_best_score": float(best_s),
                "bm25_pyvi_second_score": float(second_s),
                "bm25_pyvi_mean_score": float(mean_s),
                "bm25_pyvi_best_chunk_id": best_cid,
            })

        doc_records.sort(key=lambda x: (-x["score"], str(x["doc_id"])))
        return doc_records[:top_k]

    def search(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """Alias for retrieve."""
        return self.retrieve(query, top_k=top_k)

    def save(self, file_path: str | Path) -> Path:
        file_path = Path(file_path)
        if file_path.is_dir() or file_path.suffix == "":
            file_path.mkdir(parents=True, exist_ok=True)
            target = file_path / "bm25_pyvi_index.pkl"
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            target = file_path
        with open(target, "wb") as f:
            pickle.dump(self, f)
        return target

    @classmethod
    def load(cls, file_path: str | Path) -> "BM25PyViRetriever":
        file_path = Path(file_path)
        if file_path.is_dir():
            target = file_path / "bm25_pyvi_index.pkl"
            if not target.exists():
                candidates = list(file_path.glob("*.pkl"))
                target = candidates[0] if candidates else target
        else:
            target = file_path
        with open(target, "rb") as f:
            return pickle.load(f)

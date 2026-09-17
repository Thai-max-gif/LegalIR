"""PyVi Segmented BM25 Retriever for natural-language semantic lexical retrieval."""

from collections import Counter, defaultdict
import functools
import math
import os
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

# Hard cap on input length for PyVi word segmentation. PyVi's segmenter hangs
# effectively forever on multi-megabyte inputs (observed: a 5.7M-char anomalous
# "micro" chunk stalls ViTokenizer.tokenize beyond 90s vs ~1ms for normal
# chunks, killing full-corpus indexing at ~88%). Legitimate chunks are bounded
# (micro 100-250 tokens, fallback <= 1200 tokens); 20k chars is >10x headroom.
# Oversized inputs fall back to regex word tokens (no compound joining).
PYVI_MAX_CHARS = 20000

_pyvi_fallback_chunks = 0


def get_pyvi_fallback_count() -> int:
    """Number of inputs that exceeded PYVI_MAX_CHARS and used regex fallback."""
    return _pyvi_fallback_chunks


@functools.lru_cache(maxsize=32768)
def _tokenize_pyvi_cached(text: str) -> tuple[str, ...]:
    """Tokenize and memoize Vietnamese text segmentation for repeated strings."""
    global _pyvi_fallback_chunks
    if not isinstance(text, str) or not text:
        return ()
    cleaned = clean_legal_text(text)
    if len(cleaned) > PYVI_MAX_CHARS:
        # Pathological input: skip the segmenter (hang risk), regex-tokenize.
        _pyvi_fallback_chunks += 1
        segmented = cleaned
    elif ViTokenizer is not None:
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


def _extract_and_tokenize_doc(
    body_text: str,
    legal_num: str,
    title: str,
    article: str,
    clause: str,
    slug: str,
    w_body: int,
    w_legal: int,
    w_title: int,
    w_art: int,
    w_clause: int,
    w_slug: int,
) -> tuple[int, dict[str, int]]:
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

    if slug:
        sl_toks = _tokenize_pyvi_cached(slug)
        if sl_toks:
            weighted_tokens.extend(sl_toks if w_slug == 1 else sl_toks * w_slug)

    return len(weighted_tokens), dict(Counter(weighted_tokens))


def _tokenize_doc_batch_worker(
    batch: list[tuple[str, str, str, str, str, str, int, int, int, int, int, int]]
) -> list[tuple[int, dict[str, int]]]:
    return [_extract_and_tokenize_doc(*item) for item in batch]


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

    def fit(self, chunks: Any, show_progress: bool = False, num_workers: int | None = None) -> "BM25PyViRetriever":
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

        items: list[tuple[str, str, str, str, str, str, int, int, int, int, int, int]] = []
        if is_df:
            for idx in range(N):
                l_str = links[idx]
                slug_str = l_str.rstrip("/").split("/")[-1].replace("-", " ") if l_str else ""
                items.append((
                    _normalize_str(bodies[idx]),
                    _normalize_str(legal_nums[idx]),
                    _normalize_str(titles[idx]),
                    _normalize_str(articles[idx]),
                    _normalize_str(clauses[idx]),
                    _normalize_str(slug_str),
                    w_body,
                    w_legal,
                    w_title,
                    w_art,
                    w_clause,
                    w_slug,
                ))
        else:
            for idx, c in enumerate(records):
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
                slug_str = link.rstrip("/").split("/")[-1].replace("-", " ") if link else ""
                items.append((
                    body_text,
                    legal_num,
                    title,
                    article,
                    clause,
                    _normalize_str(slug_str),
                    w_body,
                    w_legal,
                    w_title,
                    w_art,
                    w_clause,
                    w_slug,
                ))

        t_start = time.time()
        last_log = t_start

        # Resolve workers: check override or default to multiprocessing for large corpora
        resolved_workers = num_workers
        if resolved_workers is None:
            env_w = os.environ.get("PYVI_NUM_WORKERS")
            if env_w:
                try:
                    resolved_workers = int(env_w)
                except ValueError:
                    resolved_workers = 1
            else:
                resolved_workers = min(4, max(1, (os.cpu_count() or 1) - 1)) if N >= 5000 else 1

        if resolved_workers > 1 and N >= 5000:
            import concurrent.futures
            batch_size = 2000
            batches = [items[i : i + batch_size] for i in range(0, N, batch_size)]
            print(f"[*] Extracting and tokenizing {N:,} PyVi chunks across {resolved_workers} worker processes...")
            curr_idx = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=resolved_workers) as executor:
                for b_res in executor.map(_tokenize_doc_batch_worker, batches):
                    for doc_len, tf in b_res:
                        lens[curr_idx] = doc_len
                        for term, freq in tf.items():
                            term_df[term] += 1
                            term_docs[term].append(curr_idx)
                            term_freqs[term].append(freq)
                        curr_idx += 1
        else:
            iterator = range(N)
            if show_progress:
                iterator = tqdm(iterator, total=N, desc="Indexing PyVi BM25 chunks")
            for idx in iterator:
                doc_len, tf = _extract_and_tokenize_doc(*items[idx])
                lens[idx] = doc_len
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
            f"({N / max(0.001, elapsed_total):.1f} chunks/s) | vocabulary: {len(self.idf):,} terms | "
            f"pyvi_length_fallbacks: {get_pyvi_fallback_count()}.",
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

from __future__ import annotations

from pathlib import Path


def pyvi_tokenize(text: str) -> list[str]:
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize(text).split()


class BM25SRetriever:
    def __init__(self, index_path: str | Path):
        import bm25s
        self.index_path = str(index_path)
        self.index = bm25s.BM25.load(self.index_path, load_corpus=True)

    def search(self, query: str, k: int) -> list[str]:
        results, _ = self.index.retrieve([pyvi_tokenize(query)], k=k)
        return [str(item) for item in results[0]]

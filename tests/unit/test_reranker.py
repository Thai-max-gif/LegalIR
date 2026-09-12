import json

import pytest

from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.reranker import CrossEncoderReranker

def test_evidence_pack_builder():
    macro_chunks = [
        {
            "chunk_id": "doc1_macro_01",
            "doc_id": "doc1",
            "article": "Điều 15. Thời hạn cấp phép",
            "text_norm": "[VĂN BẢN]: Thông tư 12/2020\n[ĐIỀU KHOẢN]: Điều 15. Thời hạn cấp phép\n[NỘI DUNG]: Thời hạn cấp phép là 10 ngày làm việc."
        },
        {
            "chunk_id": "doc1_macro_02",
            "doc_id": "doc1",
            "article": "Điều 16. Lệ phí",
            "text_norm": "[VĂN BẢN]: Thông tư 12/2020\n[ĐIỀU KHOẢN]: Điều 16. Lệ phí\n[NỘI DUNG]: Mức thu lệ phí là 100.000 đồng."
        }
    ]
    builder = EvidencePackBuilder(macro_chunks)

    # Build evidence pack for doc1
    pack = builder.build_evidence("Thời hạn cấp phép là bao nhiêu ngày?", "doc1")
    assert pack is not None
    assert "Thời hạn cấp phép là 10 ngày làm việc" in pack["evidence_text"]
    assert pack["chunk_id"] == "doc1_macro_01"


def test_reranker_orders_scored_tuple_candidates_by_cross_encoder_score():
    chunks = [
        {"chunk_id": "low-c", "doc_id": "low", "text_norm": "low evidence"},
        {"chunk_id": "high-c", "doc_id": "high", "text_norm": "high evidence"},
        {"chunk_id": "high-c2", "doc_id": "high", "text_norm": "more high evidence"},
    ]
    builder = EvidencePackBuilder(macro_chunks=chunks)
    seen_passages = []

    def score_pairs(pairs, batch_size=16, max_length=512):
        seen_passages.extend(passage for _, passage in pairs)
        return [10.0 if "high evidence" in passage else 1.0 for _, passage in pairs]

    reranker = CrossEncoderReranker(model_name="mock", score_fn=score_pairs)
    ranked = reranker.rerank(
        "Which evidence applies?",
        [("low", 0.0), ("high", 0.0)],
        builder,
        top_k=2,
    )

    assert [candidate["doc_id"] for candidate in ranked] == ["high", "low"]
    assert any("[EVIDENCE 2]" in passage for passage in seen_passages)


def test_reranker_resolves_manifest_model_path_for_offline_inference(tmp_path):
    model_dir = tmp_path / "bge-reranker"
    model_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"BAAI/bge-reranker-v2-m3": {"path": str(model_dir)}}),
        encoding="utf-8",
    )

    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        manifest_path=manifest_path,
        local_files_only=True,
    )

    assert reranker.model_path == model_dir
    assert reranker.local_files_only is True


def test_run_all_reranker_forwards_manifest_and_offline_mode(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    run_all_module = importlib.import_module("src.pipeline.run_all")
    captured = {}

    class StubReranker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(run_all_module, "CrossEncoderReranker", StubReranker)
    paths = SimpleNamespace(local_models=tmp_path / "models")

    run_all_module._build_reranker(paths, offline=True)

    assert captured["model_name"] == "BAAI/bge-reranker-v2-m3"
    assert captured["manifest_path"] == paths.local_models / "huggingface" / "manifest.json"
    assert captured["local_files_only"] is True


def test_run_all_builds_missing_dek21_dense_index(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    run_all_module = importlib.import_module("src.pipeline.run_all")
    calls = {}

    class StubDense:
        @classmethod
        def build(cls, **kwargs):
            calls["build"] = kwargs
            output_dir = kwargs["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "embeddings.npy").touch()
            (output_dir / "chunks_meta.parquet").touch()

        @classmethod
        def load(cls, **kwargs):
            calls["load"] = kwargs
            return cls()

    monkeypatch.setattr(run_all_module, "DenseMacroRetriever", StubDense)
    paths = SimpleNamespace(
        local_indexes=tmp_path / "indexes",
        local_models=tmp_path / "models",
    )
    chunks = [{"doc_id": "doc-1", "chunk_id": "chunk-1", "text_norm": "text"}]

    result = run_all_module._load_dense_branch(
        paths,
        {"model_name": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"},
        offline=False,
        chunks=chunks,
    )

    assert isinstance(result, StubDense)
    assert calls["build"]["chunks"] == chunks
    assert calls["build"]["model_name"] == "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    assert calls["load"]["model_name"] == "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"


def test_run_all_reranker_default_follows_configured_enabled_flag():
    import importlib

    run_all_module = importlib.import_module("src.pipeline.run_all")

    assert run_all_module._resolve_use_reranker({"ranking": {"reranker": {"enabled": True}}}, None)
    assert not run_all_module._resolve_use_reranker(
        {"ranking": {"reranker": {"enabled": True}}}, False
    )
    assert run_all_module._resolve_use_reranker({}, None)

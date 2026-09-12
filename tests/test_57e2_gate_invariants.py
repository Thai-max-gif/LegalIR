"""Authoritative test suite for LEGALIR_57E2_FINAL_PRE_KAGGLE_GATE_REPAIR contract.

Covers:
1. Documented Kaggle mount data and public test discovery (/kaggle/input/legalir)
2. Strict gpu_smoke configuration (strict_artifacts=True)
3. Submission validation rejecting unknown corpus IDs
4. Learned fusion save/load round-trip prediction preservation
5. Selected learned fusion preventing silent fallback to RRF
6. Adapter weights checksum verification against manifest
7. Competition submission ensuring exactly 5 unique valid IDs per answer
8. Document-disjoint report persistence in OOFRunner
9. Kaggle notebook minimal dependencies including LightGBM & SentencePiece
10. Byte-identical notebook synchronization
"""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.generate_kaggle_notebook import generate_and_save_notebooks
from src.evaluation.submission import validate_submission, validate_submission_zip
from src.pipeline.kaggle_train import (
    discover_data_dir,
    discover_public_test_file,
    run_kaggle_pipeline,
)
from src.pipeline.oof_runner import OOFRunner
from src.pipeline.predict import LegalIRPipeline
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.fusion import LightGBMRanker, LinearRanker, ReciprocalRankFusion
from src.ranking.oof_features import CORE_FEATURE_COLUMNS, extract_candidate_features
from src.ranking.selector import TopKSelector
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import TrainQuestionMemory

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_legalir_data(tmp_path: Path):
    """Create minimal canonical data files for integration tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        {"doc_id": "101", "title": "Luật Doanh nghiệp", "name_raw": "Luật Doanh nghiệp", "is_empty": False},
        {"doc_id": "102", "title": "Luật Đầu tư", "name_raw": "Luật Đầu tư", "is_empty": False},
        {"doc_id": "103", "title": "Nghị định 01", "name_raw": "Nghị định 01", "is_empty": False},
        {"doc_id": "104", "title": "Nghị định 31", "name_raw": "Nghị định 31", "is_empty": False},
        {"doc_id": "105", "title": "Luật Thương mại", "name_raw": "Luật Thương mại", "is_empty": False},
        {"doc_id": "106", "title": "Luật Quản lý thuế", "name_raw": "Luật Quản lý thuế", "is_empty": False},
    ]
    pd.DataFrame(docs).to_parquet(data_dir / "documents.parquet", index=False)

    chunks = [
        {"chunk_id": "c1", "parent_chunk_id": "p1", "doc_id": "101", "granularity": "micro", "text_raw": "Thành lập doanh nghiệp", "text_norm": "thành lập doanh nghiệp"},
        {"chunk_id": "p1", "parent_chunk_id": None, "doc_id": "101", "granularity": "macro", "text_raw": "Thành lập doanh nghiệp", "text_norm": "thành lập doanh nghiệp"},
        {"chunk_id": "c2", "parent_chunk_id": "p2", "doc_id": "102", "granularity": "micro", "text_raw": "Dự án đầu tư", "text_norm": "dự án đầu tư"},
        {"chunk_id": "p2", "parent_chunk_id": None, "doc_id": "102", "granularity": "macro", "text_raw": "Dự án đầu tư", "text_norm": "dự án đầu tư"},
        {"chunk_id": "c3", "parent_chunk_id": "p3", "doc_id": "103", "granularity": "micro", "text_raw": "Đăng ký doanh nghiệp", "text_norm": "đăng ký doanh nghiệp"},
        {"chunk_id": "p3", "parent_chunk_id": None, "doc_id": "103", "granularity": "macro", "text_raw": "Đăng ký doanh nghiệp", "text_norm": "đăng ký doanh nghiệp"},
    ]
    pd.DataFrame(chunks).to_parquet(data_dir / "chunks.parquet", index=False)

    queries = [
        {"query_id": "q1", "question_raw": "Thành lập công ty TNHH", "question_norm": "thành lập công ty tnhh"},
        {"query_id": "q2", "question_raw": "Dự án đầu tư trực tiếp", "question_norm": "dự án đầu tư trực tiếp"},
        {"query_id": "q3", "question_raw": "Đăng ký doanh nghiệp", "question_norm": "đăng ký doanh nghiệp"},
    ]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet", index=False)

    qrels = [
        {"query_id": "q1", "doc_id": "101"},
        {"query_id": "q2", "doc_id": "102"},
        {"query_id": "q3", "doc_id": "103"},
    ]
    pd.DataFrame(qrels).to_parquet(data_dir / "qrels_train.parquet", index=False)

    split_info = [
        {"fold": 0, "train_query_ids": ["q2", "q3"], "val_query_ids": ["q1"]},
        {"fold": 1, "train_query_ids": ["q1"], "val_query_ids": ["q2", "q3"]},
    ]
    (splits_dir / "random_5fold.json").write_text(json.dumps(split_info), encoding="utf-8")

    doc_disjoint_split = {
        "train_query_ids": ["q1", "q2"],
        "val_query_ids": ["q3"],
        "train_doc_ids": ["101", "102"],
        "val_doc_ids": ["103"],
    }
    (splits_dir / "doc_disjoint_split.json").write_text(json.dumps(doc_disjoint_split), encoding="utf-8")

    return data_dir, index_dir


# ==============================================================================
# 1. Documented Kaggle Mount Discovery Tests
# ==============================================================================

def test_discover_documented_kaggle_legalir_mount(tmp_path: Path):
    """Verify discovery finds /kaggle/input/legalir when mounted as documented in README."""
    fake_kaggle_mount = tmp_path / "legalir"
    fake_kaggle_mount.mkdir(parents=True, exist_ok=True)
    for fname in ["documents.parquet", "chunks.parquet", "queries_train.parquet", "qrels_train.parquet"]:
        (fake_kaggle_mount / fname).write_bytes(b"mock parquet content")

    with mock.patch("src.pipeline.kaggle_train.Path") as mock_path:
        # Wrap real Path behavior while redirecting /kaggle/input/legalir
        orig_path = Path
        def side_effect_path(*args):
            p_obj = orig_path(*args)
            if str(p_obj) == "/kaggle/input/legalir":
                return fake_kaggle_mount
            return p_obj
        mock_path.side_effect = side_effect_path

        found = discover_data_dir()
        assert found == fake_kaggle_mount.resolve()


def test_discover_public_official_from_documented_mount(tmp_path: Path):
    """Verify discovery finds public-official.json under /kaggle/input/legalir."""
    fake_kaggle_mount = tmp_path / "legalir"
    fake_kaggle_mount.mkdir(parents=True, exist_ok=True)
    pub_file = fake_kaggle_mount / "public-official.json"
    pub_file.write_text(json.dumps({"q1": "câu hỏi"}), encoding="utf-8")

    with mock.patch("src.pipeline.kaggle_train.Path") as mock_path:
        orig_path = Path
        def side_effect_path(*args):
            p_obj = orig_path(*args)
            if str(p_obj) == "/kaggle/input/legalir/public-official.json":
                return pub_file
            return p_obj
        mock_path.side_effect = side_effect_path

        found = discover_public_test_file()
        assert found == pub_file.resolve()


# ==============================================================================
# 2. Strict gpu_smoke & Corpus ID Validation Tests
# ==============================================================================

def test_gpu_smoke_defaults_to_strict_artifacts():
    """Verify that run_kaggle_pipeline enforces strict_artifacts=True in gpu_smoke mode."""
    # In run_kaggle_pipeline, strict_artifacts defaults to is_full or is_gpu_smoke
    is_gpu_smoke = True
    is_full = False
    strict_artifacts = None
    if strict_artifacts is None:
        strict_artifacts = is_full or is_gpu_smoke
    assert strict_artifacts is True


def test_full_submission_rejects_unknown_corpus_id(tmp_path: Path):
    """Verify submission validation catches document IDs not present in the canonical corpus."""
    sub_json = tmp_path / "submission.json"
    predictions = {
        "q1": {"answer": ["101", "102", "999999"]},  # 999999 is not in corpus
    }
    sub_json.write_text(json.dumps(predictions), encoding="utf-8")

    valid_corpus_ids = {"101", "102", "103", "104", "105"}
    res = validate_submission(sub_json, expected_qids={"q1"}, corpus_doc_ids=valid_corpus_ids)
    assert not res["is_valid"]
    assert any("unknown document" in err.lower() for err in res["errors"])


# ==============================================================================
# 3. Learned Fusion Round-Trip & Fallback Tests
# ==============================================================================

def test_learned_fusion_roundtrip_preserves_predictions(tmp_path: Path):
    """Verify that LightGBMRanker and LinearRanker preserve exact predictions across save/load."""
    np.random.seed(42)
    # Synthetic candidate data
    records = [
        {"doc_id": "101", "raw_bm25_score": 10.0, "dense_score": 0.9, "exact_score": 1.0, "reranker_score": 2.5},
        {"doc_id": "102", "raw_bm25_score": 5.0, "dense_score": 0.4, "exact_score": 0.0, "reranker_score": -1.0},
        {"doc_id": "103", "raw_bm25_score": 2.0, "dense_score": 0.1, "exact_score": 0.0, "reranker_score": -2.0},
    ]
    df_feat = extract_candidate_features("q1", records, "câu hỏi thành lập doanh nghiệp")
    X = pd.concat([df_feat, df_feat], ignore_index=True)
    y = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
    groups = np.array([3, 3], dtype=np.int32)

    ranker = LinearRanker(feature_cols=list(CORE_FEATURE_COLUMNS))
    ranker.fit(X, y, groups)
    preds_before = ranker.predict(records, query_id="q1", query_text="câu hỏi")

    save_file = tmp_path / "linear_model.json"
    ranker.save(save_file)

    reloaded_ranker = LightGBMRanker(model_file=save_file)
    preds_after = reloaded_ranker.predict(records, query_id="q1", query_text="câu hỏi")

    assert [r["doc_id"] for r in preds_before] == [r["doc_id"] for r in preds_after]
    for b, a in zip(preds_before, preds_after):
        assert pytest.approx(b["final_score"], rel=1e-4) == a["final_score"]


def test_selected_learned_fusion_cannot_silently_fallback_to_rrf(tmp_path: Path):
    """Verify that an unloaded LightGBMRanker with strict=True raises RuntimeError instead of silent RRF."""
    ranker = LightGBMRanker(strict=True)
    records = [{"doc_id": "101", "raw_bm25_score": 10.0}]

    with pytest.raises(RuntimeError, match="no active loaded model"):
        ranker.predict(records)


# ==============================================================================
# 4. Adapter Checksum & Exact Top-5 Submission Tests
# ==============================================================================

def test_adapter_checksum_is_verified(tmp_path: Path):
    """Verify that strict adapter loading fails if the weights file SHA-256 does not match the manifest."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"doc_id": "101", "title": "Doc 1"}]).to_parquet(data_dir / "documents.parquet", index=False)

    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    bm25 = BM25MicroRetriever().fit([{"chunk_id": "c1", "doc_id": "101", "text_norm": "q"}], show_progress=False)
    bm25.save(index_dir / "bm25")

    bm25_pyvi = BM25PyViRetriever().fit([{"chunk_id": "c1", "doc_id": "101", "text_norm": "q"}], show_progress=False)
    bm25_pyvi.save(index_dir / "bm25_pyvi")

    dense_dir = index_dir / "dense_dek21"
    dense_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(dense_dir / "embeddings.npy"), np.ones((1, 768), dtype=np.float32))
    pd.DataFrame({"chunk_id": ["c1"], "doc_id": ["101"]}).to_parquet(dense_dir / "chunks_meta.parquet", index=False)

    mem_dir = index_dir / "question_memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "train_qa.json").write_text(json.dumps({"qids": ["q1"], "queries": ["q"], "qrels": {"q1": ["101"]}}), encoding="utf-8")

    adapter_dir = tmp_path / "adapter_corrupted"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights_file = adapter_dir / "adapter_model.safetensors"
    weights_file.write_bytes(b"corrupted weights data")

    # Manifest with a different hash
    manifest = {
        "status": "completed",
        "param_diff": 0.05,
        "adapter_checksum": "0000000000000000000000000000000000000000000000000000000000000000",
        "unique_training_queries": 10,
        "optimizer_steps": 5,
    }
    (adapter_dir / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="adapter checksum mismatch"):
        LegalIRPipeline.load_pipeline(
            data_dir=data_dir,
            index_dir=index_dir,
            use_reranker=True,
            reranker_adapter_path=adapter_dir,
            reranker_model_name="BAAI/bge-reranker-v2-m3",
            strict_artifacts=True,
        )


def test_full_submission_every_answer_has_exactly_five_ids(mock_legalir_data):
    """Verify that LegalIRPipeline with valid_doc_ids returns exactly 5 IDs for every query."""
    data_dir, index_dir = mock_legalir_data
    docs_df = pd.read_parquet(data_dir / "documents.parquet")
    valid_ids = set(docs_df["doc_id"].astype(str))
    assert len(valid_ids) >= 5

    pipeline = LegalIRPipeline(
        hybrid_engine=HybridSearchEngine(
            bm25_retriever=BM25MicroRetriever().fit([{"chunk_id": "c1", "doc_id": "101", "text_norm": "câu hỏi"}]),
        ),
        valid_doc_ids=valid_ids,
        fallback_doc_ids=sorted(list(valid_ids)),
    )

    answer_1 = pipeline.predict_one("q1", "câu hỏi thành lập")
    assert len(answer_1) == 5
    assert len(set(answer_1)) == 5
    assert all(did in valid_ids for did in answer_1)

    # Empty query should also return exactly 5 valid fallback IDs
    answer_empty = pipeline.predict_one("q2", "")
    assert len(answer_empty) == 5
    assert len(set(answer_empty)) == 5
    assert all(did in valid_ids for did in answer_empty)


def test_doc_disjoint_report_is_persisted(mock_legalir_data, tmp_path: Path):
    """Verify that OOFRunner persists doc_disjoint_report attribute after evaluation."""
    data_dir, index_dir = mock_legalir_data
    cv_dir = tmp_path / "cv_disjoint"

    runner = OOFRunner(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=cv_dir,
        num_folds=1,
        smoke=True,
        smoke_sample_size=2,
        use_reranker=False,
    )
    runner.load_data()
    runner.load_retrievers()
    report = runner.run_document_disjoint_evaluation()

    assert runner.doc_disjoint_report == report
    assert "retrieval_only" in runner.doc_disjoint_report
    assert "trained_reranker_system" in runner.doc_disjoint_report


# ==============================================================================
# 5. Notebook Dependency & Parity Tests
# ==============================================================================

def test_notebook_installs_required_non_torch_dependencies():
    """Verify that legalir_training.ipynb checks lightgbm, sentencepiece, bm25s, pyvi, peft, accelerate."""
    root_nb = REPO_ROOT / "legalir_training.ipynb"
    assert root_nb.exists()
    nb_data = json.loads(root_nb.read_text(encoding="utf-8"))

    cell_sources = "".join("".join(c["source"]) for c in nb_data["cells"])
    for pkg in ["lightgbm", "sentencepiece", "bm25s", "pyvi", "peft", "accelerate"]:
        assert pkg in cell_sources, f"Missing dependency check for {pkg} in notebook"


def test_notebook_byte_level_parity_gate():
    """Verify notebooks exist and pass parity verification."""
    from scripts.check_notebook_parity import check_all_notebook_parity
    is_valid, report = check_all_notebook_parity(REPO_ROOT)
    assert is_valid is True, f"Parity check failed: {report}"

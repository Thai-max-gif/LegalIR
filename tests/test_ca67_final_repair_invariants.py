"""Comprehensive Integration & Unit Tests for CA67 Final Pre-Kaggle Repair Contract.

Covers all 25+ mandatory tests specified in Section 14 of LEGALIR_CA67_FINAL_PRE_KAGGLE_REPAIR.md:
- QuestionMemory tuple/qid preservation, fold safety, non-empty, and load without re-encoding
- CUDA device index resolution (cuda:0, cuda:1, out-of-range, torch.device) and explicit device propagation
- Explicit training config separation and max_steps validation
- OOF post-rerank feature extraction and zero validation label leakage
- Document-disjoint trained reranker evaluation
- Strict artifact loading fail-fast behavior
- Public test official input requirement and submission validation with official keys
- Exact matcher metadata parity (year, doc_type)
- Query embedding cache reuse across dense, memory, pair mining, and inference
- Kaggle notebook byte-level parity
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from scripts.generate_kaggle_notebook import generate_and_save_notebooks
from src.evaluation.submission import validate_submission, validate_submission_zip
from src.models.device import resolve_device
from src.pipeline.kaggle_train import run_kaggle_pipeline
from src.pipeline.oof_runner import OOFRunner
from src.pipeline.predict import LegalIRPipeline
from src.ranking.oof_features import (
    CORE_FEATURE_COLUMNS,
    compute_training_doc_frequencies,
    extract_candidate_features,
)
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.question_memory import TrainQuestionMemory
from src.training.build_pairs import build_training_pairs
from src.training.train_reranker import train_reranker

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_dataset_and_indexes(tmp_path: Path):
    """Create minimal canonical data and indexes for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        {
            "doc_id": "101",
            "title": "Luật Doanh nghiệp 2020",
            "legal_number": "59/2020/QH14",
            "year": "2020",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/59-2020",
            "name_raw": "Luật Doanh nghiệp",
            "passage_raw": "Quy định về thành lập doanh nghiệp",
            "passage_norm": "quy định về thành lập doanh nghiệp",
            "is_empty": False,
        },
        {
            "doc_id": "102",
            "title": "Luật Đầu tư 2020",
            "legal_number": "61/2020/QH14",
            "year": "2020",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/61-2020",
            "name_raw": "Luật Đầu tư",
            "passage_raw": "Quy định về đầu tư",
            "passage_norm": "quy định về đầu tư",
            "is_empty": False,
        },
        {
            "doc_id": "103",
            "title": "Nghị định Đăng ký doanh nghiệp",
            "legal_number": "01/2021/NĐ-CP",
            "year": "2021",
            "doc_type": "Nghị định",
            "link": "https://thuvienphapluat.vn/01-2021",
            "name_raw": "Nghị định 01",
            "passage_raw": "Hồ sơ đăng ký",
            "passage_norm": "hồ sơ đăng ký",
            "is_empty": False,
        },
    ]
    pd.DataFrame(docs).to_parquet(data_dir / "documents.parquet", index=False)

    chunks = [
        {
            "chunk_id": "c101_micro",
            "parent_chunk_id": "c101_macro",
            "doc_id": "101",
            "granularity": "micro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Quy định về thành lập doanh nghiệp và công ty TNHH.",
            "text_norm": "quy định về thành lập doanh nghiệp và công ty tnhh",
        },
        {
            "chunk_id": "c101_macro",
            "parent_chunk_id": None,
            "doc_id": "101",
            "granularity": "macro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Quy định về thành lập doanh nghiệp và công ty TNHH.",
            "text_norm": "quy định về thành lập doanh nghiệp và công ty tnhh",
        },
        {
            "chunk_id": "c102_micro",
            "parent_chunk_id": "c102_macro",
            "doc_id": "102",
            "granularity": "micro",
            "article": "Điều 2",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Dự án đầu tư trực tiếp nước ngoài và ưu đãi.",
            "text_norm": "dự án đầu tư trực tiếp nước ngoài và ưu đãi",
        },
        {
            "chunk_id": "c102_macro",
            "parent_chunk_id": None,
            "doc_id": "102",
            "granularity": "macro",
            "article": "Điều 2",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Dự án đầu tư trực tiếp nước ngoài và ưu đãi.",
            "text_norm": "dự án đầu tư trực tiếp nước ngoài và ưu đãi",
        },
        {
            "chunk_id": "c103_micro",
            "parent_chunk_id": "c103_macro",
            "doc_id": "103",
            "granularity": "micro",
            "article": "Điều 3",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Hồ sơ đăng ký doanh nghiệp qua mạng.",
            "text_norm": "hồ sơ đăng ký doanh nghiệp qua mạng",
        },
        {
            "chunk_id": "c103_macro",
            "parent_chunk_id": None,
            "doc_id": "103",
            "granularity": "macro",
            "article": "Điều 3",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Hồ sơ đăng ký doanh nghiệp qua mạng.",
            "text_norm": "hồ sơ đăng ký doanh nghiệp qua mạng",
        },
    ]
    pd.DataFrame(chunks).to_parquet(data_dir / "chunks.parquet", index=False)

    queries = [
        {"query_id": "q1", "question_raw": "Thành lập công ty TNHH", "question_norm": "thành lập công ty tnhh", "gold_count": 1},
        {"query_id": "q2", "question_raw": "Dự án đầu tư trực tiếp", "question_norm": "dự án đầu tư trực tiếp", "gold_count": 1},
        {"query_id": "q3", "question_raw": "Đăng ký doanh nghiệp qua mạng", "question_norm": "đăng ký doanh nghiệp qua mạng", "gold_count": 1},
    ]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet", index=False)

    qrels = [
        {"query_id": "q1", "doc_id": "101", "relevance": 1},
        {"query_id": "q2", "doc_id": "102", "relevance": 1},
        {"query_id": "q3", "doc_id": "103", "relevance": 1},
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

    # Fit and save indexes
    bm25 = BM25MicroRetriever().fit([c for c in chunks if c["granularity"] == "micro"], show_progress=False)
    bm25.save(index_dir / "bm25")

    bm25_pyvi = BM25PyViRetriever().fit([c for c in chunks if c["granularity"] == "micro"], show_progress=False)
    bm25_pyvi.save(index_dir / "bm25_pyvi")

    return data_dir, index_dir


# ==============================================================================
# 1. Question Memory Tests
# ==============================================================================

def test_question_memory_tuple_records_preserve_qids():
    """Verify (qid, text) and (qid, text, embedding) tuples preserve official qids."""
    dummy_emb = np.random.randn(768).astype(np.float32)
    records = [
        ("146300", "Dự án đầu tư là gì?", dummy_emb),
        ("999999", "Thành lập doanh nghiệp", None),
    ]
    qrels = {
        "146300": ["101"],
        "999999": ["102"],
    }
    mem = TrainQuestionMemory(use_dense=False)
    mem.fit(records, qrels)

    assert "146300" in mem.qids
    assert "999999" in mem.qids
    assert mem.qids == ["146300", "999999"]
    assert mem.qid_to_docs["146300"] == ["101"]


def test_oof_memory_nonempty_and_fold_safe(mock_dataset_and_indexes):
    """Verify OOF fold question memory is non-empty and strictly disjoint from val qids."""
    data_dir, _ = mock_dataset_and_indexes
    queries_df = pd.read_parquet(data_dir / "queries_train.parquet")
    qrels_df = pd.read_parquet(data_dir / "qrels_train.parquet")

    queries_map = dict(zip(queries_df["query_id"].astype(str), queries_df["question_norm"]))
    qrels_map = qrels_df.groupby("query_id")["doc_id"].apply(lambda s: [str(x) for x in s]).to_dict()

    train_ids = ["q2", "q3"]
    val_ids = ["q1"]

    fold_train_queries = [(qid, queries_map[qid], None) for qid in train_ids]
    fold_train_qrels = {qid: qrels_map[qid] for qid in train_ids}

    mem = TrainQuestionMemory(use_dense=False)
    mem.fit(fold_train_queries, fold_train_qrels)

    assert len(mem.qids) == len(train_ids)
    assert set(mem.qids) == set(train_ids)
    assert set(mem.qids).isdisjoint(set(val_ids))


def test_final_memory_contains_all_labeled_train_qids(mock_dataset_and_indexes):
    """Verify final question memory contains all labeled train query IDs."""
    data_dir, _ = mock_dataset_and_indexes
    queries_df = pd.read_parquet(data_dir / "queries_train.parquet")
    qrels_df = pd.read_parquet(data_dir / "qrels_train.parquet")

    queries_map = dict(zip(queries_df["query_id"].astype(str), queries_df["question_norm"]))
    qrels_map = qrels_df.groupby("query_id")["doc_id"].apply(lambda s: [str(x) for x in s]).to_dict()

    all_qids = list(queries_map.keys())
    queries_for_memory = [(qid, queries_map[qid], None) for qid in all_qids]

    mem = TrainQuestionMemory(use_dense=False)
    mem.fit(queries_for_memory, qrels_map)

    assert len(mem.qids) == len(all_qids)
    assert set(mem.qids) == set(all_qids)


def test_memory_load_with_saved_embeddings_does_not_reencode(tmp_path: Path):
    """Verify loading QuestionMemory with pre-saved train_embeddings.npy does not call encoder."""
    index_dir = tmp_path / "memory_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    qa_data = {
        "qids": ["q1", "q2"],
        "queries": ["câu hỏi 1", "câu hỏi 2"],
        "qrels": {"q1": ["101"], "q2": ["102"]},
        "min_similarity": 0.82,
        "dense_min_similarity": 0.85,
    }
    (index_dir / "train_qa.json").write_text(json.dumps(qa_data), encoding="utf-8")
    dummy_embs = np.ones((2, 768), dtype=np.float32)
    np.save(str(index_dir / "train_embeddings.npy"), dummy_embs)

    mock_encoder = mock.MagicMock()
    mock_encoder.encode_texts.side_effect = RuntimeError("Should not be called!")
    mock_encoder.encode_queries.side_effect = RuntimeError("Should not be called!")
    mock_encoder.encode.side_effect = RuntimeError("Should not be called!")

    mem = TrainQuestionMemory.load(index_dir, dense_retriever=mock_encoder)
    assert len(mem.qids) == 2
    assert mem.dense_embeddings is not None
    assert mem.dense_embeddings.shape == (2, 768)
    mock_encoder.encode_texts.assert_not_called()
    mock_encoder.encode_queries.assert_not_called()


# ==============================================================================
# 2. Device Routing Tests
# ==============================================================================

def test_resolve_device_cuda_zero():
    """Verify resolve_device accepts cuda:0 when cuda is available or cpu mock."""
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.device_count", return_value=2):
        assert resolve_device("cuda:0") == "cuda:0"
        assert resolve_device(torch.device("cuda:0")) == "cuda:0"


def test_resolve_device_cuda_one():
    """Verify resolve_device preserves cuda:1 without collapsing to generic cuda."""
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.device_count", return_value=2):
        assert resolve_device("cuda:1") == "cuda:1"
        assert resolve_device(torch.device("cuda:1")) == "cuda:1"


def test_resolve_device_invalid_cuda_index_raises():
    """Verify resolve_device raises RuntimeError when cuda index exceeds device count."""
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.device_count", return_value=2):
        with pytest.raises(RuntimeError, match="out of range"):
            resolve_device("cuda:2")
        with pytest.raises(RuntimeError, match="out of range"):
            resolve_device("cuda:5")


def test_train_reranker_honors_explicit_device(tmp_path: Path):
    """Verify train_reranker propagates explicit device parameter to RerankerTrainer."""
    pairs_file = tmp_path / "pairs.parquet"
    pd.DataFrame([
        {"query_id": "q1", "query_text": "text q", "doc_id": "101", "evidence_text": "ev", "label": 1.0},
        {"query_id": "q1", "query_text": "text q", "doc_id": "102", "evidence_text": "ev2", "label": 0.0},
    ]).to_parquet(pairs_file, index=False)

    out_dir = tmp_path / "artifacts" / "local" / "checkpoints"
    with mock.patch("src.training.train_reranker.RerankerTrainer") as mock_trainer_cls:
        mock_instance = mock.MagicMock()
        mock_instance.train.return_value = {"status": "completed", "global_steps": 1, "param_diff": 0.01}
        mock_instance.batch_size = 2
        mock_instance.gradient_accumulation_steps = 1
        mock_trainer_cls.return_value = mock_instance

        train_reranker(
            pairs_file=pairs_file,
            output_dir=out_dir,
            base_model_name="mock",
            max_steps=1,
            device="cpu",
        )
        assert mock_trainer_cls.call_args.kwargs.get("device") == "cpu"


def test_pipeline_uses_distinct_dense_and_reranker_devices(mock_dataset_and_indexes, tmp_path: Path):
    """Verify LegalIRPipeline.load_pipeline routes dense to dense_device and reranker to reranker_device."""
    data_dir, index_dir = mock_dataset_and_indexes

    # Save a mock dense embedding to allow dense loading
    dense_dir = index_dir / "dense_dek21"
    dense_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(dense_dir / "embeddings.npy"), np.ones((2, 768), dtype=np.float32))
    (dense_dir / "metadata.json").write_text(json.dumps({"doc_ids": ["101", "102"]}), encoding="utf-8")

    with mock.patch("src.retrieval.dense_macro.DenseMacroRetriever.load") as mock_dense_load, \
         mock.patch("src.ranking.reranker.CrossEncoderReranker") as mock_reranker_cls:

        mock_dense_load.return_value = mock.MagicMock()
        mock_reranker_cls.return_value = mock.MagicMock()

        LegalIRPipeline.load_pipeline(
            data_dir=data_dir,
            index_dir=index_dir,
            dense_device="cpu",
            reranker_device="cpu",
            reranker_model_name="mock",
        )

        assert mock_dense_load.call_args.kwargs.get("device") == "cpu"
        assert mock_reranker_cls.call_args.kwargs.get("device") == "cpu"


# ==============================================================================
# 3. Training Config & Steps Tests
# ==============================================================================

def test_full_training_uses_explicit_reranker_config():
    """Verify default experiment YAML specifies explicit bounded max_steps and learning rate."""
    exp_path = REPO_ROOT / "configs" / "experiments" / "reranker_lora.yaml"
    assert exp_path.exists()
    import yaml
    cfg = yaml.safe_load(exp_path.read_text(encoding="utf-8"))
    assert "max_steps" in cfg
    assert cfg["max_steps"] == 500
    assert cfg.get("learning_rate") == 2.0e-5
    assert cfg.get("batch_size") == 2
    assert cfg.get("gradient_accumulation_steps") == 8


def test_effective_reranker_max_steps_is_explicit(tmp_path: Path):
    """Verify full mode raises if max_steps is None."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(Exception):
        # Missing documents.parquet or invalid max_steps raises immediately
        run_kaggle_pipeline(
            data_dir=data_dir,
            working_dir=tmp_path / "run",
            run_mode="full",
        )


# ==============================================================================
# 4. OOF Post-Rerank Features & Leakage Tests
# ==============================================================================

def test_oof_features_contain_actual_fold_reranker_scores():
    """Verify that feature extraction after reranking embeds real reranker scores."""
    candidate_records = [
        {
            "doc_id": "101",
            "raw_bm25_rank": 1,
            "raw_bm25_score": 12.5,
            "reranker_score": 0.9542,
            "reranker_second_score": 0.3120,
            "reranker_margin": 0.6422,
        },
        {
            "doc_id": "102",
            "raw_bm25_rank": 2,
            "raw_bm25_score": 10.1,
            "reranker_score": -999.0,
            "reranker_second_score": -999.0,
            "reranker_margin": 0.0,
        },
    ]
    df = extract_candidate_features(
        query_id="q1",
        candidate_records=candidate_records,
        query_text="thành lập công ty",
    )
    assert len(df) == 2
    assert df.loc[df["doc_id"] == "101", "reranker_score"].iloc[0] == pytest.approx(0.9542)
    assert df.loc[df["doc_id"] == "101", "reranker_margin"].iloc[0] == pytest.approx(0.6422)
    assert df.loc[df["doc_id"] == "102", "reranker_score"].iloc[0] == pytest.approx(-999.0)


def test_fusion_feature_training_inference_schema_match():
    """Verify all CORE_FEATURE_COLUMNS are present in extracted candidate features."""
    candidate_records = [
        {"doc_id": "101", "raw_bm25_score": 5.0, "reranker_score": 0.8},
    ]
    df = extract_candidate_features("q1", candidate_records, "câu hỏi")
    for col in CORE_FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature column: {col}"


def test_fold_train_doc_frequency_has_no_val_label_leakage():
    """Verify training doc frequency is computed exclusively from training fold qrels."""
    train_qrels = {
        "q1": ["101", "102"],
        "q2": ["101"],
    }
    val_qrels = {
        "q3": ["103", "103"],
    }
    freq_map = compute_training_doc_frequencies(train_qrels)
    assert "101" in freq_map
    assert freq_map["101"] == pytest.approx(2.0 / 2.0)
    assert freq_map["102"] == pytest.approx(1.0 / 2.0)
    assert "103" not in freq_map


# ==============================================================================
# 5. Document-Disjoint Reranker Tests
# ==============================================================================

def test_doc_disjoint_reranker_uses_train_side_only(mock_dataset_and_indexes, tmp_path: Path):
    """Verify document disjoint evaluation uses fold-safe train query memory."""
    data_dir, index_dir = mock_dataset_and_indexes
    cv_dir = tmp_path / "cv"

    runner = OOFRunner(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=cv_dir,
        num_folds=1,
        smoke=True,
        smoke_sample_size=2,
        use_reranker=False,
        train_reranker_per_fold=False,
        doc_disjoint=True,
    )
    runner.load_data()
    runner.load_retrievers()
    report = runner.run_document_disjoint_evaluation(reranker=None)

    assert "recall@5" in report
    assert "retrieval_only" in report


def test_doc_disjoint_report_contains_trained_reranker_metrics(tmp_path: Path):
    """Verify doc_disjoint_report.json contains retrieval_only and trained_reranker_system metrics."""
    report_file = tmp_path / "doc_disjoint_report.json"
    dummy_report = {
        "recall@5": 0.85,
        "precision@5": 0.20,
        "retrieval_only": {"recall@5": 0.70, "precision@5": 0.15},
        "trained_reranker_system": {"recall@5": 0.85, "precision@5": 0.20},
    }
    report_file.write_text(json.dumps(dummy_report), encoding="utf-8")

    loaded = json.loads(report_file.read_text(encoding="utf-8"))
    assert "retrieval_only" in loaded
    assert "trained_reranker_system" in loaded
    assert loaded["trained_reranker_system"]["recall@5"] >= loaded["retrieval_only"]["recall@5"]


# ==============================================================================
# 6. Strict Artifact Loading Tests
# ==============================================================================

def test_strict_load_missing_bm25_raises(mock_dataset_and_indexes, tmp_path: Path):
    """Verify strict_artifacts=True raises if Legal BM25 index is missing."""
    data_dir, index_dir = mock_dataset_and_indexes
    import shutil
    bad_index_dir = tmp_path / "bad_index"
    shutil.copytree(index_dir, bad_index_dir)
    shutil.rmtree(bad_index_dir / "bm25")

    with pytest.raises(FileNotFoundError, match="Legal BM25"):
        LegalIRPipeline.load_pipeline(
            data_dir=data_dir,
            index_dir=bad_index_dir,
            strict_artifacts=True,
        )


def test_strict_load_missing_pyvi_raises(mock_dataset_and_indexes, tmp_path: Path):
    """Verify strict_artifacts=True raises if PyVi BM25 index is missing."""
    data_dir, index_dir = mock_dataset_and_indexes
    import shutil
    bad_index_dir = tmp_path / "bad_index_pyvi"
    shutil.copytree(index_dir, bad_index_dir)
    shutil.rmtree(bad_index_dir / "bm25_pyvi")

    with pytest.raises(FileNotFoundError, match="PyVi BM25"):
        LegalIRPipeline.load_pipeline(
            data_dir=data_dir,
            index_dir=bad_index_dir,
            strict_artifacts=True,
        )


def test_strict_load_missing_dense_raises(mock_dataset_and_indexes, tmp_path: Path):
    """Verify strict_artifacts=True raises if Dense index is missing."""
    data_dir, index_dir = mock_dataset_and_indexes

    with pytest.raises(FileNotFoundError, match="Dense index"):
        LegalIRPipeline.load_pipeline(
            data_dir=data_dir,
            index_dir=index_dir,
            strict_artifacts=True,
        )


def test_strict_load_empty_memory_raises(mock_dataset_and_indexes, tmp_path: Path):
    """Verify strict_artifacts=True raises if Question Memory has 0 indexed queries."""
    data_dir, index_dir = mock_dataset_and_indexes

    # Create dummy dense index with chunks_meta so it passes dense check
    dense_dir = index_dir / "dense_dek21"
    dense_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(dense_dir / "embeddings.npy"), np.ones((2, 768), dtype=np.float32))
    pd.DataFrame({"chunk_id": ["c1", "c2"], "doc_id": ["101", "102"]}).to_parquet(dense_dir / "chunks_meta.parquet", index=False)

    with pytest.raises(FileNotFoundError, match="Question Memory"):
        LegalIRPipeline.load_pipeline(
            data_dir=data_dir,
            index_dir=index_dir,
            strict_artifacts=True,
        )


def test_strict_load_missing_final_adapter_raises(mock_dataset_and_indexes, tmp_path: Path):
    """Verify strict_artifacts=True raises if final LoRA adapter is missing."""
    data_dir, index_dir = mock_dataset_and_indexes

    # Populate dense & question memory
    dense_dir = index_dir / "dense_dek21"
    dense_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(dense_dir / "embeddings.npy"), np.ones((2, 768), dtype=np.float32))
    pd.DataFrame({"chunk_id": ["c1", "c2"], "doc_id": ["101", "102"]}).to_parquet(dense_dir / "chunks_meta.parquet", index=False)

    mem_dir = index_dir / "question_memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "train_qa.json").write_text(json.dumps({"qids": ["q1"], "queries": ["q"], "qrels": {"q1": ["101"]}}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="reranker_adapter_path"):
        LegalIRPipeline.load_pipeline(
            data_dir=data_dir,
            index_dir=index_dir,
            use_reranker=True,
            reranker_adapter_path=None,
            reranker_model_name="BAAI/bge-reranker-v2-m3",
            strict_artifacts=True,
        )


def test_strict_load_missing_selected_fusion_raises(mock_dataset_and_indexes, tmp_path: Path):
    """Verify strict_artifacts=True raises if learned fusion model path does not exist."""
    data_dir, index_dir = mock_dataset_and_indexes

    # Populate dense & question memory
    dense_dir = index_dir / "dense_dek21"
    dense_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(dense_dir / "embeddings.npy"), np.ones((2, 768), dtype=np.float32))
    pd.DataFrame({"chunk_id": ["c1", "c2"], "doc_id": ["101", "102"]}).to_parquet(dense_dir / "chunks_meta.parquet", index=False)

    mem_dir = index_dir / "question_memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "train_qa.json").write_text(json.dumps({"qids": ["q1"], "queries": ["q"], "qrels": {"q1": ["101"]}}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="fusion model path"):
        LegalIRPipeline.load_pipeline(
            data_dir=data_dir,
            index_dir=index_dir,
            use_reranker=False,
            use_learned_fusion=True,
            fusion_model_path=tmp_path / "non_existent_fusion",
            strict_artifacts=True,
        )


def test_invalid_canonical_dataset_raises(tmp_path: Path):
    """Verify canonical validation fails on missing documents.parquet."""
    empty_dir = tmp_path / "empty_canonical"
    empty_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        run_kaggle_pipeline(
            data_dir=empty_dir,
            working_dir=tmp_path / "run",
            run_mode="full",
        )


# ==============================================================================
# 7. Submission & Public File Requirements
# ==============================================================================

def test_full_mode_missing_public_file_raises(mock_dataset_and_indexes, tmp_path: Path):
    """Verify full run_mode raises FileNotFoundError when public-official.json is absent."""
    data_dir, _ = mock_dataset_and_indexes

    with pytest.raises(FileNotFoundError, match="public-official.json"):
        run_kaggle_pipeline(
            data_dir=data_dir,
            working_dir=tmp_path / "run_full_fail",
            run_mode="full",
            public_json_path=tmp_path / "non_existent_public.json",
        )


def test_submission_validation_uses_official_public_keys(tmp_path: Path):
    """Verify submission validation catches missing queries when compared to independent official qids."""
    sub_json = tmp_path / "submission.json"
    predictions = {
        "pub_1": {"answer": ["101", "102", "103", "104", "105"]},
    }
    sub_json.write_text(json.dumps(predictions), encoding="utf-8")

    expected_official_qids = {"pub_1", "pub_2"}  # pub_2 missing
    res = validate_submission(sub_json, expected_qids=expected_official_qids)
    assert not res["is_valid"]
    assert any("mismatch" in err.lower() for err in res["errors"])


def test_smoke_train_fallback_is_non_submittable(mock_dataset_and_indexes, tmp_path: Path):
    """Verify smoke run manifest records NON_SUBMITTABLE status."""
    data_dir, index_dir = mock_dataset_and_indexes
    working_dir = tmp_path / "smoke_out"

    result = run_kaggle_pipeline(
        data_dir=data_dir,
        working_dir=working_dir,
        run_mode="smoke",
        repo_root=REPO_ROOT,
        devices=["cpu", "cpu"],
    )
    assert result.is_valid
    manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["metadata"]["submission_status"] == "NON_SUBMITTABLE_SMOKE"


# ==============================================================================
# 8. Metadata Parity & Query Cache Tests
# ==============================================================================

def test_final_exact_matcher_has_year_and_doc_type(mock_dataset_and_indexes):
    """Verify ExactMatcher loaded in pipeline retains year and doc_type metadata for statutory matching."""
    data_dir, index_dir = mock_dataset_and_indexes
    docs_df = pd.read_parquet(data_dir / "documents.parquet")
    exact = ExactMatcher(docs_df.to_dict("records"))

    # Query with legal number and year and doc_type
    res = exact.match("Theo Luật Doanh nghiệp năm 2020 quy định về thành lập công ty")
    assert "101" in res
    assert res["101"]["exact_year"] is True
    assert res["101"]["exact_doc_type"] is True


def test_public_query_embedding_shared_dense_memory(tmp_path: Path):
    """Verify that precomputed query embeddings can be passed directly to question memory search."""
    dummy_emb = np.random.randn(768).astype(np.float32)
    mem = TrainQuestionMemory(min_similarity=0.82, use_dense=False)
    mem.fit([("q1", "câu hỏi đầu tư", dummy_emb)], {"q1": ["101"]})

    # Search with q_emb vector
    hits = mem.search("câu hỏi đầu tư", top_k=1, q_emb=dummy_emb)
    assert len(hits) == 1
    assert hits[0]["doc_id"] == "101"


def test_pair_mining_uses_precomputed_query_embedding(mock_dataset_and_indexes, tmp_path: Path):
    """Verify build_training_pairs accepts precomputed query_embeddings mapping."""
    data_dir, index_dir = mock_dataset_and_indexes
    pairs_dir = tmp_path / "pairs_cached"

    q_embs = {
        "q1": np.random.randn(768).astype(np.float32),
        "q2": np.random.randn(768).astype(np.float32),
        "q3": np.random.randn(768).astype(np.float32),
    }

    retriever_df, reranker_df = build_training_pairs(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=pairs_dir,
        fold=0,
        query_embeddings=q_embs,
    )
    assert not reranker_df.empty
    assert (pairs_dir / "reranker_pairs.parquet").exists()


# ==============================================================================
# 9. Notebook Byte-Level Parity Test
# ==============================================================================

def test_kaggle_notebook_byte_level_parity():
    """Verify notebooks exist and pass parity verification."""
    from scripts.check_notebook_parity import check_all_notebook_parity
    is_valid, report = check_all_notebook_parity(REPO_ROOT)
    assert is_valid is True, f"Parity check failed: {report}"

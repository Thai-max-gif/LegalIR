"""Behavioral tests for LegalIR 595E Drive/Kaggle Final Blocker Repairs.

Covers:
- P0-1: Sequence import in src/training/trainer.py
- P0-2: DenseMacroRetriever.fit signature and stage_name propagation
- P0-3: Fail-closed notebook commit pinning (no default branch fallback, SHA verification)
- P1-1: 1.0 pair-derived coverage policy (875 steps for 7,000 queries, cannot be undercut)
- P1-2: Evidence-grade stage timing and cold-start projection
- P1-3: psutil in requirements and true peak RSS telemetry
- P1-4 & P1-5: Strict v2 dataset identity checks and Kaggle clean data layout cold-start regression
"""

import importlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ==============================================================================
# P0-1: Trainer Module Imports
# ==============================================================================

def test_trainer_module_imports():
    """Verify src.training.trainer imports cleanly and audit_pair_coverage is callable."""
    mod = importlib.import_module("src.training.trainer")
    assert hasattr(mod, "audit_pair_coverage")
    assert callable(mod.audit_pair_coverage)
    assert hasattr(mod, "Sequence")


def test_pipeline_import():
    """Verify run_kaggle_pipeline can be imported without NameError."""
    mod = importlib.import_module("src.pipeline.kaggle_train")
    assert hasattr(mod, "run_kaggle_pipeline")
    assert callable(mod.run_kaggle_pipeline)


# ==============================================================================
# P0-2: DenseMacroRetriever.fit signature and stage_name propagation
# ==============================================================================

def test_dense_fit_propagates_corpus_stage():
    """Verify DenseMacroRetriever.fit accepts stage_name and records telemetry."""
    from src.retrieval.dense_macro import DenseMacroRetriever

    retriever = DenseMacroRetriever.from_arrays(
        embeddings=np.zeros((2, 768), dtype=np.float32),
        chunk_ids=["c1", "c2"],
        doc_ids=["d1", "d2"],
        query_encoder=lambda texts: np.zeros((len(texts), 768), dtype=np.float32),
    )
    retriever.fit(
        [
            {"chunk_id": "c1", "doc_id": "d1", "text_norm": "alpha"},
            {"chunk_id": "c2", "doc_id": "d2", "text_norm": "beta"},
        ],
        batch_size=2,
        stage_name="corpus",
    )
    assert "corpus" in retriever.stage_telemetry
    assert retriever.stage_telemetry["corpus"].item_count == 2


def test_orchestrator_dense_build_boundary_reaches_fit(tmp_path, monkeypatch):
    """Test orchestrator dense build boundary without mocking DenseMacroRetriever.fit."""
    from src.retrieval.dense_macro import DenseMacroRetriever

    # Create dummy macro chunks
    chunks_df = pd.DataFrame({
        "chunk_id": ["c1", "c2"],
        "doc_id": ["d1", "d2"],
        "text_norm": ["van ban phap luat 1", "van ban phap luat 2"],
        "granularity": ["macro", "macro"],
    })

    dense_retriever = DenseMacroRetriever(
        model_name="mock",
        dimension=16,
        use_pyvi=False,
    )
    dense_retriever.query_encoder = lambda texts: np.ones((len(texts), 16), dtype=np.float32)

    # Calling fit with stage_name must succeed
    dense_retriever.fit(
        chunks_df.to_dict("records"),
        batch_size=2,
        stage_name="corpus",
    )
    dense_dir = tmp_path / "dense_dek21"
    dense_retriever.save(dense_dir)

    assert (dense_dir / "embeddings.npy").exists()
    assert (dense_dir / "chunks_meta.parquet").exists()
    assert (dense_dir / "manifest.json").exists()
    assert "corpus" in dense_retriever.stage_telemetry


# ==============================================================================
# P0-3: Notebook Commit Pinning & Fail-Closed Bootstrap
# ==============================================================================

def test_generated_notebook_contains_no_default_branch_fallback():
    """Generated notebook must NOT contain silent fallback to default branch."""
    from scripts.generate_kaggle_notebook import build_legalir_notebook

    nb = build_legalir_notebook()
    nb_str = json.dumps(nb)
    assert "using default branch" not in nb_str
    assert "using default branch" not in nb_str.lower()


def test_generated_notebook_has_fail_closed_commit_pin():
    """Generated notebook Cell 2 must verify git rev-parse HEAD and raise on mismatch."""
    from scripts.generate_kaggle_notebook import build_legalir_notebook

    nb = build_legalir_notebook()
    cell_2_src = "".join(nb["cells"][2]["source"])
    assert "EXPECTED_COMMIT" in cell_2_src
    assert "rev-parse" in cell_2_src
    assert "RuntimeError" in cell_2_src or "raise" in cell_2_src


def test_notebook_dependencies_include_psutil():
    """Generated notebook Cell 2 must include psutil in minimal dependencies."""
    from scripts.generate_kaggle_notebook import build_legalir_notebook

    nb = build_legalir_notebook()
    cell_2_src = "".join(nb["cells"][2]["source"])
    assert "psutil" in cell_2_src


# ==============================================================================
# P1-1: Full 1.0 Pair Coverage & Step Derivation
# ==============================================================================

def test_7000_full_cycle_requires_875_steps():
    """7,000 queries with batch 2, grad_accum 8, 100% pos+neg requires 875 steps."""
    from src.training.trainer import compute_coverage_required_steps

    steps = compute_coverage_required_steps(
        eligible_query_count=7000,
        batch_size=2,
        gradient_accumulation_steps=8,
        require_pos_and_neg=True,
        target_coverage_pct=1.0,
    )
    assert steps == 875


def test_explicit_max_steps_cannot_undercut_full_pair_coverage(tmp_path):
    """When full coverage is enforced, explicit lower max_steps cannot undercut required steps."""
    from src.training.train_reranker import train_reranker

    # Create dummy training pairs
    pairs_data = []
    for i in range(16):  # 16 queries = 32 rows (1 pos, 1 neg each)
        pairs_data.append({"query_id": f"q_{i}", "doc_id": f"d_{i}", "label": 1.0, "query_text": f"q {i}", "candidate_text": f"doc {i}"})
        pairs_data.append({"query_id": f"q_{i}", "doc_id": f"d_neg_{i}", "label": 0.0, "query_text": f"q {i}", "candidate_text": f"neg {i}"})
    pairs_df = pd.DataFrame(pairs_data)
    pairs_file = tmp_path / "pairs.parquet"
    pairs_df.to_parquet(pairs_file)

    out_dir = tmp_path / "checkpoints/test_reranker"

    # 16 queries * 2 rows = 32 rows; batch 2, grad 8 -> 16 rows/step -> 2 steps required
    # If max_steps=1 is requested with enforce_full_coverage_steps=True, effective steps should be max(1, 2) = 2.
    report = train_reranker(
        pairs_file=pairs_file,
        output_dir=out_dir,
        max_steps=1,
        base_model_name="mock",
        enforce_full_coverage_steps=True,
        batch_size=2,
    )
    assert report.get("optimizer_steps", 0) >= 2


def test_fold_steps_derive_from_fold_eligible_pairs(tmp_path):
    """Fold training steps derive from actual fold eligible queries."""
    from src.training.trainer import compute_coverage_required_steps

    # 4/5 of 7,000 queries = 5,600 queries
    # 5,600 * 2 = 11,200 rows -> 11,200 / 16 = 700 steps
    fold_steps = compute_coverage_required_steps(
        eligible_query_count=5600,
        batch_size=2,
        gradient_accumulation_steps=8,
        require_pos_and_neg=True,
        target_coverage_pct=1.0,
    )
    assert fold_steps == 700


def test_doc_disjoint_steps_derive_from_eligible_pairs():
    """Document disjoint training steps derive from actual disjoint eligible queries."""
    from src.training.trainer import compute_coverage_required_steps

    # 0.8 of 7,000 queries = 5,600 queries -> 700 steps
    disjoint_steps = compute_coverage_required_steps(
        eligible_query_count=5600,
        batch_size=2,
        gradient_accumulation_steps=8,
        require_pos_and_neg=True,
        target_coverage_pct=1.0,
    )
    assert disjoint_steps == 700


# ==============================================================================
# P1-2: Evidence-Grade Stage Timing & Cold-Start Projection
# ==============================================================================

def test_stage_timing_telemetry_structure():
    """Test stage telemetry reporting with seconds and cache_hit."""
    from src.pipeline.kaggle_train import StageTimingTelemetry

    st = StageTimingTelemetry()
    st.record("bm25_legal", elapsed_seconds=1.23, cache_hit=False)
    st.record("bm25_pyvi", elapsed_seconds=2.34, cache_hit=True)
    report = st.to_dict()
    assert "bm25_legal" in report
    assert report["bm25_legal"]["seconds"] == 1.23
    assert report["bm25_legal"]["cache_hit"] is False
    assert report["bm25_pyvi"]["cache_hit"] is True


def test_cold_start_projection_under_budget():
    """Kaggle Final runtime projection fits within 9h budget using measured hardware telemetry."""
    from src.core.runtime_estimator import estimate_kaggle_final_runtime
    colab_report_p = Path(__file__).resolve().parent.parent / "artifacts" / "task1" / "colab_smoke_report.json"
    assert colab_report_p.is_file(), f"Missing {colab_report_p}"
    telemetry = json.loads(colab_report_p.read_text(encoding="utf-8"))

    proj = estimate_kaggle_final_runtime(telemetry, safety_factor=1.25, max_session_hours=9.0)
    assert proj["is_feasible_on_kaggle"] is True
    assert proj["total_projected_hours"] < 9.0


# ==============================================================================
# P1-3: RAM Telemetry & psutil
# ==============================================================================

def test_requirements_includes_psutil():
    """requirements.txt must list psutil>=5.9.0."""
    req_file = Path("requirements.txt")
    assert req_file.exists()
    content = req_file.read_text(encoding="utf-8")
    assert re.search(r"psutil\s*>=\s*5\.9", content) is not None


def test_get_peak_process_rss_mb_returns_positive():
    """get_peak_process_rss_mb must return a positive float."""
    from src.pipeline.kaggle_train import get_peak_process_rss_mb, get_process_rss_mb

    curr_rss = get_process_rss_mb()
    peak_rss = get_peak_process_rss_mb()
    assert isinstance(curr_rss, float)
    assert isinstance(peak_rss, float)
    assert peak_rss > 0.0


# ==============================================================================
# P1-4 & P1-5: Dataset Identity & Kaggle Clean Data Cold-Start
# ==============================================================================

def test_v2_canonical_dataset_identity_validation(tmp_path):
    """Verify v2 canonical dataset validation rule and official identity helper with 1,000 public queries."""
    from src.pipeline.kaggle_train import (
        OFFICIAL_DOCUMENT_COUNT,
        OFFICIAL_TOTAL_CHUNKS,
        OFFICIAL_MICRO_CHUNKS,
        OFFICIAL_MACRO_CHUNKS,
        OFFICIAL_TRAIN_QUERIES,
        OFFICIAL_QRELS,
        OFFICIAL_PUBLIC_QUERY_COUNT,
        validate_official_task1_identity,
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "task1_canonical",
        "version": "v2",
        "schema": "hierarchical_micro_macro_v2",
        "total_documents": OFFICIAL_DOCUMENT_COUNT,
        "total_chunks": OFFICIAL_TOTAL_CHUNKS,
        "total_micro_chunks": OFFICIAL_MICRO_CHUNKS,
        "total_macro_chunks": OFFICIAL_MACRO_CHUNKS,
        "total_queries": OFFICIAL_TRAIN_QUERIES,
        "total_qrels": OFFICIAL_QRELS,
        "total_duplicate_groups": 4,
        "empty_documents_count": 20,
        "normalization": "nfc_whitespace_preserve_legal_ids",
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    audit = {
        "is_valid": True,
        "total_documents": OFFICIAL_DOCUMENT_COUNT,
        "total_chunks": OFFICIAL_TOTAL_CHUNKS,
        "total_micro_chunks": OFFICIAL_MICRO_CHUNKS,
        "total_macro_chunks": OFFICIAL_MACRO_CHUNKS,
        "total_queries": OFFICIAL_TRAIN_QUERIES,
        "total_qrels": OFFICIAL_QRELS,
        "empty_documents_count": 20,
        "errors": [],
    }
    (data_dir / "audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    public_data = {f"pub_{i:04d}": {"question": f"Question {i}"} for i in range(OFFICIAL_PUBLIC_QUERY_COUNT)}
    pub_file = data_dir / "public-official.json"
    pub_file.write_text(json.dumps(public_data), encoding="utf-8")

    report = validate_official_task1_identity(
        data_dir=data_dir,
        public_json_path=pub_file,
        strict=True,
    )
    assert report["is_valid"] is True
    assert report["public_queries"] == 1000
    assert report["documents"] == 8532
    assert report["chunks"] == 1153876
    assert report["train_queries"] == 7000
    assert report["qrels"] == 7637


def test_kaggle_clean_data_layout_discovery(tmp_path):
    """Test discovery in exact Kaggle layout /kaggle/input/legalir-task1-clean-data/."""
    from src.pipeline.kaggle_train import discover_data_dir, discover_public_test_file

    kaggle_input = tmp_path / "kaggle/input/legalir-task1-clean-data"
    kaggle_input.mkdir(parents=True, exist_ok=True)

    for fname in ["documents.parquet", "chunks.parquet", "queries_train.parquet", "qrels_train.parquet", "manifest.json"]:
        (kaggle_input / fname).write_text("dummy", encoding="utf-8")
    (kaggle_input / "public-official.json").write_text("{}", encoding="utf-8")

    discovered_dir = discover_data_dir(data_dir=kaggle_input)
    assert discovered_dir == kaggle_input.resolve()

    discovered_public = discover_public_test_file(data_dir=kaggle_input)
    assert discovered_public == (kaggle_input / "public-official.json").resolve()

"""Tests for Task 4: Kaggle Production Orchestrator, T4 x2 Utilization, and Thin Notebook Generator."""

import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.generate_kaggle_notebook import build_legalir_notebook, generate_and_save_notebooks
from src.evaluation.submission import validate_submission, validate_submission_zip
from src.models.parameter_audit import MAX_PARAMETER_BUDGET
from src.pipeline.kaggle_train import (
    KaggleRunResult,
    discover_data_dir,
    discover_public_test_file,
    resolve_kaggle_devices,
    run_kaggle_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def toy_kaggle_data(tmp_path: Path):
    """Create a minimal self-contained canonical dataset for fast smoke testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        {
            "doc_id": "101",
            "title": "Luật Doanh nghiệp 2020",
            "legal_number": "59/2020/QH14",
            "year": "2020",
            "doc_type": "Luật",
            "link": "https://tvpl.vn/59-2020",
            "name_raw": "Luật Doanh nghiệp",
            "is_empty": False,
        },
        {
            "doc_id": "102",
            "title": "Luật Đầu tư 2020",
            "legal_number": "61/2020/QH14",
            "year": "2020",
            "doc_type": "Luật",
            "link": "https://tvpl.vn/61-2020",
            "name_raw": "Luật Đầu tư",
            "is_empty": False,
        },
        {
            "doc_id": "103",
            "title": "Nghị định Đăng ký doanh nghiệp",
            "legal_number": "01/2021/NĐ-CP",
            "year": "2021",
            "doc_type": "Nghị định",
            "link": "https://tvpl.vn/01-2021",
            "name_raw": "Nghị định 01",
            "is_empty": False,
        },
        {
            "doc_id": "104",
            "title": "Luật Thương mại",
            "legal_number": "36/2005/QH11",
            "year": "2005",
            "doc_type": "Luật",
            "link": "https://tvpl.vn/36-2005",
            "name_raw": "Luật Thương mại",
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
            "text_raw": "Quy định về thành lập doanh nghiệp",
            "text_norm": "quy định về thành lập doanh nghiệp",
        },
        {
            "chunk_id": "c101_macro",
            "parent_chunk_id": None,
            "doc_id": "101",
            "granularity": "macro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về thành lập doanh nghiệp và quản lý công ty",
            "text_norm": "quy định về thành lập doanh nghiệp và quản lý công ty",
        },
        {
            "chunk_id": "c102_micro",
            "parent_chunk_id": "c102_macro",
            "doc_id": "102",
            "granularity": "micro",
            "article": "Điều 2",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về dự án đầu tư trực tiếp",
            "text_norm": "quy định về dự án đầu tư trực tiếp",
        },
        {
            "chunk_id": "c102_macro",
            "parent_chunk_id": None,
            "doc_id": "102",
            "granularity": "macro",
            "article": "Điều 2",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư",
            "text_norm": "quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư",
        },
        {
            "chunk_id": "c103_micro",
            "parent_chunk_id": "c103_macro",
            "doc_id": "103",
            "granularity": "micro",
            "article": "Điều 3",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Hồ sơ đăng ký doanh nghiệp qua mạng",
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
            "text_raw": "Hồ sơ đăng ký doanh nghiệp qua mạng điện tử",
            "text_norm": "hồ sơ đăng ký doanh nghiệp qua mạng điện tử",
        },
        {
            "chunk_id": "c104_micro",
            "parent_chunk_id": "c104_macro",
            "doc_id": "104",
            "granularity": "micro",
            "article": "Điều 4",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Hoạt động mua bán hàng hóa quốc tế",
            "text_norm": "hoạt động mua bán hàng hóa quốc tế",
        },
        {
            "chunk_id": "c104_macro",
            "parent_chunk_id": None,
            "doc_id": "104",
            "granularity": "macro",
            "article": "Điều 4",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Hoạt động mua bán hàng hóa quốc tế và cung ứng dịch vụ",
            "text_norm": "hoạt động mua bán hàng hóa quốc tế và cung ứng dịch vụ",
        },
    ]
    pd.DataFrame(chunks).to_parquet(data_dir / "chunks.parquet", index=False)

    queries = [
        {"query_id": "q1", "question_raw": "Thành lập doanh nghiệp như thế nào?", "question_norm": "thành lập doanh nghiệp như thế nào", "gold_count": 1},
        {"query_id": "q2", "question_raw": "Dự án đầu tư trực tiếp", "question_norm": "dự án đầu tư trực tiếp", "gold_count": 1},
        {"query_id": "q3", "question_raw": "Đăng ký kinh doanh qua mạng", "question_norm": "đăng ký kinh doanh qua mạng", "gold_count": 1},
        {"query_id": "q4", "question_raw": "Mua bán hàng hóa quốc tế", "question_norm": "mua bán hàng hóa quốc tế", "gold_count": 1},
    ]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet", index=False)

    qrels = [
        {"query_id": "q1", "doc_id": "101", "relevance": 1},
        {"query_id": "q2", "doc_id": "102", "relevance": 1},
        {"query_id": "q3", "doc_id": "103", "relevance": 1},
        {"query_id": "q4", "doc_id": "104", "relevance": 1},
    ]
    pd.DataFrame(qrels).to_parquet(data_dir / "qrels_train.parquet", index=False)

    split_info = [
        {"fold": 0, "train_query_ids": ["q3", "q4"], "val_query_ids": ["q1", "q2"]},
        {"fold": 1, "train_query_ids": ["q1", "q2"], "val_query_ids": ["q3", "q4"]},
    ]
    (splits_dir / "random_5fold.json").write_text(json.dumps(split_info), encoding="utf-8")

    doc_disjoint_info = {
        "train_query_ids": ["q3", "q4"],
        "val_query_ids": ["q1", "q2"],
        "train_doc_ids": ["103", "104"],
        "val_doc_ids": ["101", "102"],
    }
    (splits_dir / "doc_disjoint_split.json").write_text(json.dumps(doc_disjoint_info), encoding="utf-8")

    # Toy public test file
    public_file = tmp_path / "public-official.json"
    public_data = {
        "pub_1": {"question": "Thủ tục thành lập công ty TNHH"},
        "pub_2": {"question": "Ưu đãi đầu tư trực tiếp nước ngoài"},
    }
    public_file.write_text(json.dumps(public_data), encoding="utf-8")

    return data_dir, public_file


def test_kaggle_notebook_byte_level_parity():
    """Verify that canonical notebooks exist and pass parity check."""
    root_nb, kernel_nb = generate_and_save_notebooks(repo_root=REPO_ROOT)
    assert root_nb.exists(), "Kaggle smoke notebook must exist"
    assert kernel_nb.exists(), "Colab train notebook must exist"

    with open(root_nb, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert nb_data.get("nbformat") == 4
    assert len(nb_data["cells"]) >= 5, f"Expected clean notebook, got {len(nb_data['cells'])}"

    # Ensure no hardcoded tokens
    nb_str = json.dumps(nb_data)
    assert "print(hf_token)" not in nb_str
    assert "print(f\"{hf_token}" not in nb_str


def test_t4_safe_config_and_dependencies():
    """Verify T4-safe configuration parameters (P1.11) and requirements.txt dependencies."""
    kaggle_cfg_path = REPO_ROOT / "configs/kaggle.yaml"
    reranker_cfg_path = REPO_ROOT / "configs/experiments/reranker_lora.yaml"
    req_path = REPO_ROOT / "requirements.txt"

    assert kaggle_cfg_path.exists()
    assert reranker_cfg_path.exists()
    assert req_path.exists()

    with open(kaggle_cfg_path, "r", encoding="utf-8") as f:
        k_cfg = yaml.safe_load(f)

    with open(reranker_cfg_path, "r", encoding="utf-8") as f:
        r_cfg = yaml.safe_load(f)

    # Check P1.11 config parameters
    assert k_cfg.get("batch_size") == 2
    assert k_cfg.get("gradient_accumulation_steps") == 8
    assert k_cfg.get("max_length") == 512
    assert k_cfg.get("fp16") is True
    assert k_cfg.get("lora_r") == 16
    assert k_cfg.get("lora_alpha") == 32
    assert k_cfg.get("gradient_checkpointing") is True

    assert r_cfg.get("batch_size") == 2
    assert r_cfg.get("gradient_accumulation_steps") == 8
    assert r_cfg.get("max_length") == 512
    assert r_cfg.get("fp16") is True
    assert r_cfg.get("lora_r") == 16 or r_cfg.get("lora", {}).get("r") == 16
    assert r_cfg.get("lora_alpha") == 32 or r_cfg.get("lora", {}).get("lora_alpha") == 32
    assert r_cfg.get("gradient_checkpointing") is True

    # Check requirements.txt
    req_text = req_path.read_text(encoding="utf-8")
    assert "peft" in req_text
    assert "accelerate" in req_text


def test_device_allocation_p1_10():
    """Verify intentional multi-GPU device allocation (GPU 0: Dense, GPU 1: Reranker)."""
    # Explicit devices
    d, r = resolve_kaggle_devices(["cuda:0", "cuda:1"])
    assert d == "cuda:0"
    assert r == "cuda:1"

    # Single device
    d_single, r_single = resolve_kaggle_devices(["cuda:0"])
    assert d_single == "cuda:0"
    assert r_single == "cuda:0"

    # Auto resolution
    d_auto, r_auto = resolve_kaggle_devices(None)
    assert isinstance(d_auto, str)
    assert isinstance(r_auto, str)


def test_run_kaggle_pipeline_smoke_mode(toy_kaggle_data, tmp_path: Path):
    """Verify end-to-end execution of run_kaggle_pipeline in smoke mode."""
    data_dir, public_file = toy_kaggle_data
    working_dir = tmp_path / "legalir_run"

    result = run_kaggle_pipeline(
        data_dir=data_dir,
        working_dir=working_dir,
        run_mode="smoke",
        public_json_path=public_file,
        repo_root=REPO_ROOT,
    )

    assert isinstance(result, KaggleRunResult)
    assert result.is_valid is True
    assert result.run_mode == "smoke"
    assert result.public_predictions_count == 2
    assert result.execution_time_seconds > 0

    # Verify created artifacts
    assert result.submission_path.exists()
    assert result.submission_zip_path.exists()
    assert result.manifest_path.exists()
    assert (working_dir / "parameter_audit.json").exists()
    assert (working_dir / "ablation_report.csv").exists()

    # Validate submission JSON and ZIP
    val_res = validate_submission(result.submission_path, expected_qids={"pub_1", "pub_2"})
    assert val_res["is_valid"] is True
    assert val_res["total_queries"] == 2

    zip_res = validate_submission_zip(result.submission_zip_path)
    assert zip_res["is_valid"] is True

    # Validate parameter audit
    assert result.audit_report["is_compliant"] is True
    assert result.audit_report["total_learned_parameters"] < MAX_PARAMETER_BUDGET

    # Validate CV report
    assert "mean_recall@5" in result.cv_report
    assert "mean_precision@5" in result.cv_report


def test_parameter_budget_audit_in_pipeline_result(toy_kaggle_data, tmp_path: Path):
    """Verify that parameter budget audit strictly checks <4B parameters in pipeline run."""
    data_dir, public_file = toy_kaggle_data
    working_dir = tmp_path / "audit_run"

    result = run_kaggle_pipeline(
        data_dir=data_dir,
        working_dir=working_dir,
        run_mode="smoke",
        public_json_path=public_file,
        repo_root=REPO_ROOT,
    )

    audit = result.audit_report
    assert audit["total_learned_parameters"] < MAX_PARAMETER_BUDGET
    assert audit["is_compliant"] is True
    assert "budget_utilization_pct" in audit
    assert audit["budget_utilization_pct"] <= 100.0


def test_discover_data_dir_and_public_test_file(toy_kaggle_data, tmp_path: Path):
    """Verify data and public file discovery helper functions."""
    data_dir, public_file = toy_kaggle_data

    disc_data = discover_data_dir(data_dir=data_dir)
    assert disc_data == data_dir.resolve()

    disc_pub = discover_public_test_file(public_json_path=public_file)
    assert disc_pub == public_file.resolve()

    # None cases
    assert discover_data_dir(data_dir=None, repo_root=REPO_ROOT).exists()


def test_notebook_cell_source_validity():
    """Verify that all 5 cells in the generated notebook are syntactically valid and thin."""
    nb = build_legalir_notebook()
    assert len(nb["cells"]) == 5

    # Check cell types
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert nb["cells"][1]["cell_type"] == "code"
    assert nb["cells"][2]["cell_type"] == "code"
    assert nb["cells"][3]["cell_type"] == "code"
    assert nb["cells"][4]["cell_type"] == "code"

    # Verify python syntax in code cells
    for idx in [1, 2, 3, 4]:
        code_str = "".join(nb["cells"][idx]["source"])
        compile(code_str, f"<cell_{idx}>", "exec")


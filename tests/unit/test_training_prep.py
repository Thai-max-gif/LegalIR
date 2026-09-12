import pytest
import pandas as pd
from src.training.positive_localizer import PositiveLocalizer
from src.training.hard_negative_miner import HardNegativeMiner

def test_positive_localizer():
    macro_chunks = [
        {"chunk_id": "doc1_macro_01", "doc_id": "doc1", "text_norm": "quy định về thời hạn cấp phép lái xe là 10 ngày"},
        {"chunk_id": "doc1_macro_02", "doc_id": "doc1", "text_norm": "quy định về xử phạt vi phạm giao thông đường bộ"},
        {"chunk_id": "doc2_macro_01", "doc_id": "doc2", "text_norm": "thủ tục đăng ký kinh doanh doanh nghiệp"}
    ]
    localizer = PositiveLocalizer(macro_chunks)

    # Localize within gold doc1
    best_chunk = localizer.localize("Thời hạn cấp phép lái xe là bao lâu?", "doc1")
    assert best_chunk is not None
    assert best_chunk["chunk_id"] == "doc1_macro_01"

def test_hard_negative_miner():
    candidates = [
        {"doc_id": "gold1", "rrf_score": 0.05},
        {"doc_id": "neg1", "rrf_score": 0.04},
        {"doc_id": "neg2", "rrf_score": 0.03},
        {"doc_id": "neg3", "rrf_score": 0.02}
    ]
    gold_ids = ["gold1"]

    miner = HardNegativeMiner()
    mined_negs = miner.mine_negatives(candidates, gold_ids, max_negatives=2)
    assert "gold1" not in mined_negs
    assert len(mined_negs) == 2
    assert mined_negs[0] == "neg1"
    assert mined_negs[1] == "neg2"

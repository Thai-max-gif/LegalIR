from src.training.hard_negative_miner import HardNegativeMiner
from src.training.positive_localizer import PositiveLocalizer


def test_miner_excludes_duplicate_and_near_query_positives():
    blacklist = {"q1": {"2", "3"}}
    miner = HardNegativeMiner(false_negative_blacklist=blacklist)
    candidates = [
        {"doc_id": "1"},  # gold
        {"doc_id": "2"},  # in blacklist for q1
        {"doc_id": "3"},  # in blacklist for q1
        {"doc_id": "4"},  # valid negative
        {"doc_id": "5"},  # valid negative
    ]
    negs = miner.mine_negatives("q1", candidates, ["1"], max_negatives=10)
    assert "1" not in negs
    assert "2" not in negs
    assert "3" not in negs
    assert negs == ["4", "5"]


def test_positive_localizer_returns_top_chunks():
    chunks = [
        {"doc_id": "1", "chunk_id": "c1", "article": "Điều 1", "text_norm": "quy định về thủ tục cấp giấy phép đầu tư"},
        {"doc_id": "1", "chunk_id": "c2", "article": "Điều 2", "text_norm": "quy định về giải thể doanh nghiệp"},
    ]
    localizer = PositiveLocalizer(macro_chunks=chunks)
    pos = localizer.localize("thủ tục giấy phép đầu tư", "1", top_k=1)
    assert len(pos) == 1
    assert pos[0]["chunk_id"] == "c1"

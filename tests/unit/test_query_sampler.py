import pytest
from torch.utils.data import Dataset
from src.training.trainer import QueryBalancedSampler, QueryBalancedPairSampler


class DummyPairDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def test_query_balanced_sampler_interleaves_labels_per_query():
    # 4 queries, each having 1 positive (label=1.0) and 1 negative (label=0.0)
    records = [
        {"query_id": "q1", "label": 1.0},
        {"query_id": "q1", "label": 0.0},
        {"query_id": "q2", "label": 1.0},
        {"query_id": "q2", "label": 0.0},
        {"query_id": "q3", "label": 1.0},
        {"query_id": "q3", "label": 0.0},
        {"query_id": "q4", "label": 1.0},
        {"query_id": "q4", "label": 0.0},
    ]
    ds = DummyPairDataset(records)

    # Interleaved sampler (default)
    sampler_interleaved = QueryBalancedSampler(ds, seed=42, interleave=True)
    indices = list(sampler_interleaved)
    assert len(indices) == 8

    emitted_labels = [records[i]["label"] for i in indices]
    # Every adjacent pair should be 1 positive and 1 negative (50/50 balanced windows)
    for step in range(0, 8, 2):
        window = emitted_labels[step : step + 2]
        assert set(window) == {1.0, 0.0}

    # All 4 unique queries should be seen within the first 8 items
    seen_qids = {records[i]["query_id"] for i in indices[:8]}
    assert seen_qids == {"q1", "q2", "q3", "q4"}


def test_query_balanced_sampler_legacy_class_blocked_mode():
    records = [
        {"query_id": "q1", "label": 1.0},
        {"query_id": "q1", "label": 0.0},
        {"query_id": "q2", "label": 1.0},
        {"query_id": "q2", "label": 0.0},
        {"query_id": "q3", "label": 1.0},
        {"query_id": "q3", "label": 0.0},
        {"query_id": "q4", "label": 1.0},
        {"query_id": "q4", "label": 0.0},
    ]
    ds = DummyPairDataset(records)

    sampler_legacy = QueryBalancedSampler(ds, seed=42, interleave=False)
    indices = list(sampler_legacy)
    assert len(indices) == 8

    emitted_labels = [records[i]["label"] for i in indices]
    # In legacy mode, first 4 are all 1.0, next 4 are all 0.0
    assert emitted_labels == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_query_balanced_pair_sampler_alias():
    assert QueryBalancedPairSampler is QueryBalancedSampler

import json
from pathlib import Path

import pandas as pd

import src.evaluation.benchmark as benchmark
from src.core.paths import ProjectPaths
from src.evaluation.benchmark import aggregate_fold_metrics, run_benchmark, run_split_eval
from src.evaluation.evaluator import (
    _normalize_ids,
    compute_candidate_cutoffs,
    compute_candidate_recall,
    evaluate_predictions,
    normalize_candidate_cutoffs,
)


def test_candidate_cutoffs_returns_all_expected_keys():
    candidates = {"q1": ["1", "2", "3", "4", "5", "6"]}
    ground_truth = {"q1": ["1", "6"]}
    cutoffs = compute_candidate_cutoffs(candidates, ground_truth, cutoffs=[20, 50, 100, 150])
    assert set(cutoffs.keys()) == {
        "candidate_recall@20",
        "candidate_recall@50",
        "candidate_recall@100",
        "candidate_recall@150",
    }
    assert cutoffs["candidate_recall@20"] == 1.0


def test_benchmark_metrics_structure():
    pred = {"q1": ["1", "2"], "q2": ["3"]}
    truth = {"q1": ["1"], "q2": ["3"]}
    metrics = evaluate_predictions(pred, truth)
    assert "recall_at_5" in metrics
    assert "precision_at_5" in metrics
    assert "Recall@1" in metrics
    assert "Recall@3" in metrics
    assert "Recall@5" in metrics


def test_benchmark_dual_validation_metric_aliases():
    preds = {"q1": ["doc_1", "doc_2"]}
    golds = {"q1": ["doc_1"]}

    metrics = evaluate_predictions(preds, golds)

    assert metrics["recall@5"] == 1.0
    assert metrics["precision@5"] == 0.5
    assert metrics["Recall@1"] == 1.0
    assert metrics["Recall@3"] == 1.0
    assert metrics["Recall@5"] == 1.0
    assert metrics["Precision@5"] == 0.5


def test_default_candidate_cutoffs_cover_standard_diagnostics():
    candidates = {"q1": ["doc_1", "doc_2"]}
    ground_truth = {"q1": ["doc_1"]}

    metrics = compute_candidate_cutoffs(candidates, ground_truth)

    assert set(metrics) == {
        "candidate_recall@20",
        "candidate_recall@50",
        "candidate_recall@100",
        "candidate_recall@150",
        "candidate_recall@200",
    }


def test_candidate_recall_accepts_candidate_records():
    candidates = {
        "q1": [
            {"doc_id": "doc_x", "score": 1.0},
            {"doc_id": "doc_1", "score": 0.5},
        ]
    }
    ground_truth = {"q1": ["doc_1"]}

    metrics = compute_candidate_cutoffs(candidates, ground_truth, cutoffs=[1, 2])

    assert metrics == {
        "candidate_recall@1": 0.0,
        "candidate_recall@2": 1.0,
    }


def test_normalize_ids_handles_scalar_mapping_values():
    assert _normalize_ids({"answer": "doc_1"}) == ["doc_1"]
    assert _normalize_ids({"answer": b"doc_3"}) == ["doc_3"]
    assert _normalize_ids({"doc_id": 7}) == ["7"]
    assert _normalize_ids({"answer": {"doc_id": "doc_2"}}) == ["doc_2"]


def test_normalize_candidate_cutoffs_accepts_scalar_integer():
    assert normalize_candidate_cutoffs(50) == [50]
    assert normalize_candidate_cutoffs("20") == [20]


def test_candidate_recall_accepts_scalar_candidate_mapping():
    metrics = compute_candidate_cutoffs(
        {"q1": {"doc_id": "doc_1"}},
        {"q1": ["doc_1"]},
        cutoffs=[1],
    )

    assert metrics == {"candidate_recall@1": 1.0}


def test_candidate_recall_accepts_scalar_document_mapping():
    assert compute_candidate_recall(
        {"q1": {"document_id": "doc_1"}},
        {"q1": ["doc_1"]},
        k=1,
    ) == 1.0


def test_candidate_recall_accepts_scalar_candidate_string():
    assert compute_candidate_recall(
        {"q1": "doc_1"},
        {"q1": ["doc_1"]},
        k=1,
    ) == 1.0


def test_aggregate_fold_metrics_reports_all_final_metrics():
    fold_metrics = [
        {
            "recall_at_1": 0.5,
            "recall_at_3": 0.75,
            "recall_at_5": 1.0,
            "precision_at_5": 0.4,
            "candidate_recall@20": 0.8,
            "candidate_recall@50": 0.9,
            "candidate_recall@100": 1.0,
        },
        {
            "recall_at_1": 0.0,
            "recall_at_3": 0.5,
            "recall_at_5": 0.5,
            "precision_at_5": 0.2,
            "candidate_recall@20": 0.6,
            "candidate_recall@50": 0.8,
            "candidate_recall@100": 0.9,
        },
    ]

    summary = aggregate_fold_metrics(fold_metrics, [20, 50, 100])

    assert summary["mean_recall_at_1"] == 0.25
    assert summary["mean_recall_at_3"] == 0.625
    assert summary["mean_recall_at_5"] == 0.75
    assert summary["std_recall_at_5"] == 0.25
    assert summary["mean_precision_at_5"] == 0.30000000000000004
    assert summary["candidate_recalls"] == {
        "candidate_recall@20": 0.7,
        "candidate_recall@50": 0.8500000000000001,
        "candidate_recall@100": 0.95,
    }
    assert summary["final_ranking_metrics"] == {
        "Recall@1": 0.25,
        "Recall@3": 0.625,
        "Recall@5": 0.75,
        "Precision@5": 0.30000000000000004,
    }


def test_run_benchmark_reports_both_validation_protocols(tmp_path, monkeypatch):
    canonical_dir = tmp_path / "canonical"
    splits_dir = canonical_dir / "splits"
    splits_dir.mkdir(parents=True)

    query_ids = [f"q{i}" for i in range(1, 6)]
    pd.DataFrame({"doc_id": [f"doc_{qid}" for qid in query_ids]}).to_parquet(
        canonical_dir / "documents.parquet"
    )
    pd.DataFrame({"granularity": ["micro"]}).to_parquet(
        canonical_dir / "chunks.parquet"
    )
    pd.DataFrame(
        {
            "query_id": query_ids,
            "question_norm": [f"question-{qid}" for qid in query_ids],
        }
    ).to_parquet(canonical_dir / "queries_train.parquet")
    pd.DataFrame(
        {
            "query_id": query_ids,
            "doc_id": [f"doc_{qid}" for qid in query_ids],
        }
    ).to_parquet(canonical_dir / "qrels_train.parquet")

    random_folds = [
        {
            "train_query_ids": [qid for qid in query_ids if qid != val_qid],
            "val_query_ids": [val_qid],
        }
        for val_qid in query_ids
    ]
    (splits_dir / "random_5fold.json").write_text(
        json.dumps(random_folds), encoding="utf-8"
    )
    (splits_dir / "doc_disjoint_split.json").write_text(
        json.dumps(
            {
                "train_query_ids": query_ids[:3],
                "val_query_ids": query_ids[3:],
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    config = {
        "paths": {"canonical": "canonical"},
        "evaluation": {"candidate_cutoffs": [20, 50, 100]},
    }
    paths = ProjectPaths(
        repo=tmp_path,
        shared=tmp_path / "artifacts" / "shared",
        canonical=canonical_dir,
        local=tmp_path / "artifacts" / "local",
        local_models=tmp_path / "models",
        local_indexes=tmp_path / "indexes",
        local_runs=tmp_path / "runs",
    )

    class FakeBM25:
        def fit(self, records, show_progress=False):
            return self

        def save(self, path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"index")

    class FakeMemory:
        instances = []

        def __init__(self, rows, min_similarity=0.82):
            self.training_query_ids = frozenset(row["query_id"] for row in rows)
            FakeMemory.instances.append(self)

    class FakeHybrid:
        def __init__(self, question_memory=None, **kwargs):
            self.question_memory = question_memory

        def search_candidates(self, query, exclude_qid=None, top_k=100):
            qid = query.rsplit("-", 1)[-1]
            return [{"doc_id": f"doc_{qid}"}]

    class FakeFuser:
        def rank_candidates(self, candidates):
            return candidates

    class FakeSelector:
        def __init__(self, max_k=5):
            self.max_k = max_k

        def select(self, ranked):
            return [candidate["doc_id"] for candidate in ranked[: self.max_k]]

    monkeypatch.setattr(benchmark, "load_pipeline_config", lambda _: config)
    monkeypatch.setattr(
        benchmark.ProjectPaths,
        "from_repo",
        staticmethod(lambda repo_root=None: paths),
    )
    monkeypatch.setattr(
        benchmark,
        "validate_canonical_dataset",
        lambda _: {"is_valid": True},
    )
    monkeypatch.setattr(benchmark, "BM25MicroRetriever", FakeBM25)
    monkeypatch.setattr(benchmark, "QuestionMemory", FakeMemory)
    monkeypatch.setattr(benchmark, "HybridSearchEngine", FakeHybrid)
    monkeypatch.setattr(benchmark, "ReciprocalRankFusion", FakeFuser)
    monkeypatch.setattr(benchmark, "TopKSelector", FakeSelector)
    monkeypatch.setattr(benchmark, "ExactMatcher", lambda records: object())

    report = run_benchmark(config_path=config_path, label="test")

    candidate_keys = {
        "candidate_recall@20",
        "candidate_recall@50",
        "candidate_recall@100",
    }
    candidate_aliases = {"recall@20", "recall@50", "recall@100"}
    final_keys = {"Recall@1", "Recall@3", "Recall@5", "Precision@5"}
    for split_name in ("random_5fold", "document_disjoint"):
        split = report[split_name]
        assert candidate_keys <= split.keys()
        assert candidate_aliases <= split.keys()
        assert candidate_keys <= split["candidate_recalls"].keys()
        assert final_keys <= split.keys()
        assert set(split["final_ranking_metrics"]) == final_keys

    assert report["candidate_cutoffs"] == [20, 50, 100]
    assert len(report["random_5fold"]["folds"]) == 5
    expected_random_memory = [
        set(query_ids) - {val_qid} for val_qid in query_ids
    ]
    assert [
        set(memory.training_query_ids) for memory in FakeMemory.instances[:5]
    ] == expected_random_memory
    assert set(FakeMemory.instances[5].training_query_ids) == set(query_ids[:3])


def test_run_split_eval_reports_candidate_and_final_metrics():
    class StubEngine:
        def search_candidates(self, query, exclude_qid=None, top_k=100):
            return [
                {"doc_id": "doc_x", "score": 1.0},
                {"doc_id": "doc_1", "score": 0.5},
            ]

    class StubFuser:
        def rank_candidates(self, candidates):
            return candidates

    class StubSelector:
        def select(self, ranked):
            return [candidate["doc_id"] for candidate in ranked[:2]]

    metrics, predictions, candidates = run_split_eval(
        split_name="test",
        val_query_ids=["q1"],
        queries_dict={"q1": "question"},
        qrels_dict={"q1": ["doc_1"]},
        hybrid_engine=StubEngine(),
        fuser=StubFuser(),
        selector=StubSelector(),
    )

    assert set(key for key in metrics if key.startswith("candidate_recall@")) == {
        "candidate_recall@20",
        "candidate_recall@50",
        "candidate_recall@100",
        "candidate_recall@150",
        "candidate_recall@200",
    }
    assert metrics["Recall@1"] == 0.0
    assert metrics["Recall@3"] == 1.0
    assert metrics["Recall@5"] == 1.0
    assert metrics["Precision@5"] == 0.5
    assert predictions == {"q1": ["doc_x", "doc_1"]}
    assert candidates == {"q1": ["doc_x", "doc_1"]}


def test_isolated_memory_uses_train_memory_fit_without_validation_qrels(monkeypatch):
    calls = {}

    class FakeTrainMemory:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.training_query_ids = frozenset()

        def fit(self, train_queries, qrels):
            calls["fit"] = (train_queries, qrels)
            self.training_query_ids = frozenset(row["query_id"] for row in train_queries)
            return self

    monkeypatch.setattr(benchmark, "TrainQuestionMemory", FakeTrainMemory)

    memory = benchmark._build_isolated_memory(
        train_query_ids=["train"],
        validation_query_ids=["val"],
        queries_dict={"train": "train question", "val": "validation question"},
        qrels_dict={"train": ["doc-train"], "val": ["doc-val"]},
    )

    assert memory.training_query_ids == frozenset({"train"})
    fit_queries, fit_qrels = calls["fit"]
    assert [row["query_id"] for row in fit_queries] == ["train"]
    assert fit_qrels == {"train": ["doc-train"]}


def test_run_split_eval_applies_optional_reranker_before_fusion():
    class StubEngine:
        def search_candidates(self, query, exclude_qid=None, top_k=100):
            return [{"doc_id": "doc_x"}, {"doc_id": "doc_1"}]

    class StubReranker:
        def __init__(self):
            self.calls = []

        def rerank(self, **kwargs):
            self.calls.append(kwargs)
            return [{"doc_id": "doc_1", "reranker_score": 3.0}]

    class StubFuser:
        def rank_candidates(self, candidates):
            return candidates

    class StubSelector:
        def select(self, ranked):
            return [candidate["doc_id"] for candidate in ranked]

    reranker = StubReranker()
    _, predictions, candidates = run_split_eval(
        split_name="test",
        val_query_ids=["q1"],
        queries_dict={"q1": "question"},
        qrels_dict={"q1": ["doc_1"]},
        hybrid_engine=StubEngine(),
        fuser=StubFuser(),
        selector=StubSelector(),
        reranker=reranker,
        evidence_builder=object(),
        rerank_k=1,
        rerank_batch_size=2,
        rerank_max_length=128,
    )

    assert candidates == {"q1": ["doc_x", "doc_1"]}
    assert predictions == {"q1": ["doc_1"]}
    assert reranker.calls[0]["top_k"] == 1
    assert reranker.calls[0]["batch_size"] == 2
    assert reranker.calls[0]["max_length"] == 128

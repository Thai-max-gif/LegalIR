from src.retrieval.fusion import aggregate_chunks_to_documents, reciprocal_rank_fusion


def test_rrf_weighted_order():
    result = reciprocal_rank_fusion({"bm25": ["a", "b"], "dense": ["b", "a"]}, {"bm25": 1.0, "dense": 2.0}, 60)
    assert result[0][0] == "b"


def test_chunk_document_aggregation_deduplicates():
    result = aggregate_chunks_to_documents([("c1", 2.0), ("c2", 1.0), ("c3", 1.5)], {"c1": "d1", "c2": "d1", "c3": "d2"}, "max")
    assert result == [("d1", 2.0), ("d2", 1.5)]

# LegalIR System Architecture

This describes the pipeline, not release readiness. See [HISTORY_FIXES.md](HISTORY_FIXES.md) for the historical candidate review and [release workflow](REPRODUCIBLE_TRAINING_WORKFLOW.md) for qualification.

## Data and evaluation boundaries

The canonical dataset contains 8,532 legal documents, 934,416 micro chunks, 219,460 macro chunks, and 7,000 labeled training queries. The 1,000 public queries are **inference inputs**, not a locally labeled evaluation set.

Use only canonical Task 1 data. No external legal corpus, Task 2 data, crawling, synthetic LLM examples, or external inference APIs. Question memory, supervised mining, and fold adapters must use only the permitted training partition. Preserve five-fold OOF and document-disjoint evaluation. Train the final adapter separately on all training queries after held-out evaluation.

## Retrieval and ranking

```text
Query
  ├─ Legal BM25 on micro chunks
  ├─ PyVi compound-aware BM25
  ├─ DEk21 dense retrieval on macro chunks
  ├─ Exact legal matching
  └─ Fold-local question memory
        ↓
  Candidate fusion → query-aware macro evidence
        ↓
  BGE cross-encoder + fold/final LoRA adapter
        ↓
  Document aggregation and deterministic ranking
        ↓
  Validated top-document submission
```

Candidate and reranking depths vary by stage and runtime overrides. Do not substitute one universal cutoff for the effective configuration recorded by the run.

| Component | Source / responsibility |
|---|---|
| Legal BM25 | `src/retrieval/bm25_micro.py` |
| PyVi BM25 | `src/retrieval/bm25_pyvi.py`; bounded tokenization parallelism |
| Dense macro retrieval | `src/retrieval/dense_macro.py`; encoder and FAISS index |
| Hybrid retrieval | `src/retrieval/hybrid_search.py`; branch candidate combination |
| Pair mining | `src/training/build_pairs.py`; fold-local supervised pair assembly |
| Reranking | `src/ranking/reranker.py`; model loading, multi-query flatten/scatter, document scores |
| Rank fusion | `src/ranking/fusion.py`; ignores missing branch ranks represented by sentinel values >=900 |
| OOF orchestration | `src/pipeline/oof_runner.py`; random folds and document-disjoint evaluation |
| Production orchestration | `src/pipeline/kaggle_train.py`; FULL stage sequence and final delivery |

RRF combines retrieved branch ranks as `sum(weight / (k + rank))`. A missing branch must contribute zero. Cross-encoder scores are relevance scores; calibrated probabilities are not established by this architecture.

## Model and configuration contract

- Dense model: `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2`.
- Reranker: `BAAI/bge-reranker-v2-m3`; registry pin `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- Configuration sources: `configs/algorithm/legalir_v2.yaml`, `configs/experiments/reranker_lora.yaml`, and backend profiles under `configs/runtime/`.
- Current reranker experiment: LoRA rank 8, alpha 16, maximum length 384, batch size 8, BF16. Runtime overrides and coverage-enforced steps must be recorded rather than inferred from this document.
- Parameter audit totals: dense 134,998,272 + reranker 567,755,777 = **702,754,049**. This is the audited combined model total, not the number of trainable LoRA adapter parameters. Re-audit model changes against the 4B ceiling.
- The repair candidate records base-model revision and propagates it into adapter reload. Dense cache provenance and completed-stage identity still have gaps described in `fix.md`; a registry pin alone does not validate every reused artifact.

## Performance mechanisms and their limits

**Query-balanced training:** interleaves a positive and negative pair per eligible query. Expected-query audits are necessary because eligibility alone can omit queries. Coverage-enforced steps are a minimum exposure contract, not evidence of convergence or >96% Recall@5.

**Static mining cache:** shares label-independent branch candidates across folds. Fold-local supervised memory remains separate. It reduces repeated mining searches but does not remove every inference retrieval or index load.

**Batched reranking:** flattens pairs across queries, runs mixed-precision batches, and scatters document scores back deterministically. Throughput gains require actual A100 measurement; larger batches are not automatically safe.

**Evidence memory:** lazy Arrow-backed evidence and a bounded LRU limit one cache. They do not bound all host memory, indexes, worker processes, or GPU allocations.

**Completed-stage reuse:** completion markers and artifacts exist, but current identity/completeness validation is not sufficient for arbitrary output-directory reuse. Modal creates a fresh UUID each launch. Persisted files are not cross-attempt resume, and no exact optimizer/scheduler/RNG-state resume is promised.

## Quality measurement

Report official Recall@5 and Precision@5, candidate recall, and diagnostic ranking metrics on the correct held-out query population. For each query with gold set `G` and candidate set `C`:

- Corpus capacity at five: `min(5, |G|) / |G|`.
- Candidate top-five oracle: `min(5, |G ∩ C|) / |G|`.

A high ceiling is not a measured score. The historical 82.54% Recall@5 is one old fold; neither an improved full OOF score nor the >96% target is established.

## Storage and backend roles

- **Kaggle dataset:** canonical data and manifests, not Git-hosted heavy corpus files.
- **Git repository:** source, effective-configuration definitions, generated notebooks, tests, and release evidence.
- **Kaggle dual T4:** bounded real-model smoke only.
- **Modal / Colab A100:** separately approved real training and qualification.
- **Modal Volume / Colab recovery:** run artifacts with backend-specific durability limits.
- **Hugging Face target:** approved model delivery with visibility checks and immutable receipts; no publication implied by this document.

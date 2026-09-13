# LegalIR System Architecture & Technical Specification

## 1. System Overview
LegalIR is a high-recall, high-precision legal information retrieval system designed for the **UIT Data Science Challenge 2026 (Task 1: Legal Information Retrieval)**.

The corpus consists of **8,532 Vietnamese legal documents** and **1,153,876 text chunks** (934,416 micro chunks and 219,460 macro chunks). The system is trained on **7,000 legal queries** and evaluates on **1,000 public test queries**.

```
Query (Vietnamese Legal Question)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Multi-Branch Candidate Retrieval (Top 100 Candidates)    │
│   ├── Legal BM25 (Exact legal keyword match)                │
│   ├── PyVi BM25 (Vietnamese compound word tokenization)     │
│   └── DEk21 Dense (Dense neural semantic retrieval)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Reciprocal Rank Fusion & Dynamic Macro-Evidence Store    │
│   - Merges candidate ranks with robust RRF scoring          │
│   - Lazy-fetches article text without RAM exhaustion        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Deep Reranking (BAAI/bge-reranker-v2-m3 + LoRA PEFT)     │
│   - 568M parameter multilingual cross-encoder               │
│   - LoRA target modules: query, value, key (r=16, alpha=32) │
│   - Outputs calibrated relevance scores                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Prediction Filtering & Strict Invariant Validation       │
│   - Produces top 1 to 5 legal document IDs                  │
│   - Formats as official submission.zip                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Design

### 2.1 Multi-Branch Candidate Retrieval
1. **Legal BM25 (`src/retrieval/bm25_micro.py`)**:
   - Matches raw legal keywords, decree numbers, and article identifiers.
2. **PyVi Segmented BM25 (`src/retrieval/bm25_pyvi.py`)**:
   - Uses `pyvi.ViTokenizer` to group Vietnamese compound words (e.g. `thủ_tục`, `đăng_ký`, `doanh_nghiệp`), preventing spurious unigram splits.
3. **DEk21 Dense Macro Retriever (`src/retrieval/dense_macro.py`)**:
   - FAISS index using precomputed document embeddings to capture semantic similarity even when vocabulary differs.
4. **Reciprocal Rank Fusion (`src/retrieval/fusion.py`)**:
   - Merges ranks using formula:
     $$RRF(d) = \sum_{m \in M} \frac{w_m}{k + \text{rank}_m(d)}$$

### 2.2 Neural Cross-Encoder Reranker
- **Base Model**: `BAAI/bge-reranker-v2-m3` (568M parameters).
- **PEFT / LoRA Adapter**:
  - Rank: $r=16$
  - Scaling: $\alpha=32$
  - Dropout: $0.05$
  - Target Modules: `["query", "value", "key"]`
  - Learned parameters: $< 4,000,000,000$ (strictly compliant with UIT <4B rule).
- **Training Strategy**: Binary cross-entropy with logits over query-passage pairs.

### 2.3 Evidence Store & Memory Guard
- **MacroEvidenceStore**: Arrow-backed lazy reader with an LRU cache bounded at 512 MB to prevent Out-Of-Memory (OOM) on resource-constrained environments.

---

## 3. Storage & Role Boundaries

| Storage | Contents | Owner |
| :--- | :--- | :--- |
| **Kaggle Dataset (`phucdangg/legalir-task1-clean-data`)** | Canonical Parquet tables (`documents`, `chunks`, `queries_train`, `qrels_train`, `public-official.json`, `splits/`, `manifest.json`). Zero heavy data in Git. | Data Owner (A1) |
| **GitHub (`silent9669/LegalIR`)** | Source code (`src/`), notebooks (`notebooks/`), configs (`configs/`), test suites (`tests/`), CI/CD workflows. | Training Owner (B1) |
| **Kaggle Notebook (`notebooks/kaggle_t4x2_smoke.ipynb`)** | Fast 3-minute CUDA smoke gate (B1.1) on Tesla T4 / 2×T4 GPU. Outputs `kaggle_t4x2_report.json`. | Training Owner (B1) |
| **Google Colab Notebook (`notebooks/colab_a100_train.ipynb`)** | Full production training (B1.2) on NVIDIA A100 with BF16 precision. Exports model to Hugging Face. | Training Owner (B1) |

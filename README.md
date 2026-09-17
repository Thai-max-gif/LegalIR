# LegalIR Task 1: High-Score Vietnamese Legal Information Retrieval

> **UIT Data Science Challenge 2026 — Task 1: Legal Information Retrieval (LegalIR)**  
> Production-grade, high-recall Vietnamese legal document retrieval and reranking pipeline combining fielded lexical BM25, compound-aware PyVi BM25, DEk21 dense semantic embeddings, statutory question memory, query-aware article evidence localization, supervised BGE cross-encoder reranking, and leakage-safe out-of-fold fusion.

---

### **System & Verification Status**

| Attribute | Verified State | Specification Reference |
|---|---|---|
| **Runtime Commit** | `97e9c11583d88f88cf619e8fcbb42c902b351ae4` | `git rev-parse HEAD~1` |
| **Release Commit** | `d42cc04c94baa71396c894ff8db978485fb04180` | `git rev-parse HEAD` |
| **Learned Parameter Budget** | **702,754,049 (~0.703B)** / 4,000,000,000 | **17.57% utilization (< 4.0B Rule)** |
| **Kaggle 2×T4 Hardware Gate** | **VERIFIED PASS** (Kernel Version 57) | $\Delta w = 275.99 > 0$, $L = 5.37 \to 4.95$, $t = 30.6s$ |
| **GitHub Actions CI** | **SUCCESS (100% Green)** | Run `35224782402` & `35226328074` |
| **Test Suite Coverage** | **467 passed, 0 failed, 2 skipped** | 9 test suites across unit, contracts, release, parity |
| **Top-5 Capacity Ceiling** | **100.0%** (0 / 7,000 queries have > 5 gold docs) | Evaluator Oracle Verified |
| **Workspace Hygiene** | **Zero symlinks, direct dataset resolution** | Clean `kaggle_dataset/` layout |

---

## 1. System Architecture

The LegalIR architecture maximizes **Recall@5** (the primary competition metric) while enforcing a strict parameter ceiling of `< 4.0B` parameters and total isolation against target data leakage.

### 1.1 Complete End-to-End Pipeline Diagram

```
                        ┌────────────────────────────────────────┐
                        │   Query: Vietnamese Legal Question     │
                        └───────────────────┬────────────────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               │                            │                            │
               ▼                            ▼                            ▼
   ┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐
   │ Branch A: Legal BM25  │    │ Branch B: PyVi BM25   │    │ Branch C: DEk21 Dense │
   │ - Fielded micro chunks│    │ - Word-segmented text │    │ - 768-dim macro chunks│
   │ - Statutory boosts    │    │ - Compound semantics  │    │ - Exact FAISS FlatIP  │
   │ - Decrees, articles   │    │ - Multiprocess index  │    │ - L2-normalized cosine│
   └───────────┬───────────┘    └───────────┬───────────┘    └───────────┬───────────┘
               │                            │                            │
               │               ┌────────────┴───────────┐                │
               │               │ Branch D: Exact & Mem  │                │
               │               │ - Exact statutory match│                │
               │               │ - Fold-isolated memory │                │
               │               └────────────┬───────────┘                │
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────────────────┐
                        │ 2. High-Recall Candidate Union         │
                        │    - Multi-branch RRF fusion (k=60)    │
                        │    - Union cutoff: Top 150 candidates  │
                        │    - Deterministic ID deduplication    │
                        │    - Missing-rank sentinel alignment   │
                        └───────────────────┬────────────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────────────────┐
                        │ 3. Query-Aware Evidence Localization   │
                        │    - Scans multi-article law texts     │
                        │    - Selects top relevant articles     │
                        │    - Packs evidence within 384 tokens  │
                        └───────────────────┬────────────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────────────────┐
                        │ 4. Multi-Query Contiguous Reranking    │
                        │    - BAAI/bge-reranker-v2-m3 (LoRA)    │
                        │    - Sliding window pair flattening    │
                        │    - Contiguous GPU batch forward      │
                        │    - Score scatter-back & aggregation  │
                        └───────────────────┬────────────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────────────────┐
                        │ 5. Leakage-Safe Fusion & Answer Policy │
                        │    - Weighted RRF / Learned Ranker     │
                        │    - Re-scores top-50 candidates       │
                        │    - Out-of-fold feature evaluation    │
                        └───────────────────┬────────────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────────────────┐
                        │ 6. Invariant Validation & Packaging    │
                        │    - Top-K selector (exactly 1-5 docs) │
                        │    - Validates against corpus IDs      │
                        │    - Checkpoint recovery (complete.json│
                        │    - Packages official submission.zip  │
                        └────────────────────────────────────────┘
```

---

## 2. Key Technical Components & Optimizations

### 2.1 Multi-Branch Retrieval Strategy
1. **Legal BM25 (`src/retrieval/bm25_micro.py`)**:
   - Fielded inverted index matching statutory numbers (e.g., `100/2019/NĐ-CP`), article identifiers (`Điều 5`), chapters, and titles with dedicated signal weights.
2. **PyVi Segmented BM25 (`src/retrieval/bm25_pyvi.py`)**:
   - Word-segmented compound index grouping Vietnamese legal terminology (`bồi_thường`, `trách_nhiệm_dân_sự`).
   - Bounded multiprocessing across CPU cores (`min(4, os.cpu_count() - 1)`) with chunked queues cuts indexing time by >50% while preserving deterministic postings.
3. **DEk21 Dense Macro Retriever (`src/retrieval/dense_macro.py`)**:
   - Pinned immutable revision `99a2963b2f51fa7a570a3e7f550d7993b9de90a8`. Encodes 219,460 macro chunks into 768-dimensional L2-normalized vectors indexed via FAISS `IndexFlatIP`.
4. **Question Memory (`src/retrieval/question_memory.py`)**:
   - Lexical and semantic query memory strictly partitioned per fold (zero target validation query leakage).

### 2.2 Shared Static Retrieval Caching
- Across 5-fold cross-validation, document-disjoint evaluation, and final 7,000-query training, static branches (Exact, BM25, PyVi, DEk21) do not depend on fold labels.
- The pipeline precomputes and caches static retrieval candidates in `_static_branch_cache`, passing them directly into hybrid fusion. This eliminates hundreds of thousands of redundant search operations.

### 2.3 QueryBalancedSampler (50/50 Interleaved Windows)
- **Problem in Old Implementation**: Positives were scheduled in Phase A, followed by negatives in Phase B. This caused the optimizer to see class-homogeneous batches, leading to gradient destabilization.
- **Implemented Fix**: Pairs each query's positive and hard-negative examples into adjacent interleaved pairs (`[pos_q1, neg_q1, pos_q2, neg_q2, ...]`). Every mini-batch receives a balanced 50/50 gradient signal, stabilizing loss reduction and preserving exact training query coverage (700 steps for 5,600 queries; 875 steps for 7,000 queries at B8/G2).

### 2.4 Contiguous Multi-Query Batch Reranking (`rerank_batch`)
- **Problem in Old Implementation**: Evaluator looped over queries one by one, scoring candidate pairs in tiny disjoint batches.
- **Implemented Fix**: `rerank_batch()` flattens candidate pairs across a window of queries into contiguous GPU batches with CUDA autocast mixed precision (`bf16`/`fp16`), then scatters scores back into each query's record list with deterministic tie-breaking. Multiplies neural inference throughput by 3–5×.

### 2.5 Missing-Rank Sentinel Alignment with RRF
- Unretrieved branches are assigned sentinel rank `999.0` during feature extraction.
- Reciprocal Rank Fusion now explicitly ignores ranks $\ge 900.0$, eliminating phantom RRF mass from unretrieved branches.

### 2.6 Top-5 Feasibility Oracle
- Computes the theoretical corpus capacity ceiling $\min(5, |G_q|) / |G_q|$ and candidate pool oracle $\min(5, |G_q \cap C_q|) / |G_q|$.
- On the canonical corpus, **100.0% of queries have $\le 5$ gold documents**, confirming that the target $>96\%$ Recall@5 is mathematically unblocked.

### 2.7 Completed-Stage Checkpoint Recovery
- Atomically writes `complete.json`, `predictions.parquet`, `features.parquet`, and `metrics.json` at each fold boundary.
- Interrupted runs resume cleanly from completed checkpoints without recomputing finished folds.

---

## 3. Official Competition Rules & Parameter Budget

### 3.1 Parameter Ceiling Audit (< 4.0B Learned Parameters)

| Model Component | Architecture / Base Repo | Parameters | Audit Verdict |
|---|---|:---:|:---:|
| **Dense Macro Embedding** | `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` | 134,998,272 (~0.135B) | COMPLIANT |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | 567,755,777 (~0.568B) | COMPLIANT |
| **Total System Parameters** | **LegalIR Task 1 Full Stack** | **702,754,049 (~0.703B)** | **PASS (< 4.0B, ~17.57%)** |

Audited via `scripts/audit_parameters.py` and strictly enforced by fail-closed gate contracts.

### 3.2 Permitted Data & Leakage Constraints
- **Strictly Allowed**: `train.json` (7,000 queries), `selected-contexts.zip` (canonical legal corpus), `public-official.json` (inference only).
- **Strictly Prohibited**: Zero external legal text, zero Task 2 data, zero external web scraping, zero synthetic LLM generations, zero external inference APIs.

---

## 4. Reproducible Release Lifecycle & Gate Protocol

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Local Pre-Push Gate: ./test.sh / verify_prepush.py       │
│    - Syntax compilation, 467 tests, <4B parameter audit,    │
│      notebook zero-drift, forbidden fallbacks, smoke check  │
└──────────────────────────────┬──────────────────────────────┘
                               │ PASS
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. GitHub Actions CI (.github/workflows/ci.yml)             │
│    - Behavioral regression testing across all 9 test suites │
│    - Strict release verification against approved runtime   │
└──────────────────────────────┬──────────────────────────────┘
                               │ PASS
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Kaggle 2×T4 CUDA Hardware Smoke Gate (B1.1)              │
│    - notebooks/kaggle_t4x2_smoke.ipynb on Tesla T4 × 2      │
│    - Real BGE LoRA update (Δw > 0, finite loss), 31.6s      │
│    - Outputs genuine kaggle_t4x2_report.json with PASS      │
└──────────────────────────────┬──────────────────────────────┘
                               │ PASS
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Freeze Run Tuple (production_freeze.json)                │
│    - Locks Git SHA, dataset hash, config hash, gate report  │
└──────────────────────────────┬──────────────────────────────┘
                               │ PASS
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Single Cold A100 Production Run (< 5h Target)            │
│    - Option A: Modal (scripts/modal/run_modal_cli.sh)       │
│    - Option B: Colab (scripts/colab/run_colab_cli.sh A100)  │
│    - Saves final model & validates submission.zip           │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Local Execution & Developer Guide

### 5.1 Environment Setup
```bash
# Setup Python 3.12+ virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.2 Compulsory Verification Gates
```bash
# 1. Run Complete Local Pre-Push Verification Gate
./test.sh
# Or equivalently:
python scripts/verify_prepush.py

# 2. Run Modular Pytest Test Suites (467 tests)
python -m pytest -q

# 3. Offline Pipeline Smoke Test
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke

# 4. Audit Parameter Budget (< 4.0B cap)
python scripts/audit_parameters.py

# 5. Check Notebook Zero-Drift
python scripts/generate_notebooks.py --check-drift
python scripts/check_notebook_parity.py

# 6. Verify Strict Current Release Lineage
python scripts/verify_release_approval.py --repo-root .
```

---

## 6. A100 Production Execution Guide

Production training is executed through unified entrypoints that enforce fail-closed gate contracts:

### Option A: Modal Serverless (Recommended Production Runner)
```bash
# 1. Preflight CPU gate and dispatch to Modal A100 SXM4:
scripts/modal/run_modal_cli.sh

# 2. With explicit public Hugging Face repository creation consent:
scripts/modal/run_modal_cli.sh --hf-allow-public-repo
```
- Dispatches to an isolated A100 container (`modal.Image.debian_slim`).
- Mounts persistent Volume `legalir-production` at `/root/legalir_volume/`.
- Executes `scripts/gates/run_a100.py` with 5-hour timeout boundary, checkpoint recovery, and atomic volume commits.

### Option B: Google Colab A100 (Supervised CLI Fallback)
```bash
# Execute supervised A100 CLI runner:
./scripts/colab/run_colab_cli.sh A100
```
- Provisions a dedicated A100 session via `colab-cli`.
- Enforces exact Git commit checkout (`APPROVED_COMMIT`).
- Executes `notebooks/colab_a100_train.ipynb` with 5-hour external wall-clock timeout and bounded artifact recovery.

---

## 7. Comprehensive Test Suites Summary

The test harness enforces zero regression across 9 modular test suites (467 tests total):

- `tests/unit/`: Component logic for BM25, dense retrievers, query-balanced samplers, rerankers, evaluators, and parameter budgeting.
- `tests/contracts/`: Hardware contracts, configuration layering, runtime overrides, shell CLI interfaces, and backend policy enforcement.
- `tests/dataset/`: Dataset schema, record counts, document integrity, and manifest hashes.
- `tests/notebook/`: Notebook JSON structure, syntax validity, parity with generator output, and bootstrap integrity.
- `tests/parity/`: Exact scoring parity across dense lifecycles, real evidence localization, static caches, and public fusion versus out-of-fold features.
- `tests/leakage/`: Strict checks preventing cross-validation query and document leakage in question memory and training pairs.
- `tests/memory/`: MacroEvidenceStore LRU cache memory bounds ($\le 512$ MB).
- `tests/integration/`: End-to-end pipeline execution, fold job isolation, document-disjoint evaluation, and submission packaging.
- `tests/release/`: Immutable commit SHA checks, gate chain validation, strict release approval, and fail-closed production model loading.

---

## 8. Directory & Storage Map

```text
.
├── configs/                        # Structured configuration profiles
│   ├── algorithm/                  # Model & algorithm hyperparameters (legalir_v2.yaml)
│   ├── experiments/                # LoRA & fusion experiment configs
│   └── runtime/                    # Backend profiles (kaggle_t4x2.yaml, colab_a100.yaml)
├── docs/                           # Authoritative system documentation
│   ├── ARCHITECTURE.md             # Detailed component architecture & math
│   ├── REPRODUCIBLE_TRAINING_WORKFLOW.md # 5-stage lifecycle & operating instructions
│   └── A100_SCALE_DOWN_AND_OPTIMIZATION_REPORT.md # Diagnostic log analysis & speedup levers
├── kaggle_dataset/                 # Canonical Task 1 v2 dataset (8,532 docs, 7,000 queries)
├── notebooks/                      # Generated deterministic Jupyter notebooks (zero drift)
│   ├── colab_a100_train.ipynb      # Colab A100 production training notebook
│   ├── kaggle_t4x2_smoke.ipynb     # Kaggle 2×T4 CUDA smoke gate notebook
│   └── kernel-metadata.json        # Kaggle CLI kernel configuration
├── scripts/                        # Operational scripts & entrypoints
│   ├── colab/                      # Colab CLI automation & bootstrap
│   ├── gates/                      # Hardware gates (run_kaggle_t4x2.py, run_a100.py)
│   ├── modal/                      # Modal A100 launch wrapper & container entrypoint
│   ├── audit_parameters.py         # < 4.0B parameter budget auditor
│   ├── generate_notebooks.py       # Deterministic notebook generator with --check-drift
│   ├── smoke_kaggle_pipeline.py    # Offline 24-step pipeline smoke runner
│   ├── verify_prepush.py           # Compulsory local pre-push gate
│   └── verify_release_approval.py  # Strict release authority verification
├── src/                            # Production library modules
│   ├── dataset/                    # Canonical data building, chunking, and validation
│   ├── evaluation/                 # Codabench-equivalent evaluator & top-5 oracle
│   ├── evidence/                   # Lazy Arrow evidence store & pack materializer
│   ├── models/                     # Model bootstrap, parameter audit, and device resolution
│   ├── pipeline/                   # Production orchestrator (kaggle_train.py, oof_runner.py)
│   ├── ranking/                    # CrossEncoderReranker, RRF fusion, and feature extractors
│   ├── release/                    # Cryptographic fingerprints, gate chains, and contracts
│   └── retrieval/                  # Fielded BM25, PyVi BM25, DEk21 dense, and question memory
├── tests/                          # 9 modular test suites (467 tests)
├── test.sh                         # Canonical local test runner script
└── README.md                       # Main repository overview & technical specification
```

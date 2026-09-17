# LegalIR Task 1: High-Score Vietnamese Legal Information Retrieval

> **UIT Data Science Challenge 2026 — Task 1: Legal Information Retrieval (LegalIR)**  
> High-Recall Vietnamese Legal Information Retrieval Pipeline with Dual Lexical Retrieval, Dense Macro Embeddings, Query-Aware Evidence Localization, Supervised LoRA Cross-Encoder Reranking, and Leakage-Safe OOF Fusion.

### Current System Status & Production Readiness (2026-09-17)
- **Verified Runtime Commit:** `0ca7c135bcefb58b7fcb4f18ed9035a4e06d9428`
- **Verified Release Commit:** `aadd3f242f6258b2bacba5feef4783ccf373d84f`
- **Kaggle 2×T4 Smoke Gate (B1.1):** **VERIFIED PASS** on live Kaggle GPU hardware (`phucdangg/legalir-training`, Version 55, $\Delta w = 279.72 > 0$, $t = 31.63s$).
- **GitHub Actions CI:** **VERIFIED PASS** (Run `35220762427`, `test` in 7m51s, `strict-release` in 2m2s).
- **Test Suite Status:** 467/467 tests passed, 0 failures, 0 regressions, offline-compatible.
- **Top-5 Feasibility Oracle:** 100.0% corpus capacity ceiling (no query has >5 gold documents).
- **Workspace State:** Fully clean workspace, no symlinks, all production dependencies and scripts in place for Modal and Google Colab execution.

---

## 1. System Architecture

The LegalIR Task 1 system is organized into a single canonical pipeline designed for maximum **Recall@5** (the primary competition metric) while remaining strictly within competition constraints and the `< 4.0B` parameter budget.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     1. Multi-Branch Retrieval                                     │
├──────────────────────────┬──────────────────────────┬───────────────────────┬─────────────────────┤
│      Branch A (BM25)     │   Branch B (PyVi BM25)   │ Branch C (DEk21 Dense)│Branch D (Exact Match│
│  Fielded micro-chunk     │ Vietnamese word-segmented│  768-dim macro chunk  │& Train Query Memory)│
│  BM25 with legal entity  │   BM25 index preserving  │  embeddings (Huydang- │Statutory number/art │
│    signal boosting       │    compound semantics    │        DEk21)         │  & fold-safe memory │
└─────────────┬────────────┴─────────────┬────────────┴───────────┬───────────┴──────────┬──────────┘
              │                          │                        │                      │
              └──────────────────────────┼────────────────────────┴──────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            2. High-Recall Candidate Union & Fusion                                │
│           Merge candidate pools (cutoff k=150-200), deduplicate IDs deterministically,           │
│                         and extract multi-branch rank & score features                            │
└────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           3. Query-Aware Evidence Localization                                    │
│    Scans multi-article law texts to extract and pack only the query-relevant statutory articles   │
│             into structured evidence representations within the tokenizer token budget            │
└────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       4. Supervised PEFT/LoRA Cross-Encoder Reranking                             │
│     BAAI/bge-reranker-v2-m3 fine-tuned with RankNet/BCE loss on multi-band hard negative pairs    │
└────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          5. Leakage-Safe OOF Fusion & Scoring                                     │
│        Learned ranker / Weighted Reciprocal Rank Fusion (RRF) with out-of-fold validation         │
└────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      6. Deterministic Top-5 Selection & Compliance Validation                     │
│    Selects exactly 1-5 (default 5) unique valid document IDs with deterministic fallback order;   │
│           runs strict submission invariant validator and packages root submission.zip             │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Official Competition Rules & Parameter Budget

### 2.1 Model Parameter Budget (< 4,000,000,000 parameters)
The total learned parameters of every model used in the final Task 1 system must be **strictly below 4.0B**. LoRA adapters, quantization, and pruning do **not** reduce base model parameter counts for competition compliance purposes.

| Model Component | Base Model Repository | Architecture Parameters | Audit Status |
| :--- | :--- | :---: | :---: |
| **Dense Macro Embedding** | `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` | 134,998,272 (~0.135B) | COMPLIANT |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | 567,755,777 (~0.568B) | COMPLIANT |
| **Total System Parameters** | **LegalIR 4-Branch Stack** | **702,754,049 (~0.703B)** | **PASS (< 4.0B)** |

*Budget utilization is **~17.57%** of the 4.0B cap, leaving ample headroom while guaranteeing 100% compliance.*

### 2.2 Data Restrictions
- **Allowed Data**: Task 1 `train.json` (7,000 training queries), Task 1 `selected-contexts.zip` (canonical legal corpus), and Task 1 `public-official.json` (test queries for inference only).
- **Strictly Prohibited**: Zero external legal corpus, zero Task 2 data, zero external web scraping/crawling, zero synthetic LLM data generation, zero external inference API calls.

### 2.3 Official Scoring Semantics (Codabench Equivalence)
- Primary metric: **Mean Recall@5** across all test queries.
- Secondary / Tie-break metric: **Precision@5**.
- Invariant rules: Empty answer yields `0.0`; answers with `len > 5` yield `0.0`; duplicate IDs reduce precision; non-corpus document IDs are strictly rejected.

---

## 3. Reproducible Training Workflow & Release Architecture

All code releases follow the authoritative workflow defined in [`docs/REPRODUCIBLE_TRAINING_WORKFLOW.md`](docs/REPRODUCIBLE_TRAINING_WORKFLOW.md):

```text
Data Owner Release (Kaggle Dataset) 
       ↓
Local Pre-Push Gate (python scripts/verify_prepush.py) 
       ↓
GitHub Actions CI (PASS) 
       ↓
Kaggle 2×T4 Smoke Gate (notebooks/kaggle_t4x2_smoke.ipynb -> PASS) 
       ↓
Freeze Run Tuple (Git SHA + Dataset Hash + Smoke Report) 
       ↓
Google Colab A100 Production Run (notebooks/colab_a100_train.ipynb) 
       ↓
Hugging Face Artifacts & Codabench Submission
```

1. **Step 1 — Local Pre-Push Gate (`scripts/verify_prepush.py`) & GitHub CI**:
   - Single command `python scripts/verify_prepush.py` runs Python syntax compilation, modular pytest suites (135+ tests), `<4B` parameter budget audit, notebook zero-drift check (`scripts/generate_notebooks.py --check-drift`), and offline pipeline smoke.
2. **Step 2 — Kaggle 2×T4 CUDA Smoke Gate (B1.1)**:
   - Executes `notebooks/kaggle_t4x2_smoke.ipynb` on free Kaggle GPU (T4 / 2×T4).
   - Mounts `/kaggle/input/datasets/phucdangg/legalir-task1-clean-data` directly without repository bloat.
   - Mines a 50-query leakage-safe subset on the fly, executes real BGE+LoRA fine-tuning ($\Delta w > 0$, finite loss), tests checkpoint reload, and outputs `kaggle_smoke_report.json` with PASS verdict in ~3 minutes.
3. **Step 3 — Google Colab A100 Production Training (B1.2)**:
   - Executes `notebooks/colab_a100_train.ipynb` on NVIDIA A100.
   - Enforces A100 GPU and verifies the approved Git SHA passed the Kaggle Smoke Gate.
   - Full training on all 7,000 queries using `torch.bfloat16`, multi-branch candidate fusion, top-5 submission validation, and Hugging Face artifact export.

### Score Promotion Protocol (`scripts/check_score_promotion.py`)
Score-affecting changes are gated on leakage-safe out-of-fold cross-validation evidence:
- **Recall@5-First**: Higher Recall@5 wins; Precision@5 breaks ties.
- **Guardrails**: Candidate Recall@50/150 must not regress by >0.5%; Document-disjoint Recall@5 must not regress by >2.0%.
- **Sequential Ablation Order**:
  1. Candidate retrieval branch & RRF weights
  2. Rerank depth (`rerank_k = 40 / 50 / 80`)
  3. Loss function (`bce` vs `pairwise_logistic`)
  4. Training step scaling above coverage minimum
  5. Candidate pool size (`candidate_k = 150 / 200` only after miss analysis)

---

## 4. Local Execution & Verification Guide

Execution and verification are streamlined through canonical entrypoint scripts:

```bash
# 0. Setup environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Run Compulsory Pre-Push Verification Gate (All Systems)
./test.sh
# Or equivalently:
python scripts/verify_prepush.py

# 2. Run Modular Test Suites via Pytest (467 tests)
pytest -q

# 3. Verify Offline Pipeline Smoke (end-to-end 24-step simulation)
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke

# 4. Audit Learned Parameter Budget (< 4.0B cap)
python scripts/audit_parameters.py

# 5. Verify Zero Drift in Generated Notebooks
python scripts/generate_notebooks.py --check-drift
python scripts/check_notebook_parity.py

# 6. Verify Release Approval & Gate Lineage
python scripts/verify_release_approval.py --repo-root .
```

---

## 5. A100 Production Execution & Cloud Backends

The production pipeline is designed to run in a single cold run on an NVIDIA A100 GPU (targeting <5 hours end-to-end) delivering both the trained model and verified submission.

### 5.1 Pre-A100 Hardware Gate: Kaggle 2×T4 Smoke Gate (B1.1)
- **Notebook**: `notebooks/kaggle_t4x2_smoke.ipynb`
- **Dataset**: `phucdangg/legalir-task1-clean-data` mounted at `/kaggle/input/datasets/phucdangg/legalir-task1-clean-data`
- **Execution**: Runs in ~30s on Kaggle 2×T4 GPU (`cuda:0` Dense + `cuda:1` Reranker).
- **Invariants Verified**: Non-zero weight delta ($\Delta w > 0$), finite loss, real model weights (no mock fallback), checkpoint reload, and outputs `kaggle_t4x2_report.json` with verdict `PASS`.
- **Status**: Live verified on Kaggle Kernel Version 55 with `verdict: PASS`.

### 5.2 Option A: Modal Serverless (Recommended Production Attempt)
```bash
scripts/modal/run_modal_cli.sh
scripts/modal/run_modal_cli.sh --hf-allow-public-repo
```
- Dispatches to an isolated A100 container (`modal.Image.debian_slim`).
- Mounts persistent storage volume `legalir-production` at `/root/legalir_volume/`.
- Executes the unified production runner (`scripts/gates/run_a100.py`) with fail-closed provenance validation, 5-hour timeout boundary, and atomic volume commits at fold boundaries.

### 5.3 Option B: Google Colab A100 (Supervised CLI Fallback)
```bash
./scripts/colab/run_colab_cli.sh A100
```
- Provisions a dedicated A100 session via `colab-cli`.
- Enforces exact Git commit checkout (`APPROVED_COMMIT`) matching the verified release.
- Executes `notebooks/colab_a100_train.ipynb` with 5-hour external wall-clock timeout and bounded artifact recovery.

---

## 6. Modular Test Suites & System Verification

The repository contains **467 tests** across 9 specialized test suites ensuring complete system correctness and zero data leakage:

```bash
pytest tests/ -v
```

### Test Suite Structure:
- `tests/unit/`: Unit tests for tokenizers, BM25, dense retrievers, query-balanced samplers, rerankers, evaluators, and parameter auditing.
- `tests/contracts/`: Device contracts, configuration layering, runtime overrides, shell CLI interfaces, and backend policy enforcement (Kaggle restricted to smoke; FULL runs on Modal/Colab).
- `tests/dataset/`: Canonical dataset schema, record counts, document integrity, and manifest hashes.
- `tests/notebook/`: Notebook JSON structure, syntax validity, parity with generator output, and bootstrap integrity.
- `tests/parity/`: Exact scoring parity across dense lifecycles, real evidence localization, static caches, and public fusion versus out-of-fold features.
- `tests/leakage/`: Strict checks preventing cross-validation query and document leakage in question memory and training pairs.
- `tests/memory/`: MacroEvidenceStore LRU cache memory bounds ($\le 512$ MB).
- `tests/integration/`: End-to-end pipeline execution, fold job isolation, document-disjoint evaluation, and submission packaging.
- `tests/release/`: Immutable commit SHA checks, gate chain validation, strict release approval, and fail-closed production model loading.

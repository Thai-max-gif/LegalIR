# LegalIR Task 1: High-Score Vietnamese Legal Information Retrieval

> **UIT Data Science Challenge 2026 — Task 1: Legal Information Retrieval (LegalIR)**  
> High-Recall Vietnamese Legal Information Retrieval Pipeline with Dual Lexical Retrieval, Dense Macro Embeddings, Query-Aware Evidence Localization, Supervised LoRA Cross-Encoder Reranking, and Leakage-Safe OOF Fusion.

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

## 4. Local Execution Guide

All pipeline stages are modularized under `scripts/01_*.py` through `scripts/08_*.py`:

```bash
# 0. Setup environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Build Canonical Dataset from raw contexts and train.json
python scripts/01_build_dataset.py

# 2. Build BM25 (Raw & PyVi) Indexes and Precompute DEk21 Dense Embeddings
python scripts/02_build_indexes.py

# 3. Mine Hard Negatives and Build Supervised Reranker Training Pairs
python scripts/03_build_training_pairs.py

# 4. Supervised PEFT/LoRA Fine-Tuning of BGE Cross-Encoder Reranker
python scripts/04_train_reranker.py

# 5. Run 5-Fold Cross-Validation and Extract OOF Feature Matrices
python scripts/05_run_oof.py

# 6. Train Final Reranker & Fusion Models on All 7,000 Training Queries
python scripts/06_train_final.py

# 7. Run Public Test Inference and Produce submission.json
python scripts/07_predict_submission.py

# 8. Strict Submission, Invariant, and Parameter Budget Validation
python scripts/08_validate_submission.py

# 9. Verify All Mandatory Invariants via Pytest
pytest -q

# 10. Check Score Promotion Eligibility against Baseline
python scripts/check_score_promotion.py --candidate artifacts/task1/cv/cv_report.json
```

---

## 5. Kaggle Final Production Execution Guide

The Kaggle smoke gate uses the authoritative runner (`scripts/run_kaggle_smoke.py` -> `scripts/gates/run_kaggle_t4x2.py`) wrapped by the synchronized competition notebook (`notebooks/kaggle_t4x2_smoke.ipynb`). Colab gates use `notebooks/colab_t4_smoke.ipynb` (Single-T4) and `notebooks/colab_a100_train.ipynb` (A100 production). All notebooks are generated by `scripts/generate_notebooks.py` (zero-drift enforced).

### 5.1 Kaggle Kernel Setup
1. **Notebook**: Open `notebooks/kaggle_t4x2_smoke.ipynb`.
2. **Datasets**:
   - Attach official canonical dataset: **`phucdangg/legalir-task1-clean-data`**
   - Attach verified immutable production bundle: **`legalir-production-bundle`**
3. **Accelerator**: Select **GPU T4** or **GPU T4 x2**.
4. **Internet**: Toggle **On** (for downloading base reranker weights and minimal missing packages).
5. **Kaggle Secret (`HF_TOKEN`)**: Add `HF_TOKEN` under **Add-ons -> Secrets** for high-bandwidth Hugging Face downloads.

### 5.2 Notebook Execution Flow
Click **Run All** or **Save Version -> Save & Run All (Commit)**:

1. **Cell 0-1**: Environment & GPU preflight, PyTorch and CUDA verification.
2. **Cell 2**: Repository bootstrap, fail-closed commit SHA verification, and minimal dependency checks (zero PyTorch reinstallation).
3. **Cell 3**: Canonical runner execution via `scripts/run_kaggle_smoke.py`:
   - Verifies runtime Git commit SHA and canonical dataset identity.
   - Verifies production bundle fingerprints and semantic integrity.
   - Trains final BGE LoRA adapter on all 7,000 queries with effective batch 16.
   - Reranks public candidates using the frozen fusion winner (RRF or Learned Ranker).
   - Validates strict submission criteria and packages `submission.zip`.
4. **Cell 4**: Verification of final `submission.zip` artifact.

### 5.3 Exported Artifacts in `/kaggle/working/`
- `submission.zip`: Official competition submission archive containing **strictly `submission.json` at root**.
- `submission.json`: Exact 1,000 public test query predictions with 5 unique valid document IDs per query.
- `final_adapter/`: Saved LoRA adapter weights and `final_run_manifest.json` recording base model, optimizer steps, and parameter counts.
- `submission_manifest.json`: Verification manifest with SHA-256 hashes, query counts, and compliance checks.

---

## 5. Verification & Testing

The repository includes a dedicated test suite verifying all **24 mandatory requirements** from Section 20 of `LEGALIR_KAGGLE_HIGH_SCORE_AGENT.md`:

```bash
pytest tests/test_mandatory_24_invariants.py -v
```

### Verified Invariants Summary:
1. `test_invariant_01`: Legal identifier normalization is lossless (statutory numbers, articles, clauses, points).
2. `test_invariant_02`: Raw BM25 corpus and query tokenizer consistency.
3. `test_invariant_03`: PyVi BM25 corpus and query tokenizer consistency.
4. `test_invariant_04`: Legal entity boosts elevate exact statutory matches over generic text.
5. `test_invariant_05`: Exact statutory matcher handles NaN/null metadata without exceptions.
6. `test_invariant_06`: Candidate union deduplicates IDs deterministically across branches.
7. `test_invariant_07`: Train question memory excludes validation/self queries to prevent leakage.
8. `test_invariant_08`: Duplicate-group blacklist in negative miner prevents false negatives.
9. `test_invariant_09`: Query-aware evidence localization selects relevant article chunks.
10. `test_invariant_10`: Reranker evidence pack strictly respects token/character limits.
11. `test_invariant_11`: Supervised reranker training updates trainable weights ($\Delta w > 0$).
12. `test_invariant_12`: Trained reranker LoRA checkpoints reload cleanly with identical outputs.
13. `test_invariant_13`: 5-fold cross-validation partition has zero target leakage.
14. `test_invariant_14`: Learned fusion rankers train without validation fold labels.
15. `test_invariant_15`: Parameter auditor sums all learned models in the final pipeline.
16. `test_invariant_16`: Parameter auditor strictly rejects architectures $\ge 4.0\text{B}$ parameters.
17. `test_invariant_17`: Metric evaluator matches official Codabench scorer across all edge cases.
18. `test_invariant_18`: Submission query keys match public test queries exactly.
19. `test_invariant_19`: Submission enforces the 1-to-5 answers constraint for every query.
20. `test_invariant_20`: Submission enforces unique, valid document IDs from the official corpus.
21. `test_invariant_21`: Packaged `submission.zip` contains only `submission.json` at root.
22. `test_invariant_22`: Top-K selector uses deterministic fallback ranking order.
23. `test_invariant_23`: Training notebook parses as valid `nbformat` v4.
24. `test_invariant_24`: Pipeline configuration paths resolve safely in local and Kaggle environments.

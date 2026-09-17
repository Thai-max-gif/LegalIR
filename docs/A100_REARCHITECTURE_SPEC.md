# LegalIR A100 Re-Architecture Specification & Historical Reference

**Document Version:** 1.0 (Consolidated 2026-09-17)  
**Status:** Implemented & Verified in Production Pipeline  
**Runtime Git SHA:** `97e9c11583d88f88cf619e8fcbb42c902b351ae4`  
**Release Git SHA:** `d42cc04c94baa71396c894ff8db978485fb04180`  

---

## 1. Objectives & Non-Negotiable Boundaries

### 1.1 Dual Objective
1. **Quality Target:** Maximize **Mean Recall@5** (competition primary metric) targeting **>96%** with honest evaluation.
2. **Cold First-Run Performance Target:** Complete the full end-to-end first cold run in **< 5 hours** (270 minutes nominal working budget + 30 minutes contingency), delivering both the reloadable trained model and validated official `submission.zip`.

### 1.2 Non-Negotiable Invariants
- **Permitted Data:** Strictly limited to canonical Task 1 `train.json` (7,000 queries), `selected-contexts.zip` (canonical legal corpus of 8,532 documents), and `public-official.json` (test queries for inference only). No external legal corpora, Task 2 data, web crawling, or synthetic LLM generations.
- **Learned Parameter Ceiling:** Strictly `< 4.0B` learned parameters. The current 4-branch pipeline uses **702,754,049 (~0.703B)** parameters (~17.57% utilization).
- **Target Leakage:** 5-fold cross-validation and document-disjoint evaluations must enforce zero validation query or document exposure in question memory, negative mining, and training pairs.
- **Backend Policy:** Kaggle is strictly limited to bounded smoke contracts (`smoke`, `gpu_smoke`). FULL 7,000-query production training and submission generation run exclusively on Modal or Google Colab A100 environments.

---

## 2. Diagnosis of Historical 2026-09-16 Modal A100 Run

The raw execution log (`/tmp/modal-a100-launch.log`, 2,028 lines, SHA-256: `4ad86f1ba1...`) established:
- **Hardware:** NVIDIA A100-SXM4-40GB (39.5 GiB visible VRAM), PyTorch 2.5.1+cu124.
- **PyVi Indexing:** 934,416 micro chunks took **2,648.6 s (44.14 min)**.
- **Fold 0 Pair Pool:** 73,745 available pairs (6,102 positives + 67,643 hard negatives).
- **Fold 0 Training:** 700 steps (~15 min) using B8/G2 and BF16.
- **Fold 0 Evaluation:** 1,400 queries took **3,103.2 s (51.72 min)**.
  - *Correction:* The initial estimate of "2.2 queries/s" was a unit inversion. True rate was **0.451 queries/s** (~2.217 s/query).
- **Fold 0 Metrics:** Recall@5 = 82.54%, Precision@5 = 17.60%, nDCG@5 = 0.7066. Candidate Recall@50 = 98.05%, Candidate Recall@150 = 98.89%.
- **Interruption:** Local client terminal disconnected after ~1.7 hours; no checkpoint resume was available, losing all progress.

---

## 3. Prioritized Implementation Sequence & Delivered Levers

### Lever 1: QueryBalancedSampler Interleaved Windows (Q0 / M1)
- **Problem:** Positives were scheduled in Phase A, followed by negatives in Phase B, producing class-homogeneous optimizer steps.
- **Solution:** `QueryBalancedSampler` pairs positive and negative examples per query into adjacent 50/50 interleaved windows (`[pos_q1, neg_q1, pos_q2, neg_q2, ...]`).
- **Benefit:** Stabilizes gradient norm and loss convergence during fine-tuning while guaranteeing full query coverage (700 steps for 5,600 queries; 875 steps for 7,000 queries at B8/G2).

### Lever 2: Shared Static Branch Retrieval Caching (P2 / M2)
- **Problem:** All 5 CV folds and final training repeatedly called BM25, PyVi BM25, exact matching, and DEk21 dense macro searches for identical training queries.
- **Solution:** Decoupled static retrieval from fold-local question memory. Static candidates are computed once and stored in `_static_branch_cache`, then supplied directly into `hybrid_engine.search_candidates(..., branch_candidates=...)`.
- **Benefit:** Eliminates hundreds of thousands of redundant search operations across the pipeline.

### Lever 3: Multi-Query Contiguous Batch Reranker Inference (P3 / M3)
- **Problem:** Single-query serial loops underutilized GPU tensor cores during held-out evaluation and public inference.
- **Solution:** `rerank_batch()` flattens candidate pairs across multiple queries into contiguous GPU minibatches with CUDA autocast (`bf16`/`fp16`), scattering scores back into individual query candidate lists with exact document aggregation and deterministic tie-breaking.
- **Benefit:** Multiplies evaluation throughput by 3–5×, reducing evaluation time per fold from ~52 minutes to ~10–15 minutes.

### Lever 4: Bounded Multiprocessing for PyVi Indexing (P4 / M2)
- **Problem:** Single-threaded PyVi compound word tokenization took 44.14 minutes.
- **Solution:** Implemented chunked multiprocessing across `min(4, os.cpu_count() - 1)` worker processes with deterministic single-order postings assembly.
- **Benefit:** Reduces cold PyVi indexing time by >50% while preserving bit-identical postings and IDF values.

### Lever 5: Missing-Rank Sentinel Alignment (Q0 / M1)
- **Problem:** Unretrieved branch rank sentinel `999.0` injected artificial RRF mass.
- **Solution:** Reciprocal Rank Fusion now explicitly ignores ranks $\ge 900.0$, aligning feature tables with RRF scoring.

### Lever 6: Completed-Stage Checkpoint Recovery (P1 / M4)
- **Problem:** Interrupted jobs lost all work and required starting from scratch.
- **Solution:** Saves atomic completion markers (`complete.json`), predictions (`predictions.parquet`), candidate pools (`candidates.parquet`), and metrics (`metrics.json`) at each fold boundary.
- **Benefit:** Re-running an interrupted job automatically reuses completed folds, incurring zero recomputation.

### Lever 7: Top-5 Feasibility Oracle (Q1 / M5)
- **Implementation:** Evaluator calculates corpus capacity ceiling $\min(5, |G_q|) / |G_q|$ and candidate pool oracle $\min(5, |G_q \cap C_q|) / |G_q|$.
- **Verification:** On canonical data, 0 / 7,000 queries have $>5$ gold documents (100.0% ceiling).

---

## 4. Sub-5-Hour Cold First-Run Budget Allocation

The cold first-run working budget allocates **270 minutes** nominal execution plus **30 minutes** contingency:

| Budget Item | Target Time | Stages Included |
|---|:---:|---|
| **Cold Startup & Indexes** | 45 min | Package download, canonical load, parallel PyVi + BM25 indexing, DEk21 dense indexing, and static validation. |
| **Shared Static Branch Retrieval** | 25 min | Precomputing Exact, BM25, PyVi, and Dense candidates across all 7,000 queries. |
| **Supervised Assembly** | 20 min | Fold-specific memory fitting, negative mining, and evidence pack generation. |
| **All Training Jobs (7 total)** | 90 min | 5 fold adapters (3,500 updates) + 1 doc-disjoint adapter (700 updates) + 1 final adapter (875 updates). |
| **Held-Out Evaluation** | 55 min | 5-fold OOF evaluation + document-disjoint evaluation using `rerank_batch`. |
| **Fusion & Reporting** | 5 min | OOF feature aggregation and RRF / LightGBM selection. |
| **Final Inference & Packaging** | 30 min | Final model reload, public batch prediction, submission validation, and artifact persistence. |
| **Total Nominal Work** | **270 min** | **4 hours 30 minutes** |
| **Contingency Envelope** | **30 min** | Operational reserve for transient network or disk I/O |

---

## 5. Execution Protocol for Cloud Production Runs

### Modal Serverless Runner
```bash
scripts/modal/run_modal_cli.sh
# With explicit public Hugging Face repository opt-in:
scripts/modal/run_modal_cli.sh --hf-allow-public-repo
```
- Provisions container with persistent Volume `legalir-production` at `/root/legalir_volume/`.
- Executes `scripts/gates/run_a100.py` with 5-hour timeout boundary and atomic commits.

### Google Colab CLI Runner
```bash
./scripts/colab/run_colab_cli.sh A100
```
- Provisions dedicated A100 session via `colab-cli`.
- Enforces exact Git commit checkout matching the verified release.
- Executes `notebooks/colab_a100_train.ipynb` with external 5-hour wall clock and bounded recovery.

# LegalIR Task 1 Production Readiness Review

You are acting as an expert MLOps and Machine Learning Infrastructure Reviewer. We are preparing to run a large-scale, 5-Fold Cross-Validated Information Retrieval training pipeline on an Nvidia A100 (using either Google Colab or Modal). 

Please review the current repository state and pipeline architecture to confirm that we are 100% production-ready to trigger the training orchestrator without wasting compute credits or encountering silent failures.

## 1. The Dataset & Domain
- **Challenge:** UIT Data Science Challenge 2026 - Vietnamese Legal Information Retrieval (Task 1).
- **Corpus:** 8,532 Vietnamese legal documents chunked into hierarchical micro and macro levels (1,153,876 total chunks).
- **Queries:** 7,000 canonical training queries, 7,637 Qrels (positive labels).
- **Data Source:** Hosted on Kaggle (`phucdangg/legalir-task1-clean-data`).

## 2. The Retrieval & Ranking Algorithm
We are using a multi-stage retrieval architecture:
1. **Hybrid Union Retrieval (Stage 1):**
   - Combines 4 branches: Fielded Legal BM25, PyVi Segmented BM25, Exact Match, and Dense Macro Retrieval.
   - The Dense Macro branch uses `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` encoded via FAISS.
   - We extract `top_k_candidates=100` per query for downstream reranking.
2. **LoRA Cross-Encoder Reranking (Stage 2):**
   - Base model: `BAAI/bge-reranker-v2-m3` (560M parameters).
   - Fine-tuned via PEFT LoRA (r=8, alpha=16) on all 7,000 queries.
   - Trains using BCE loss, mixed precision (`bf16`), and gradient checkpointing.
   - Reranks the top 30 candidates from the union pool.
3. **Fusion & Selection (Stage 3):**
   - Candidate scores are fused using Reciprocal Rank Fusion (RRF, k=60) or cross-fitted LightGBM.
   - Final selector outputs the top 5 legal documents per query.
- **Constraints:** The competition enforces a strict `< 4.0B` learned parameter limit.

## 3. The Orchestration & Execution Logic
The pipeline is orchestrated via `src/pipeline/kaggle_train.py` which runs a 24-step pipeline:
- Validates the dataset fingerprints cryptographically before starting.
- Checks the `parameter_audit` budget (< 4.0B cap).
- Performs 5-Fold Out-of-Fold (OOF) cross-validation with fold-isolated hard-negative pairs and question memory.
- Uses `QueryBalancedSampler` with 50/50 interleaved positive/negative windows to stabilize gradients during LoRA fine-tuning.
- Caches static retrieval branches across all 5 folds and final training, eliminating duplicate BM25/PyVi/dense query searches.
- Uses contiguous multi-query batch reranking (`rerank_batch`) with deterministic score scatter-back for high GPU throughput.
- Evaluates the final pipeline against a strict document-disjoint split.
- Checkpoint recovery: atomically persists `complete.json`, predictions, and metrics at fold boundaries so an interrupted run resumes without recomputing completed folds.
- Packages final deliverables (adapter weights, training logs, ablation reports, and `submission.zip`).

## 4. The Cloud Delivery Mechanisms
We support two A100 execution backends:

### A. Google Colab (via Local CLI Automation)
- Launched via `scripts/colab/run_colab_cli.sh A100`.
- Executes a securely generated Jupyter Notebook (`notebooks/colab_a100_train.ipynb`).
- Pre-verifies `HF_TOKEN_WRITE` access and Kaggle datasets credentials *locally* before spinning up the Colab VM.
- Ensures the Colab VM checks out the *exact* 40-character Git SHA (no branch drift).
- Forces `pip` to respect Colab's native CUDA-compiled PyTorch installation via `legalir-torch-constraint.txt`.
- Pulls artifacts and a `recovery.tar.gz` back to the local machine even if the notebook execution crashes.

### B. Modal (Serverless Container)
- Launched via `modal run scripts/modal/run_modal_a100.py`.
- Deploys a `modal.Image.debian_slim` container natively loaded with PyTorch, CUDA, and our requirements.
- Secures `KAGGLE_USERNAME`, `KAGGLE_KEY`, and `HF_TOKEN` via Modal's built-in Secret vault (no `.env` uploads).
- Enforces a strict 5-hour function execution timeout (`timeout=18000`) to strictly protect the $30 credit limit.
- Runs the exact same core pipeline (`run_a100_production_gate`) as Colab.

## Your Task
Please review the above context, architecture, and deployment strategies, and answer the following questions:

1. **Algorithm & Constraints:** Are the optimized LoRA hyperparameters (r=8) and reduced token lengths (max_length=384) safe for a Vietnamese Legal Cross-Encoder without violating the competition parameter limit or degrading quality?
2. **Data Leakage & Robustness:** Does the 5-Fold OOF approach combined with a Document-Disjoint split provide sufficient protection against data leakage for this specific corpus size?
3. **Infrastructure Resilience:** Compare the Colab recovery strategy vs the Modal timeout strategy. Are there any hidden billing or execution risks with the A100 allocation?
4. **Final Deliverables:** Is the plan to automatically deploy the LoRA adapters and `submission.json` directly to Hugging Face Hub (upon success) robust enough given the fail-closed provenance gates?

*Provide a detailed, critical analysis. If everything is sound, confirm that the system is ready for the production A100 execution.*
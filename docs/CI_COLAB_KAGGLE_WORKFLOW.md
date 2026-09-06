# LegalIR Release & Verification Workflow: GitHub CI → Colab T4 → Kaggle T4×2

This document defines the authoritative release and verification workflow for the LegalIR Task 1 pipeline.

---

## 1. The Three-Gate Release Pipeline

To protect competition score, avoid expensive redundant compute, and eliminate OOF leakage, all code changes must pass through three strictly sequential gates:

```text
┌───────────────────────────────────────────────────────────────────────────┐
│ [1] Code Push to GitHub (main)                                            │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [Gate A] GitHub Actions: `LegalIR CI`                                     │
│   • CPU-only, source & syntax compileall                                 │
│   • Import verifications (pipeline, oof, pairs)                           │
│   • Full pytest invariant suite (375+ tests)                             │
│   • Parameter audit (<4B rule preflight)                                 │
│   • Tiny CPU smoke execution                                             │
│   • Byte-for-byte notebook parity check                                   │
│   • Strictly zero model weight downloads                                 │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼  (MUST BE GREEN)
┌───────────────────────────────────────────────────────────────────────────┐
│ [Gate B] Google Colab Single-T4 Contract Smoke                            │
│   • Verified Tesla T4 GPU running sequential stages on cuda:0             │
│   • Pins exact GREEN Git commit SHA from Gate A                           │
│   • Deterministic official canonical v2 data subset                       │
│   • Real DEk21 Dense encoding & FAISS index -> Model Unload & GPU purge   │
│   • Leakage-safe pair mining on subset train QIDs only                    │
│   • Real BGE+LoRA fine-tuning (optimizer steps, param_diff > 0, SHA)     │
│   • Prediction list contract validation on public queries                 │
│   • Exports `colab_smoke_report.json` with PASS verdict                   │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼  (MUST BE PASS FOR EXACT SHA)
┌───────────────────────────────────────────────────────────────────────────┐
│ [Gate C] Kaggle Final Production Run                                      │
│   • Executes thin final runner via `scripts/run_kaggle_final.py`          │
│   • Pins same approved Git runtime commit SHA                             │
│   • Verifies official canonical v2 dataset and immutable production bundle│
│   • Trains single real BAAI/bge-reranker-v2-m3 LoRA adapter on 7k queries │
│   • Reranks public candidates under frozen fusion winner                  │
│   • Generates root `submission.zip` containing `submission.json`          │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Invalidation & Gate Failure Rules

1. **Gate A Failure (CI RED)**:
   - Any failure in syntax, imports, tests, parameter audit, CPU smoke, or notebook parity immediately halts the release.
   - Fix code locally, commit, and push again.
2. **Gate B Failure (Colab FAIL)**:
   - If Colab smoke fails (e.g. non-finite loss, parameter diff = 0, VRAM OOM, prediction contract violation), inspect logs, fix the root cause locally, and restart from Gate A.
3. **Commit Invalidation Rule**:
   - A Colab PASS is valid **only for its exact commit SHA**.
   - Any source code commit pushed after Colab PASS immediately **invalidates** the previous smoke approval. Kaggle FULL must never run on an unverified SHA.

---

## 3. Google Colab Single-T4 Execution Guide

1. Open `colab/legalir_t4_smoke.ipynb` in Google Colab.
2. **Runtime Configuration**:
   - Navigate to **Runtime → Change runtime type**.
   - Select **T4 GPU** as the Hardware accelerator.
   - Confirm via `!nvidia-smi` that the assigned GPU is an NVIDIA Tesla T4.
3. **Secrets Setup**:
   - In Colab Secrets (key icon on sidebar), add:
     - `HF_TOKEN`: Hugging Face User Access Token (read).
     - `GITHUB_TOKEN`: (Optional) GitHub PAT for high-rate-limit API verification.
4. **Target SHA & Paths**:
   - Set `TARGET_SHA` to the exact 40-character commit SHA that concluded `success` in GitHub CI.
   - Set `DATA_DIR` to the mounted Google Drive path of canonical v2 data (e.g. `/content/drive/MyDrive/legalir-task1-clean-data`).
   - Set `OUTPUT_DIR` to the desired output folder.
5. Click **Runtime → Run all**.
6. Inspect `colab_smoke_report.json` and verify `result == "PASS"`.

---

## 4. Score-Promotion Guardrails & Ablation Protocol

To prevent accidental regressions and ensure systematic optimization, any score-affecting change must be evaluated against the accepted baseline using `scripts/check_score_promotion.py`.

### 4.1 Decision Hierarchy
1. **Primary Metric**: **Mean Recall@5** on 5-fold OOF CV.
   - Higher Recall@5 wins unconditionally.
2. **Tie-Break Metric**: **Precision@5**.
   - Used only when Recall@5 is identical within numerical precision.
3. **Candidate Recall Guardrail**:
   - Candidate Recall@50 and Candidate Recall@150 must not regress by more than `0.005` (0.5%).
4. **Document-Disjoint Robustness Guardrail**:
   - Document-disjoint Recall@5 must be evaluated and must not regress by more than `0.02` (2.0%).
5. **Parameter Budget**:
   - Total learned parameters must remain strictly `< 4,000,000,000` (4.0B).

### 4.2 Sequential Ablation Protocol
Do **not** perform broad combinatorial grid searches. Evaluate one component at a time in this strict order:
1. **Candidate Retrieval Branch & RRF Weights**: Tune BM25 / PyVi / Dense / Exact / Memory weights on candidate recall.
2. **Rerank Depth (`rerank_k`)**: Test `40` vs `50` vs `80` candidates reranked.
3. **Loss Function**: Compare `bce` vs `pairwise_logistic` vs `pairwise_margin`.
4. **Training Steps**: Evaluate step count scaling above the full-query coverage minimum.
5. **Candidate Pool Size (`candidate_k`)**: Test `150` vs `200` only if candidate miss analysis indicates headroom.

---

## 5. Verification Commands Reference

```bash
# Run local CI checks
python -m compileall -q src scripts
pytest -q
python scripts/audit_parameters.py
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
python scripts/check_notebook_parity.py

# Verify GitHub CI for target commit
python scripts/verify_github_ci.py --repo silent9669/LegalIR --sha <40-char-SHA>

# Build Colab deterministic subset
python scripts/build_colab_smoke_subset.py --data-dir artifacts/task1/data --out-dir artifacts/task1/smoke_subset

# Run Colab T4 smoke locally or on GPU
python scripts/run_colab_t4_smoke.py --data-dir artifacts/task1/data --work-dir artifacts/task1/colab_smoke --allow-non-t4

# Check score promotion eligibility
python scripts/check_score_promotion.py --candidate artifacts/task1/cv/cv_report.json
```

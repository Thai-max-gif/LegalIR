# Reproducible Training Workflow (Step-by-Step Guide)

## 1. Workflow Lifecycle

```
Data Owner (Kaggle Dataset)
              │
              ▼
Local Pre-Push Gate (python scripts/verify_prepush.py)
              │
              ▼
GitHub Actions CI (PASS: behavioral tests only, not production authorization)
              │
              ▼
Kaggle 2×T4 Smoke Gate (notebooks/kaggle_t4x2_smoke.ipynb -> PASS, sole pre-A100 gate)
              │
              ▼
Freeze Run Tuple (Git SHA + Dataset Hash + Smoke Report + profile digest)
              │
              ▼
A100 Production Run (Modal first via scripts/modal/run_modal_cli.sh,
                     or Colab fallback via scripts/colab/run_colab_cli.sh A100)
              │
              ▼
Hugging Face Release & Codabench Submission
```

Strict release validation (`python scripts/verify_release_approval.py --repo-root .`)
runs separately on the exact evidence-bearing release. A new runtime without
fresh evidence is expected to pass CI while strict rejects its stale freeze.

---

## 2. Stage-by-Stage Operating Instructions

### Stage 1: Local Pre-Push Verification
Before pushing any code or notebook updates to GitHub, run the local gate:
```bash
python scripts/verify_prepush.py
```
This executes:
1. Python syntax compilation across `src/` and `scripts/`.
2. Modular pytest suites (`tests/unit`, `tests/contracts`, `tests/dataset`, `tests/notebook`, `tests/parity`, `tests/leakage`, `tests/memory`, `tests/integration`, `tests/release`).
3. Parameter budget audit (`scripts/audit_parameters.py` < 4B).
4. Notebook zero-drift check (`scripts/generate_notebooks.py --check-drift`) and parity (`scripts/check_notebook_parity.py`).
5. Forbidden-fallback scan (`scripts/check_no_fallbacks.py`).
6. Offline Kaggle pipeline smoke.
7. Git working tree hygiene.

---

### Stage 2: Kaggle 2×T4 Smoke Gate (B1.1, sole pre-A100 gate)
1. **Open Notebook on Kaggle**:
   - URL: `https://www.kaggle.com/code/phucdangg/legalir-training` (or upload `notebooks/kaggle_t4x2_smoke.ipynb`).
2. **Attach Dataset**:
   - Kaggle Dataset: `phucdangg/legalir-task1-clean-data` (attached at `/kaggle/input/datasets/phucdangg/legalir-task1-clean-data` or `/kaggle/input/legalir-task1-clean-data`).
3. **Accelerator**:
   - Set Accelerator to **GPU T4 × 2**.
4. **Click "Run All"**:
   - Execution time: ~3 minutes.
   - Mines a 50-query leakage-safe subset on the fly.
   - Runs 3 optimizer updates on `BAAI/bge-reranker-v2-m3` + LoRA.
   - Asserts finite loss, weight update delta $\Delta w > 0$, and adapter checkpoint save/reload.
   - Generates `kaggle_t4x2_report.json` with verdict `"PASS"`.

### Stage 2b: Colab Single-T4 Contract Gate (B1.15) — RETIRED
Retired. Kaggle T4x2 (Stage 2) is the sole pre-A100 hardware gate. The
`colab_t4_smoke.ipynb` notebook is no longer generated and the A100 chain does
not consume `colab_t4_report.json`. (The `run_colab_t4.py` gate script remains
for manual use only.)

---

### Stage 3: A100 Production Training (B1.2)

Repair both backends locally; qualify one candidate; use Modal first unless a
human chooses otherwise. Do not launch both concurrently.

#### Option A: Modal (Recommended First Attempt)
```bash
scripts/modal/run_modal_cli.sh
scripts/modal/run_modal_cli.sh --hf-allow-public-repo
```
- CPU provenance gate runs before image build/dispatch; remote repeats
  checkout/provenance → HF access → dataset → train.
- Durable attempt path: `/root/legalir_volume/<sha>/attempts/<id>/` with
  explicit Volume commits. No resume; final bytes may be lost on hard kill.
- 5h function timeout caps duration, not spend. Consent defaults private-only.
- Record app ID, attempt path, image, SHA, start UTC, ceiling. Enforce kill
  criteria (OOM/nonfinite; >20min without meaningful progress in chatty phases;
  5h limit; spend ceiling). Confirm app termination via `modal app stop <id> --yes`
  (verify CLI syntax first), not just client exit.

#### Option B: Colab CLI Fallback (Supervised)
```bash
./scripts/colab/run_colab_cli.sh A100
```
This script automatically:
1. Validates mode/tools/deadlines (`COLAB_TIMEOUT` default 18000s = 5h
   whole-notebook wall clock, enforced externally; the same value is also
   passed per-cell to `colab exec --timeout`, which alone cannot bound a
   multi-cell notebook. Setup and bounded-cleanup time are separate. None of
   these is a billing cap).
2. Runs local provenance preflight before allocation.
3. Provisions a unique A100 session (`colab new -s legalir-a100-production-<rand> --gpu A100`).
4. Uploads allowlisted `.env`, required launch JSON, and gate/freeze overrides
   (required when present; failure never falls back to different evidence).
5. Executes `notebooks/colab_a100_train.ipynb` with the 5h wait.
6. Trains FULL: retrieval/indexing, five fold trainings/evaluations,
   doc-disjoint training/evaluation, fusion evaluation, final 7,000-query
   training (≈875 steps), inference, validation, HF artifact + receipt upload,
   recovery finalization. Five-hour completion is unmeasured, not assured.
7. Bounded recovery to `artifacts/task1/production/<session>/` (manifest/log +
   archive or both submissions required for exit 0) and bounded stop (30s).
   Exit precedence: primary wins; else stop failure=70; else incomplete=74.
8. Publishes to `https://huggingface.co/dangphuc2109/legalir-task1-reranker`
   only with explicit consent for public repos; auth always required.

Retired: `./scripts/run_colab_cli.sh T4` and any 4-hour/11.1h wording. The old
`COLAB_TIMEOUT=40000s` (11.1h) exceeded typical VM lifetime and is replaced by
the 5h default. No promise that FULL fits 5h.

#### Option C: Manual Web Interface (Colab, explicit release required)
1. **Find the evidence-bearing release commit**:
   - Use the latest `chore(release): evidence bundle for runtime …` commit whose
     `python scripts/verify_release_approval.py --repo-root .` passes strict
     verification (CPU-only). Call it `S`. The notebook baked into `S` pins its
     runtime `R` by design — embedding `S` there would be self-referential — so
     you must select `S` explicitly at launch; the default pin alone checks out
     stale evidence and fails strict bootstrap after GPU allocation and installs.
2. **Open Notebook on Google Colab**:
   - Open `notebooks/colab_a100_train.ipynb` from commit `S` (generated; do not hand-edit).
3. **Select Runtime**:
   - Runtime $\rightarrow$ Change runtime type $\rightarrow$ **NVIDIA A100 GPU** (High-RAM).
4. **Configure Secrets**:
   - In the Colab left sidebar 🔑 **Secrets**, add `HF_TOKEN_WRITE` (or `HF_TOKEN`)
     with write access to `HF_REPO_ID` (defaults to `dangphuc2109/legalir-task1-reranker`),
     plus Kaggle vars and optionally `HF_ALLOW_PUBLIC_REPO=1` for explicit public
     opt-in (Secrets win over uploaded `.env`). **Also add `LEGALIR_COMMIT_SHA`
     set to the full 40-character SHA of `S`.** Without it the notebook refuses
     to check out anything (fail fast, before installs) instead of silently
     using the stale runtime pin.
5. **Click "Run All"**:
   - Verifies A100 before training (runtime already allocated at this point) and
     confirms the Kaggle T4x2 gate passed for the selected release.
   - Executes FULL training as above, validates submission, publishes artifacts
     with immutable receipts, and finalizes recovery.

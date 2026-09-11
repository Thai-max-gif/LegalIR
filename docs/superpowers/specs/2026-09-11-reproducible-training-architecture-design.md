# LegalIR Task 1: Reproducible Training & Workspace Architecture Design

- **Date:** 2026-09-11
- **Author:** Training Owner / Claude Code
- **Status:** Approved by User / Ready for Implementation Planning
- **Spec Reference:** Notion DSC 2026 Reproducible Training Workflow (`https://dangphuc.notion.site/dscc`)

---

## 1. Executive Summary & Goals

### 1.1 Context
In the UIT Data Science Challenge 2026 (Task 1: Legal Information Retrieval), the team requires a training workflow that is fully reproducible, minimizes expensive GPU credit burn on Google Colab A100, and complies with BTC audit requirements. Every final model must be traceable from:
$$\text{Dataset Release} \longrightarrow \text{Code (Git SHA)} \longrightarrow \text{Config Fingerprint} \longrightarrow \text{Kaggle Smoke PASS} \longrightarrow \text{Colab A100 Full Train} \longrightarrow \text{Evidence / HF Artifact}$$

### 1.2 Core Objectives
1. **Bulletproof Reliability (Highest Percentage of Working Versions)**:
   - Zero crashes on paid Google Colab A100 runs.
   - All code, dataset paths, CUDA device visibility, and training logic are verified on free Kaggle GPU (T4 / 2×T4) beforehand via a 3-minute smoke test.
2. **Maximum Competitive Output (Highest Leaderboard Retrieval Score)**:
   - **Multi-Branch Retrieval**: High-recall candidate mining combining BM25 (exact keyword match), PyVi BM25 (Vietnamese compound word segmentation), and DEk21 Dense embeddings (semantic match).
   - **Dynamic Macro-Evidence Store**: Fast in-memory/Arrow-backed legal article lookups without RAM overflow.
   - **Deep Reranking**: Fine-tuned `BAAI/bge-reranker-v2-m3` using LoRA PEFT ($r=16, \alpha=32$), optimized on leakage-safe query-candidate pairs.
   - **Submission Invariants**: Strict validation enforcing $1 \le |\text{answer}| \le 5$, unique official document IDs, and exact format compliance.
3. **Workspace Reorganization & De-duplication**:
   - Eliminate scattered and duplicate folders (`kaggle_kernel/`, `kaggle_kernel_task1/`, `colab/`, root `.ipynb` files).
   - Establish a clean, canonical structure: `notebooks/`, `scripts/`, `src/`, `tests/`, `configs/`, `docs/`.
4. **Clean Testing & CI/CD**:
   - Create dedicated test suites for dataset verification (`tests/dataset/`) and notebook integrity (`tests/notebook/`).
   - Implement a fast local pre-push gate (`scripts/verify_prepush.py`).
   - Rewrite GitHub Actions (`.github/workflows/ci.yml`) to enforce fail-closed verification.

---

## 2. System Architecture & Dual-Environment Workflow

### 2.1 Storage & Role Boundaries

| Role | Storage / Channel | Artifacts & Responsibilities |
| :--- | :--- | :--- |
| **Data Owner (A1)** | Kaggle Dataset (`legalir-task1-clean-data`) | Prepares canonical v2 dataset (`documents.parquet`, `chunks.parquet`, `queries_train.parquet`, `qrels_train.parquet`, `public-official.json`, `splits/`, `manifest.json`). Published directly to Kaggle. Zero heavy parquet data committed to GitHub. |
| **Training Owner (B1)** | GitHub (`silent9669/LegalIR`) | Source code (`src/`), configuration profiles (`configs/`), thin launcher notebooks (`notebooks/`), automated test suite (`tests/`), CI/CD workflows. |
| **Smoke Gate (B1.1)** | Kaggle GPU (Tesla T4 / 2×T4) | Real CUDA verification on a 50-query deterministic subset. Proves dataset hash match, CUDA execution, finite loss, weight update delta $\Delta w > 0$, adapter save/reload. Outputs `kaggle_smoke_report.json`. |
| **Production Run (B1.2)** | Google Colab NVIDIA A100 | Real full training on all 7,000 queries. Uses `torch.bfloat16`, larger micro-batch sizes, and multi-worker DataLoaders. Produces final LoRA adapter, `run_manifest.json`, metrics, and uploads to Hugging Face. |
| **Lead / Integration (C)** | Notion & HF Registry | Verifies frozen run tuple before approving submission. |

### 2.2 End-to-End Pipeline Data Flow

```
[Approved Kaggle Dataset vN] (phucdangg/legalir-task1-clean-data)
                 │
                 ▼
[GitHub Code & Notebooks] (silent9669/LegalIR @ Commit SHA)
                 │
                 ▼
   Local Pre-push Verification (scripts/verify_prepush.py)
   & GitHub Actions CI (.github/workflows/ci.yml)
                 │
                 ▼ (PASS)
   Kaggle 2×T4 Smoke Test Gate (notebooks/kaggle_smoke.ipynb)
   - Real CUDA initialization & T4 check
   - Dataset manifest & hash verification
   - Forward + Backward + Optimizer step
   - Finite loss & Weight delta Δw > 0
   - Adapter checkpoint save & reload
   - Outputs: kaggle_smoke_report.json
                 │
                 ▼ (PASS: Freeze Run Tuple)
   Google Colab A100 Full Training (notebooks/colab_a100_train.ipynb)
   - Verifies A100 GPU & frozen tuple
   - Full 7,000 queries BGE-reranker-v2-m3 + LoRA
   - High-throughput BF16 execution
   - Validation & Top-5 submission generation
   - Outputs: final_adapter/, run_manifest.json, logs, metrics
                 │
                 ▼
[Hugging Face Model Hub & Evidence Package]
```

---

## 3. Workspace Reorganization Specification

### 3.1 Target Directory Layout

```
LegalIR/
├── .github/
│   └── workflows/
│       └── ci.yml                         # Clean GitHub Actions workflow
├── configs/
│   ├── base.yaml                          # Base retrieval & reranker hyperparameters
│   ├── kaggle_smoke.yaml                  # Kaggle 2×T4 smoke profile (fast subset)
│   ├── colab_a100.yaml                    # Colab A100 production profile (full epochs, BF16)
│   └── parameter_audit.json               # Learned parameter audit (<4B budget)
├── notebooks/
│   ├── kaggle_smoke.ipynb                 # Dedicated Kaggle T4x2 Smoke Gate launcher
│   └── colab_a100_train.ipynb             # Dedicated Colab A100 Production Training launcher
├── kaggle/
│   └── kernel-metadata.json               # Kaggle CLI metadata for notebook push
├── scripts/
│   ├── 01_build_dataset.py                # Dataset builder for Data Owner
│   ├── generate_notebooks.py              # Deterministic generator for both notebooks
│   ├── run_kaggle_smoke.py                # CLI runner for Kaggle Smoke Gate (B1.1)
│   ├── run_colab_train.py                 # CLI runner for Colab A100 Production (B1.2)
│   ├── audit_parameters.py                # Parameter budget verifier (<4B)
│   ├── verify_dataset.py                  # Dataset schema, hash & manifest verifier
│   └── verify_prepush.py                  # Single-command pre-push verification gate
├── src/                                   # Clean modular core packages
│   ├── core/                              # Config, memory guard, hashing
│   ├── data/                              # Parquet loaders, normalization, splits
│   ├── retrieval/                         # BM25, PyVi BM25, DEk21 dense, exact match
│   ├── reranker/                          # BGE-reranker-v2-m3, LoRA module
│   ├── pipeline/                          # Smoke runner, production runner, fusion
│   └── evaluation/                        # Codabench metrics, submission validator
├── tests/
│   ├── unit/                              # Component unit tests
│   ├── dataset/                           # Dedicated canonical dataset validation tests
│   ├── notebook/                          # Dedicated notebook structure & cell syntax tests
│   ├── integration/                       # Pipeline integration tests
│   ├── parity/                            # Retrieval & ranking parity tests
│   └── release/                           # Parameter audit & release gate invariant tests
├── docs/
│   ├── ARCHITECTURE.md                    # System architecture & role boundaries
│   └── REPRODUCIBLE_TRAINING_WORKFLOW.md  # Step-by-step workflow guide
└── requirements/
    ├── base.txt                           # Core dependencies
    ├── kaggle.txt                         # Kaggle environment overrides
    └── gpu.txt                            # CUDA & training dependencies
```

### 3.2 Files to Remove / Consolidate
1. **Remove directory `kaggle_kernel/`**: Contains legacy `legalqa_gpu_pipeline.ipynb` (Task 2) and old copies.
2. **Remove directory `kaggle_kernel_task1/`**: Redundant copy of notebooks.
3. **Remove directory `colab/`**: Redundant old Colab smoke copy.
4. **Remove root `legalir_training.ipynb`**: Clean root directory.
5. **Consolidate flat tests in `tests/`**: Move or categorize root test files into `tests/dataset/`, `tests/notebook/`, `tests/unit/`, or `tests/release/`.

---

## 4. Component Details

### 4.1 Dataset Discovery Protocol (`src/data/canonical.py`)
The dataset resolver checks the following candidate paths in order:
1. `/kaggle/input/legalir-task1-clean-data` (Kaggle attached dataset)
2. `/kaggle/input/task1-canonical-v2` (Legacy Kaggle mount)
3. `/content/drive/MyDrive/legalir-task1-clean-data` (Google Colab Drive mount)
4. `/content/data/task1_canonical_v2` (Colab local mount)
5. `data/task1_canonical_v2` or `artifacts/task1/data` (Local development)

**Integrity Verification:**
- Resolves `manifest.json` and validates SHA-256 digests of:
  - `documents.parquet` (8,532 rows)
  - `queries_train.parquet` (7,000 rows)
  - `qrels_train.parquet` (7,637 rows)
  - `public-official.json` (1,000 queries)
  - `duplicate_groups.json` (4 groups)
  - `splits/random_5fold.json` & `splits/doc_disjoint_split.json`

### 4.2 Notebook Generator (`scripts/generate_notebooks.py`)
Generates both `notebooks/kaggle_smoke.ipynb` and `notebooks/colab_a100_train.ipynb` from structured Python templates to guarantee zero manual editing drift:
- Injects exact pinned Git SHA (defaults to latest HEAD or approved commit).
- Minimal dependency bootstrap without reinstalling PyTorch/CUDA.
- Proper Jupyter metadata (`kernelspec`: `python3`, `nbformat: 4`, `nbformat_minor: 5`).
- Supports `--check-drift` flag in CI to fail if committed notebooks differ from generated output.

### 4.3 Kaggle Smoke Runner (`scripts/run_kaggle_smoke.py`)
Implements Notion B1.1 specification:
1. Verifies CUDA availability (`torch.cuda.is_available()`), logs GPU model name (Tesla T4 / 2×T4).
2. Verifies canonical dataset integrity.
3. Initializes retrieval components and loads `BAAI/bge-reranker-v2-m3` with LoRA configuration.
4. Mines mini training pairs on a 50-query subset using fold-local question memory and 4 duplicate group blacklist.
5. Runs 3 optimizer steps with AdamW.
6. Asserts:
   - Loss is finite ($L < 100$, not NaN).
   - Parameter difference $\Delta w = \|w_{\text{final}} - w_{\text{initial}}\|_2 > 0$.
   - Saves adapter checkpoint to disk and reloads into a fresh model instance.
   - Evaluates a 5-query prediction probe.
7. Saves `kaggle_smoke_report.json` containing:
   - `verdict`: `"PASS"`
   - `git_sha`: 40-char commit SHA
   - `dataset_hash`: `manifest.json` SHA-256
   - `gpu_info`: device name, VRAM, count
   - `loss_initial`, `loss_final`, `weight_delta`
   - `peak_vram_gb`

### 4.4 Colab A100 Production Runner (`scripts/run_colab_train.py`)
Implements Notion B1.2 specification:
1. Enforces GPU is NVIDIA A100 (aborts immediately if running on T4/V100/CPU).
2. Verifies that `kaggle_smoke_report.json` exists for the exact Git SHA and verdict is `"PASS"`.
3. Loads `configs/colab_a100.yaml`:
   - Precision: `torch.bfloat16`
   - Effective batch size: 16 (per-device batch 8, gradient accumulation 2)
   - Max length: 512
   - Epochs: 3 (or configured full schedule)
4. Trains BGE LoRA on all 7,000 queries.
5. Generates public predictions against the 1,000 public test queries.
6. Enforces submission invariants ($1 \le |answer| \le 5$, official IDs only, no duplicates).
7. Emits `run_manifest.json` and packages Hugging Face upload bundle.

---

## 5. Dedicated Test Suite (`tests/`)

### 5.1 Dataset Validation Suite (`tests/dataset/`)
- `test_dataset_schema.py`: Verifies schemas and column types for all Parquet files.
- `test_dataset_integrity.py`: Verifies row counts, non-empty text, valid URLs, and legal numbers.
- `test_dataset_splits.py`: Verifies 5-fold coverage (all 7,000 queries appear in exactly 1 test fold) and document-disjoint boundaries.
- `test_dataset_manifest.py`: Verifies SHA-256 checksums in `manifest.json`.

### 5.2 Notebook Validation Suite (`tests/notebook/`)
- `test_notebook_structure.py`: Asserts valid JSON, valid notebook metadata, and kernelspec.
- `test_notebook_syntax.py`: Parses all python code cells with `ast.parse` to catch syntax errors before push.
- `test_notebook_parity.py`: Verifies generated notebooks match committed notebooks (`--check-drift`).

### 5.3 Release & Invariant Suite (`tests/release/`)
- `test_parameter_audit.py`: Learned parameter count strictly $< 4,000,000,000$.
- `test_leakage_guard.py`: Verifies negative pair mining respects fold boundaries and duplicate blacklists.

---

## 6. Pre-Push Verification & GitHub Actions CI

### 6.1 Local Pre-Push Tool (`scripts/verify_prepush.py`)
A single gate command executed before git push:
```bash
python scripts/verify_prepush.py
```
Steps executed:
1. `python -m compileall -q src scripts`
2. `pytest -q tests/unit tests/dataset tests/notebook tests/parity tests/release`
3. `python scripts/audit_parameters.py`
4. `python scripts/generate_notebooks.py --check-drift`
5. Checks git status to warn about uncommitted files or untracked large data files.

### 6.2 GitHub Actions CI (`.github/workflows/ci.yml`)
Runs on all pull requests and pushes to `main`:
```yaml
name: LegalIR CI

on:
  push:
    branches: ["main"]
  pull_request:
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: legalir-ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    env:
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
      TOKENIZERS_PARALLELISM: "false"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"
      - run: pip install -r requirements.txt
      - run: python -m compileall -q src scripts
      - run: pytest -q tests/unit tests/dataset tests/notebook tests/parity tests/release
      - run: python scripts/audit_parameters.py
      - run: python scripts/generate_notebooks.py --check-drift
      - run: python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
```

---

## 7. Documentation & Memory Plan
1. Update `docs/ARCHITECTURE.md` with system design and role boundaries.
2. Create `docs/REPRODUCIBLE_TRAINING_WORKFLOW.md` with step-by-step operating instructions.
3. Update Memory:
   - `legalir-architecture-and-dataset-guide.md`: Reflect Kaggle Smoke + Colab A100 flow and canonical paths.
   - `legalir-ci-colab-release-gate.md`: Update release gate definitions.

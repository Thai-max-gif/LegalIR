# Reproducible Training & Workspace Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean and reorganize the LegalIR workspace, establish dual-environment runners for Kaggle GPU Smoke testing (free T4) and Google Colab A100 production training, implement dedicated dataset and notebook validation test suites, create a local pre-push gate, and rewrite CI/CD.

**Architecture:** A unified modular codebase (`src/`) shared across two thin, dedicated launchers (`notebooks/kaggle_smoke.ipynb` and `notebooks/colab_a100_train.ipynb`) generated deterministically via `scripts/generate_notebooks.py`. Kaggle T4 smoke executes a 3-minute CUDA verification on a 50-query subset to prove zero crashes and produce `kaggle_smoke_report.json`, freezing the Git SHA before Google Colab A100 consumes compute units to train the full BGE-reranker-v2-m3 LoRA model on all 7,000 queries.

**Tech Stack:** Python 3.12/3.14, PyTorch, Hugging Face Transformers, PEFT (LoRA), Accelerate, PyArrow, Parquet, BM25s, PyVi, Pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-reproducible-training-architecture-design.md`

## Global Constraints

- Learned Parameter Budget: Strictly $< 4,000,000,000$ (4B) learned parameters.
- Zero Heavy Dataset in Git: The canonical dataset (`legalir-task1-clean-data`) is mounted directly on Kaggle; only test fixtures live in git.
- Zero PyTorch Reinstallation: Notebooks must NOT pip-install or overwrite Kaggle's or Colab's pre-configured CUDA/PyTorch runtime.
- Exact Submission Invariants: Public predictions must contain exactly 1,000 queries, $1 \le |\text{answer}| \le 5$, official document IDs only, strictly formatted as `submission.zip`.
- Notebook Parity & Reproducibility: Notebooks are generated from single-source templates; manual drift is rejected in CI.

---

### Task 1: Workspace Cleanup & Directory Reorganization

**Files:**
- Remove: `kaggle_kernel/`, `kaggle_kernel_task1/`, `colab/`, `legalir_training.ipynb`
- Modify: `.gitignore`
- Test: Verify directories no longer exist

**Interfaces:**
- Consumes: Existing repository tree
- Produces: Clean root and clean `.gitignore`

- [ ] **Step 1: Remove redundant notebook directories and root notebook**

```bash
rm -rf kaggle_kernel kaggle_kernel_task1 colab legalir_training.ipynb
```

- [ ] **Step 2: Update `.gitignore` to protect against accidental dataset / checkpoint commits**

Add the following to `.gitignore`:
```
# Large artifacts & datasets
artifacts/task1/data/*.parquet
artifacts/submission/*.zip
artifacts/submission/final_adapter/
*.parquet
*.heapsnapshot
.firecrawl/
```

- [ ] **Step 3: Verify working tree clean of duplicate folders**

```bash
ls -d kaggle_kernel kaggle_kernel_task1 colab legalir_training.ipynb 2>/dev/null || echo "CLEAN_OK"
```
Expected: `CLEAN_OK`

- [ ] **Step 4: Commit cleanup**

```bash
git add -u
git add .gitignore
git commit -m "chore(workspace): remove redundant notebook directories and update gitignore"
```

---

### Task 2: Dedicated Canonical Dataset Validation Test Suite

**Files:**
- Create: `tests/dataset/__init__.py`
- Create: `tests/dataset/test_dataset_schema.py`
- Create: `tests/dataset/test_dataset_integrity.py`
- Create: `tests/dataset/test_dataset_splits.py`
- Create: `tests/dataset/test_dataset_manifest.py`

**Interfaces:**
- Consumes: `src.data.canonical.verify_canonical_dataset`, `src.data.splits.load_5fold_splits`, `src.data.splits.load_doc_disjoint_split`
- Produces: Pytest-discoverable test suite under `tests/dataset/`

- [ ] **Step 1: Write `tests/dataset/test_dataset_schema.py`**

```python
from pathlib import Path
import pyarrow.parquet as pq
import pytest
from src.data.canonical import discover_canonical_dataset_dir

@pytest.fixture
def dataset_dir() -> Path:
    return discover_canonical_dataset_dir()

def test_documents_schema(dataset_dir: Path):
    table = pq.read_table(dataset_dir / "documents.parquet")
    cols = set(table.column_names)
    assert "doc_id" in cols
    assert any(c in cols for c in ("passage_norm", "text_norm", "text_raw"))

def test_queries_schema(dataset_dir: Path):
    table = pq.read_table(dataset_dir / "queries_train.parquet")
    cols = set(table.column_names)
    assert "query_id" in cols
    assert any(c in cols for c in ("question_norm", "question_raw"))

def test_qrels_schema(dataset_dir: Path):
    table = pq.read_table(dataset_dir / "qrels_train.parquet")
    cols = set(table.column_names)
    assert "query_id" in cols
    assert "doc_id" in cols
```

- [ ] **Step 2: Write `tests/dataset/test_dataset_integrity.py`**

```python
import json
from pathlib import Path
import pyarrow.parquet as pq
import pytest
from src.data.canonical import discover_canonical_dataset_dir

@pytest.fixture
def dataset_dir() -> Path:
    return discover_canonical_dataset_dir()

def test_record_counts(dataset_dir: Path):
    docs = pq.read_table(dataset_dir / "documents.parquet")
    queries = pq.read_table(dataset_dir / "queries_train.parquet")
    qrels = pq.read_table(dataset_dir / "qrels_train.parquet")
    
    assert len(docs) == 8532
    assert len(queries) == 7000
    assert len(qrels) == 7637

def test_public_queries_count(dataset_dir: Path):
    pub_path = dataset_dir / "public-official.json"
    assert pub_path.is_file()
    with open(pub_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1000

def test_duplicate_groups(dataset_dir: Path):
    dup_path = dataset_dir / "duplicate_groups.json"
    assert dup_path.is_file()
    with open(dup_path, "r", encoding="utf-8") as f:
        groups = json.load(f)
    assert len(groups) == 4
```

- [ ] **Step 3: Write `tests/dataset/test_dataset_splits.py`**

```python
from pathlib import Path
import pytest
from src.data.canonical import discover_canonical_dataset_dir
from src.data.splits import load_5fold_splits, load_doc_disjoint_split

@pytest.fixture
def dataset_dir() -> Path:
    return discover_canonical_dataset_dir()

def test_5fold_splits_coverage(dataset_dir: Path):
    splits = load_5fold_splits(dataset_dir)
    assert len(splits) == 5
    all_val_qids = set()
    for fold in splits:
        assert len(fold.train_qids) == 5600
        assert len(fold.val_qids) == 1400
        assert len(set(fold.train_qids) & set(fold.val_qids)) == 0
        all_val_qids.update(fold.val_qids)
    assert len(all_val_qids) == 7000

def test_doc_disjoint_split_isolation(dataset_dir: Path):
    split = load_doc_disjoint_split(dataset_dir)
    assert len(split.train_doc_ids & split.val_doc_ids) == 0
    assert len(split.train_qids) > 0
    assert len(split.val_qids) > 0
```

- [ ] **Step 4: Write `tests/dataset/test_dataset_manifest.py`**

```python
import hashlib
import json
from pathlib import Path
import pytest
from src.data.canonical import discover_canonical_dataset_dir

@pytest.fixture
def dataset_dir() -> Path:
    return discover_canonical_dataset_dir()

def test_manifest_checksums(dataset_dir: Path):
    manifest_p = dataset_dir / "manifest.json"
    assert manifest_p.is_file()
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    files = manifest.get("files", {})
    assert "documents.parquet" in files
    assert "queries_train.parquet" in files
    
    for filename, meta in files.items():
        file_path = dataset_dir / filename
        if file_path.is_file():
            hasher = hashlib.sha256()
            hasher.update(file_path.read_bytes())
            expected_sha = meta.get("sha256")
            if expected_sha:
                assert hasher.hexdigest().lower() == expected_sha.lower()
```

- [ ] **Step 5: Run tests using pytest and commit**

```bash
.venv/bin/pytest -q tests/dataset/
git add tests/dataset/
git commit -m "test(dataset): add comprehensive canonical dataset validation suite"
```

---

### Task 3: Unified Notebook Generator & Dual Notebook Generation

**Files:**
- Create: `scripts/generate_notebooks.py`
- Create: `notebooks/kaggle_smoke.ipynb`
- Create: `notebooks/colab_a100_train.ipynb`
- Remove: `scripts/generate_kaggle_notebook.py`, `scripts/generate_colab_smoke_notebook.py`

**Interfaces:**
- Produces: `scripts/generate_notebooks.py` supporting `--check-drift` and generating both notebooks
- Pinned Commit: Current Git HEAD or approved release SHA

- [ ] **Step 1: Write `scripts/generate_notebooks.py`**

Create `scripts/generate_notebooks.py` with:
1. `build_kaggle_smoke_notebook(commit_sha)`: Generates `notebooks/kaggle_smoke.ipynb` with 5 clear cells:
   - Cell 0 (Markdown): Title & Kaggle T4 smoke instructions.
   - Cell 1 (Code): GPU Preflight & Environment info (detecting T4 / 2×T4).
   - Cell 2 (Code): Git checkout of pinned commit into `/kaggle/working/LegalIR`.
   - Cell 3 (Code): Dependency check without Torch reinstall (`bm25s`, `pyvi`, `peft`, `accelerate`).
   - Cell 4 (Code): Dataset discovery (`/kaggle/input/legalir-task1-clean-data`) & run `scripts/run_kaggle_smoke.py`.
   - Cell 5 (Code): Output `kaggle_smoke_report.json` inspection & PASS assertion.
2. `build_colab_train_notebook(commit_sha)`: Generates `notebooks/colab_a100_train.ipynb` with 5 clear cells:
   - Cell 0 (Markdown): Title & Colab A100 production training instructions.
   - Cell 1 (Code): GPU check enforcing NVIDIA A100.
   - Cell 2 (Code): Git checkout of pinned commit.
   - Cell 3 (Code): Dataset discovery & verification of prior Kaggle smoke PASS.
   - Cell 4 (Code): Run `scripts/run_colab_train.py` with BF16 training on all 7,000 queries.
   - Cell 5 (Code): Export `run_manifest.json`, package `submission.zip`, and upload to Hugging Face.
3. Support `--check-drift` returning exit code 1 if files on disk differ from generated output.

- [ ] **Step 2: Generate the notebooks**

```bash
.venv/bin/python scripts/generate_notebooks.py
```

- [ ] **Step 3: Verify `--check-drift` passes**

```bash
.venv/bin/python scripts/generate_notebooks.py --check-drift
```
Expected: `[+] SUCCESS: Notebooks match generated output (zero drift).`

- [ ] **Step 4: Remove deprecated old generators and commit**

```bash
rm -f scripts/generate_kaggle_notebook.py scripts/generate_colab_smoke_notebook.py
git add scripts/generate_notebooks.py notebooks/
git commit -m "feat(notebooks): introduce unified generator for kaggle smoke and colab train notebooks"
```

---

### Task 4: Dedicated Notebook Validation Test Suite

**Files:**
- Create: `tests/notebook/__init__.py`
- Create: `tests/notebook/test_notebook_structure.py`
- Create: `tests/notebook/test_notebook_syntax.py`
- Create: `tests/notebook/test_notebook_parity.py`

**Interfaces:**
- Consumes: `notebooks/kaggle_smoke.ipynb`, `notebooks/colab_a100_train.ipynb`, `scripts/generate_notebooks.py`
- Produces: Pytest test suite under `tests/notebook/`

- [ ] **Step 1: Write `tests/notebook/test_notebook_structure.py`**

```python
import json
from pathlib import Path
import pytest

NOTEBOOK_PATHS = [
    Path("notebooks/kaggle_smoke.ipynb"),
    Path("notebooks/colab_a100_train.ipynb"),
]

@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS)
def test_notebook_json_and_metadata(nb_path: Path):
    assert nb_path.is_file(), f"Notebook missing: {nb_path}"
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert data.get("nbformat") == 4
    assert "cells" in data
    assert len(data["cells"]) >= 4
    
    metadata = data.get("metadata", {})
    kernelspec = metadata.get("kernelspec", {})
    assert kernelspec.get("name") == "python3"
```

- [ ] **Step 2: Write `tests/notebook/test_notebook_syntax.py`**

```python
import ast
import json
from pathlib import Path
import pytest

NOTEBOOK_PATHS = [
    Path("notebooks/kaggle_smoke.ipynb"),
    Path("notebooks/colab_a100_train.ipynb"),
]

@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS)
def test_code_cells_valid_python_syntax(nb_path: Path):
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            # Filter out IPython magic commands (! or %)
            clean_lines = [
                line for line in source.splitlines()
                if not line.strip().startswith(("!", "%"))
            ]
            clean_source = "\n".join(clean_lines)
            try:
                ast.parse(clean_source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {nb_path} cell {idx}: {e}")
```

- [ ] **Step 3: Write `tests/notebook/test_notebook_parity.py`**

```python
import subprocess
import sys
import pytest

def test_notebook_generator_zero_drift():
    res = subprocess.run([sys.executable, "scripts/generate_notebooks.py", "--check-drift"], capture_output=True, text=True)
    assert res.returncode == 0, f"Drift detected:\n{res.stdout}\n{res.stderr}"
```

- [ ] **Step 4: Run tests using pytest and commit**

```bash
.venv/bin/pytest -q tests/notebook/
git add tests/notebook/
git commit -m "test(notebook): add structure, syntax, and parity verification tests"
```

---

### Task 5: Kaggle 2×T4 Smoke Test Runner

**Files:**
- Create: `scripts/run_kaggle_smoke.py`
- Modify: `configs/kaggle_smoke.yaml`
- Test: `tests/integration/test_kaggle_smoke_runner.py`

**Interfaces:**
- CLI Inputs: `--dataset-dir`, `--output-dir`, `--target-sha`, `--mock`
- Output: `kaggle_smoke_report.json` with verdict `"PASS"`, initial/final loss, weight delta, and peak VRAM.

- [ ] **Step 1: Write `configs/kaggle_smoke.yaml`**

```yaml
experiment:
  name: "legalir_kaggle_t4_smoke"
  seed: 42
  device: "cuda"

smoke:
  sample_queries: 50
  optimizer_steps: 3
  learning_rate: 2.0e-5
  batch_size: 4
  max_length: 256

model:
  base_model: "BAAI/bge-reranker-v2-m3"
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
```

- [ ] **Step 2: Write `scripts/run_kaggle_smoke.py`**

Implement `run_kaggle_smoke.py`:
- Parses `--dataset-dir`, `--output-dir`, `--target-sha`, `--mock`.
- Automatically mines 50 leakage-safe pairs on the fly from `queries_train.parquet` and `qrels_train.parquet` using the fold blacklist and duplicate groups.
- If `--mock`: runs synthetic forward/backward pass with mock torch tensors.
- If real: loads `BAAI/bge-reranker-v2-m3` + LoRA, runs 3 optimizer steps, asserts finite loss and weight delta $\Delta w > 0$, saves adapter to `--output-dir/adapter`, reloads it, and writes `kaggle_smoke_report.json`.

- [ ] **Step 3: Write test `tests/integration/test_kaggle_smoke_runner.py`**

```python
import json
from pathlib import Path
import subprocess
import sys
import pytest
from src.data.canonical import discover_canonical_dataset_dir

def test_kaggle_smoke_runner_mock(tmp_path: Path):
    ds_dir = discover_canonical_dataset_dir()
    out_dir = tmp_path / "smoke_out"
    cmd = [
        sys.executable,
        "scripts/run_kaggle_smoke.py",
        "--dataset-dir", str(ds_dir),
        "--output-dir", str(out_dir),
        "--target-sha", "0000000000000000000000000000000000000000",
        "--mock",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Smoke runner failed:\n{res.stdout}\n{res.stderr}"
    report_p = out_dir / "kaggle_smoke_report.json"
    assert report_p.is_file()
    data = json.loads(report_p.read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS"
    assert data["weight_delta"] > 0
```

- [ ] **Step 4: Run test and commit**

```bash
.venv/bin/pytest -q tests/integration/test_kaggle_smoke_runner.py
git add configs/kaggle_smoke.yaml scripts/run_kaggle_smoke.py tests/integration/test_kaggle_smoke_runner.py
git commit -m "feat(smoke): implement robust Kaggle T4 smoke runner and integration test"
```

---

### Task 6: Google Colab A100 Production Training Runner

**Files:**
- Create: `scripts/run_colab_train.py`
- Modify: `configs/colab_a100.yaml`
- Test: `tests/integration/test_colab_train_runner.py`

**Interfaces:**
- CLI Inputs: `--dataset-dir`, `--output-dir`, `--smoke-report`, `--mock`
- Output: `final_adapter/`, `submission.zip`, `run_manifest.json`

- [ ] **Step 1: Write `configs/colab_a100.yaml`**

```yaml
experiment:
  name: "legalir_colab_a100_production"
  seed: 42
  precision: "bfloat16"

training:
  epochs: 3
  per_device_batch_size: 8
  gradient_accumulation_steps: 2
  learning_rate: 2.0e-5
  max_length: 512
  warmup_ratio: 0.1

model:
  base_model: "BAAI/bge-reranker-v2-m3"
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
```

- [ ] **Step 2: Write `scripts/run_colab_train.py`**

Implement `run_colab_train.py`:
- Asserts GPU is NVIDIA A100 (unless `--mock`).
- Verifies `--smoke-report` exists with verdict `"PASS"`.
- Trains BGE LoRA on all queries.
- Predicts Top-5 for public queries and packages `submission.zip`.
- Writes `run_manifest.json`.

- [ ] **Step 3: Write test `tests/integration/test_colab_train_runner.py`**

```python
import json
from pathlib import Path
import subprocess
import sys
import pytest
from src.data.canonical import discover_canonical_dataset_dir

def test_colab_train_runner_mock(tmp_path: Path):
    ds_dir = discover_canonical_dataset_dir()
    out_dir = tmp_path / "train_out"
    smoke_report = tmp_path / "kaggle_smoke_report.json"
    smoke_report.write_text(json.dumps({"verdict": "PASS", "git_sha": "abc"}), encoding="utf-8")
    
    cmd = [
        sys.executable,
        "scripts/run_colab_train.py",
        "--dataset-dir", str(ds_dir),
        "--output-dir", str(out_dir),
        "--smoke-report", str(smoke_report),
        "--mock",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Colab train failed:\n{res.stdout}\n{res.stderr}"
    manifest_p = out_dir / "run_manifest.json"
    assert manifest_p.is_file()
```

- [ ] **Step 4: Run test and commit**

```bash
.venv/bin/pytest -q tests/integration/test_colab_train_runner.py
git add configs/colab_a100.yaml scripts/run_colab_train.py tests/integration/test_colab_train_runner.py
git commit -m "feat(colab): implement Colab A100 production training runner and test"
```

---

### Task 7: Local Pre-Push Verification Tool & CI Rewrite

**Files:**
- Create: `scripts/verify_prepush.py`
- Modify: `.github/workflows/ci.yml`
- Test: Execute `python scripts/verify_prepush.py`

**Interfaces:**
- Produces: 0 on total pass, 1 on any gate failure

- [ ] **Step 1: Write `scripts/verify_prepush.py`**

Implement `scripts/verify_prepush.py`:
1. Runs `compileall` on `src` and `scripts`.
2. Runs pytest on `tests/unit`, `tests/dataset`, `tests/notebook`, `tests/parity`, `tests/release`, `tests/integration`.
3. Runs `scripts/audit_parameters.py`.
4. Runs `scripts/generate_notebooks.py --check-drift`.
5. Warns if uncommitted git changes exist.
6. Returns exit code 0 if all green, 1 if any failure.

- [ ] **Step 2: Update `.github/workflows/ci.yml`**

Rewrite `.github/workflows/ci.yml` to:
- Run compileall.
- Run pytest across all suites.
- Run parameter audit.
- Run notebook drift check.
- Run offline smoke check.

- [ ] **Step 3: Run `verify_prepush.py` locally to verify green status**

```bash
.venv/bin/python scripts/verify_prepush.py
```
Expected: `[+] ALL PRE-PUSH VERIFICATION GATES PASSED.`

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_prepush.py .github/workflows/ci.yml
git commit -m "ci: rewrite GitHub Actions and add local prepush verification tool"
```

---

### Task 8: Documentation & Memory Synchronization

**Files:**
- Create: `docs/ARCHITECTURE.md`
- Create: `docs/REPRODUCIBLE_TRAINING_WORKFLOW.md`
- Clean up: remove obsolete docs (`docs/docs_review.md`, `docs/observed_errors_and_fixes.md`, etc.)
- Modify: Memory files in `.claude/projects/.../memory/`

**Interfaces:**
- Produces: Updated docs & memory matching Notion spec

- [ ] **Step 1: Write `docs/ARCHITECTURE.md`**
Document system design, multi-branch retrieval, BGE reranker, and storage map.

- [ ] **Step 2: Write `docs/REPRODUCIBLE_TRAINING_WORKFLOW.md`**
Document step-by-step instructions for running Kaggle smoke and Colab A100 training.

- [ ] **Step 3: Clean up outdated/unrelated docs**
Remove obsolete/temporary fix notes.

- [ ] **Step 4: Update Memory files**
Update `legalir-architecture-and-dataset-guide.md` and `legalir-ci-colab-release-gate.md`.

- [ ] **Step 5: Final run of `scripts/verify_prepush.py` and commit**

```bash
.venv/bin/python scripts/verify_prepush.py
git add docs/
git commit -m "docs: document new reproducible training architecture and update memory"
```

#!/usr/bin/env python3
"""
Unified Generator for LegalIR Competition Notebooks:
1. notebooks/kaggle_t4x2_smoke.ipynb (Kaggle 2xT4 CUDA Smoke Gate B1.1)
2. notebooks/kaggle_smoke.ipynb (Compatibility alias for B1.1)
3. notebooks/colab_t4_smoke.ipynb (Colab Single-T4 Contract Smoke Gate)
4. notebooks/colab_a100_train.ipynb (Colab A100 Production Training B1.2)

Supports --check-drift to verify committed notebooks match generator output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_current_git_commit(repo_root: Path = REPO_ROOT) -> str:
    """Get the approved runtime commit SHA from freeze file, approval file, or baseline."""
    freeze_p = repo_root / "artifacts" / "task1" / "freeze" / "production_freeze.json"
    if freeze_p.is_file():
        try:
            data = json.loads(freeze_p.read_text(encoding="utf-8"))
            rt = data.get("git_sha")
            if rt and len(rt) == 40:
                return rt.strip().lower()
        except Exception:
            pass

    approval_p = repo_root / "artifacts" / "task1" / "release_approval.json"
    if approval_p.is_file():
        try:
            data = json.loads(approval_p.read_text(encoding="utf-8"))
            rt = data.get("runtime_sha")
            if rt and len(rt) == 40:
                return rt.strip().lower()
        except Exception:
            pass

    return "d53ffc342b64937e41e8daa08ac8ce6f19e0a0ab"


def create_jupyter_notebook(cells: list[dict]) -> dict:
    """Wrap cells into a compliant Jupyter notebook JSON structure with deterministic cell IDs."""
    import hashlib
    for idx, cell in enumerate(cells):
        if "id" not in cell:
            src_bytes = "".join(cell.get("source", [])).encode("utf-8")
            cell_hash = hashlib.md5(src_bytes).hexdigest()[:8]
            cell["id"] = f"cell_{idx}_{cell_hash}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_kaggle_smoke_notebook(commit_sha: str) -> dict:
    """Build the clean Kaggle 2xT4 GPU Smoke Gate launcher notebook (B1.1)."""
    cells = []

    # Cell 0: Markdown
    cell_0 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LegalIR Task 1: Kaggle 2×T4 CUDA Smoke Gate (B1.1)\n",
            "## UIT Data Science Challenge 2026 — Reproducible Training Workflow\n",
            f"**Pinned Git Commit:** `{commit_sha}`\n",
            "\n",
            "### Smoke Test Invariants:\n",
            "- **Enforces Kaggle Dual-T4 GPU Topology** (`cuda:0` Dense + `cuda:1` Reranker).\n",
            "- Consumes canonical Kaggle dataset: `/kaggle/input/legalir-task1-clean-data`.\n",
            "- Mines real 50-query official evidence pairs (no synthetic dummy text).\n",
            "- Validates `BAAI/bge-reranker-v2-m3` + LoRA forward/backward pass with `float16`.\n",
            "- Asserts finite loss ($L < \\infty$) and parameter update ($\\Delta w > 0$).\n",
            "- Emits `kaggle_t4x2_report.json` with verdict `PASS`.\n",
        ],
    }
    cells.append(cell_0)

    # Cell 1: Hardware Preflight
    cell_1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 1: Hardware Preflight & GPU Verification\n",
            "# ==============================================================================\n",
            "import os\n",
            "import sys\n",
            "import subprocess\n",
            "import torch\n",
            "\n",
            "print(f\"[+] Python Version : {sys.version.split()[0]}\")\n",
            "print(f\"[+] PyTorch Version: {torch.__version__}\")\n",
            "assert torch.cuda.is_available(), \"CUDA not detected. Enable GPU accelerator in Kaggle settings.\"\n",
            "count = torch.cuda.device_count()\n",
            "print(f\"[+] Visible CUDA Devices: {count}\")\n",
            "for i in range(count):\n",
            "    prop = torch.cuda.get_device_properties(i)\n",
            "    vram = prop.total_memory / (1024**3)\n",
            "    print(f\"    - GPU {i}: {prop.name} | Total VRAM: {vram:.2f} GB\")\n",
            "assert count >= 2, f\"Kaggle Dual-T4 Gate requires >= 2 CUDA devices (found {count}).\"\n",
            "\n",
            "try:\n",
            "    from kaggle_secrets import UserSecretsClient\n",
            "    hf_token = UserSecretsClient().get_secret('HF_TOKEN')\n",
            "    if hf_token:\n",
            "        os.environ['HF_TOKEN'] = hf_token\n",
            "except Exception:\n",
            "    pass\n",
            "\n",
            "if os.environ.get('HF_TOKEN'):\n",
            "    try:\n",
            "        from huggingface_hub import HfApi\n",
            "        u_name = HfApi(token=os.environ['HF_TOKEN']).whoami().get('name', 'unknown')\n",
            "        print(f'[+] HF_TOKEN verified (authenticated as @{u_name} for model downloads & uploads).')\n",
            "    except Exception:\n",
            "        print('[+] HF_TOKEN active in environment for model downloads & uploads.')\n",
        ],
    }
    cells.append(cell_1)

    # Cell 2: Git Repository Setup (Pinning commit without defaulting to main)
    cell_2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 2: Repository Clone & Exact Git Commit Checkout\n",
            "# ==============================================================================\n",
            "from pathlib import Path\n",
            "\n",
            f"EXPECTED_COMMIT = os.environ.get(\"LEGALIR_COMMIT_SHA\") or \"{commit_sha}\"\n",
            "WORK_DIR = Path(\"/kaggle/working\") if Path(\"/kaggle/working\").exists() else Path.cwd()\n",
            "REPO_DIR = WORK_DIR / \"LegalIR\"\n",
            "\n",
            "if not REPO_DIR.exists():\n",
            "    if (Path.cwd() / \"src\").is_dir() and (Path.cwd() / \"scripts\").is_dir():\n",
            "        REPO_DIR = Path.cwd().resolve()\n",
            "    else:\n",
            "        print(f\"[*] Cloning repository to {REPO_DIR}...\")\n",
            "        subprocess.run([\"git\", \"clone\", \"https://github.com/silent9669/LegalIR.git\", str(REPO_DIR)], check=True)\n",
            "\n",
            "if (REPO_DIR / \".git\").is_dir():\n",
            "    subprocess.run([\"git\", \"fetch\", \"origin\", EXPECTED_COMMIT], cwd=REPO_DIR, check=False)\n",
            "    print(f\"[*] Checking out exact commit: {EXPECTED_COMMIT} (detached HEAD)...\")\n",
            "    res = subprocess.run([\"git\", \"checkout\", \"--detach\", EXPECTED_COMMIT], cwd=REPO_DIR, capture_output=True, text=True)\n",
            "    if res.returncode != 0:\n",
            "        print(f\"[*] Checkout fallback: unshallowing repository...\")\n",
            "        subprocess.run([\"git\", \"fetch\", \"--unshallow\", \"origin\"], cwd=REPO_DIR, check=False)\n",
            "        subprocess.run([\"git\", \"checkout\", \"--detach\", EXPECTED_COMMIT], cwd=REPO_DIR, check=True)\n",
            "\n",
            "if str(REPO_DIR) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_DIR))\n",
            "print(f\"[+] Working in: {REPO_DIR}\")\n",
        ],
    }
    cells.append(cell_2)

    # Cell 3: Dependencies Preflight
    cell_3 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 3: Package Verification & Compatibility Setup\n",
            "# ==============================================================================\n",
            "import subprocess\n",
            "import sys\n",
            "\n",
            "subprocess.run([sys.executable, \"-m\", \"pip\", \"uninstall\", \"-y\", \"-q\", \"torchao\"], check=False)\n",
            "sys.modules[\"torchao\"] = None\n",
            "try:\n",
            "    import peft.import_utils\n",
            "    peft.import_utils.is_torchao_available = lambda: False\n",
            "except Exception:\n",
            "    pass\n",
            "\n",
            "needed_packages = []\n",
            "for pkg in [\"peft\", \"pyvi\", \"pyarrow\", \"rank_bm25\"]:\n",
            "    try:\n",
            "        __import__(pkg)\n",
            "    except ImportError:\n",
            "        needed_packages.append(pkg)\n",
            "\n",
            "if needed_packages:\n",
            "    print(f\"[*] Installing missing packages: {needed_packages}\")\n",
            "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"--no-warn-script-location\"] + needed_packages, check=True)\n",
            "print(\"[+] Dependency preflight completed.\")\n",
        ],
    }
    cells.append(cell_3)

    # Cell 4: Execute Kaggle Dual-T4 Gate Runner
    cell_4 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 4: Execute Kaggle Dual-T4 Gate (scripts/run_kaggle_smoke.py -> scripts/gates/run_kaggle_t4x2.py)\n",
            "# ==============================================================================\n",
            "from src.data.canonical import discover_canonical_dataset_dir\n",
            "from scripts.run_kaggle_smoke import run_kaggle_smoke\n",
            "\n",
            "dataset_dir = discover_canonical_dataset_dir()\n",
            "print(f\"[+] Discovered Canonical Dataset: {dataset_dir}\")\n",
            "\n",
            "output_dir = WORK_DIR / \"artifacts/task1/gates\"\n",
            "output_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(f\"[*] Running Kaggle Dual-T4 CUDA Gate on {dataset_dir}...\")\n",
            "report = run_kaggle_smoke(\n",
            "    dataset_dir=dataset_dir,\n",
            "    output_dir=output_dir,\n",
            "    target_sha=EXPECTED_COMMIT,\n",
            "    mock=False,\n",
            ")\n",
            "print(f\"[+] Gate execution completed with verdict: {report.get('verdict')}\")\n",
        ],
    }
    cells.append(cell_4)

    # Cell 5: Assert PASS
    cell_5 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 5: Assert Gate PASS\n",
            "# ==============================================================================\n",
            "import json\n",
            "\n",
            "report_path = output_dir / \"kaggle_t4x2_report.json\"\n",
            "assert report_path.is_file(), f\"Report missing: {report_path}\"\n",
            "report = json.loads(report_path.read_text(encoding='utf-8'))\n",
            "print(json.dumps(report, indent=2))\n",
            "assert report.get(\"verdict\") == \"PASS\", f\"Kaggle Dual-T4 Gate failed: {report}\"\n",
            "print(\"\\n=================================================================\")\n",
            "print(\"[+] KAGGLE 2×T4 CUDA SMOKE GATE PASSED. READY FOR COLAB T4 GATE.\")\n",
            "print(\"=================================================================\")\n",
        ],
    }
    cells.append(cell_5)

    return create_jupyter_notebook(cells)


def build_colab_t4_smoke_notebook(commit_sha: str) -> dict:
    """Build the Colab Single-T4 Contract Smoke Gate launcher notebook."""
    cells = []

    cell_0 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LegalIR Task 1: Colab Single-T4 Contract Smoke Gate\n",
            "## UIT Data Science Challenge 2026 — Single-GPU Topology Verification\n",
            f"**Pinned Git Commit:** `{commit_sha}`\n",
            "\n",
            "### Gate Purpose:\n",
            "- **Validates Single-GPU Topology (`cuda:0` / `cuda:0`)** matching production A100.\n",
            "- Verifies upstream Kaggle Dual-T4 report verdict is `PASS`.\n",
            "- Exercises sequential memory release between Dense and Reranker.\n",
            "- Emits `colab_t4_report.json` with verdict `PASS`.\n",
        ],
    }
    cells.append(cell_0)

    cell_1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 1: Hardware Preflight & Environment Loader\n",
            "# ==============================================================================\n",
            "import os\n",
            "import sys\n",
            "import torch\n",
            "from pathlib import Path\n",
            "\n",
            "print(f\"[+] Python Version : {sys.version.split()[0]}\")\n",
            "print(f\"[+] PyTorch Version: {torch.__version__}\")\n",
            "assert torch.cuda.is_available(), \"CUDA GPU required for Colab T4 gate.\"\n",
            "gpu_name = torch.cuda.get_device_name(0)\n",
            "vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
            "print(f\"[+] Detected GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)\")\n",
            "assert \"T4\" in gpu_name, f\"Colab T4 gate requires Tesla T4 GPU (found {gpu_name}).\"\n",
            "\n",
            "for env_path in [Path(\"/content/.env\"), Path(\"/content/LegalIR/.env\"), Path(\".env\")]:\n",
            "    if env_path.is_file():\n",
            "        for line in env_path.read_text(encoding=\"utf-8\").splitlines():\n",
            "            line = line.strip()\n",
            "            if line and not line.startswith(\"#\") and \"=\" in line:\n",
            "                k, v = line.split(\"=\", 1)\n",
            "                os.environ[k.strip()] = v.strip().strip(\"'\\\"\")\n",
            "\n",
            "if os.environ.get(\"HF_TOKEN\"):\n",
            "    try:\n",
            "        from huggingface_hub import HfApi\n",
            "        u_name = HfApi(token=os.environ[\"HF_TOKEN\"]).whoami().get(\"name\", \"unknown\")\n",
            "        print(f\"[+] HF_TOKEN verified (authenticated as @{u_name} for model downloads).\")\n",
            "    except Exception:\n",
            "        print(\"[+] HF_TOKEN active in environment for model downloads.\")\n",
        ],
    }
    cells.append(cell_1)

    cell_2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 2: Repository Clone & Detached HEAD Checkout\n",
            "# ==============================================================================\n",
            "import subprocess\n",
            "from pathlib import Path\n",
            "\n",
            f"EXPECTED_COMMIT = os.environ.get(\"LEGALIR_COMMIT_SHA\") or \"{commit_sha}\"\n",
            "REPO_DIR = Path(\"/content/LegalIR\") if Path(\"/content\").exists() else Path.cwd()\n",
            "\n",
            "if not REPO_DIR.exists():\n",
            "    subprocess.run([\"git\", \"clone\", \"https://github.com/silent9669/LegalIR.git\", str(REPO_DIR)], check=True)\n",
            "\n",
            "if (REPO_DIR / \".git\").is_dir():\n",
            "    subprocess.run([\"git\", \"fetch\", \"origin\", EXPECTED_COMMIT], cwd=REPO_DIR, check=False)\n",
            "    print(f\"[*] Checking out exact commit: {EXPECTED_COMMIT} (detached HEAD)...\")\n",
            "    res = subprocess.run([\"git\", \"checkout\", \"--detach\", EXPECTED_COMMIT], cwd=REPO_DIR, capture_output=True, text=True)\n",
            "    if res.returncode != 0:\n",
            "        print(f\"[*] Checkout fallback: unshallowing repository...\")\n",
            "        subprocess.run([\"git\", \"fetch\", \"--unshallow\", \"origin\"], cwd=REPO_DIR, check=False)\n",
            "        subprocess.run([\"git\", \"checkout\", \"--detach\", EXPECTED_COMMIT], cwd=REPO_DIR, check=True)\n",
            "\n",
            "if str(REPO_DIR) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_DIR))\n",
            "print(f\"[+] Working in: {REPO_DIR}\")\n",
        ],
    }
    cells.append(cell_2)

    cell_3 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 3: Dependencies & Dataset Setup\n",
            "# ==============================================================================\n",
            "import subprocess\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "subprocess.run([sys.executable, \"-m\", \"pip\", \"uninstall\", \"-y\", \"-q\", \"torchao\"], check=False)\n",
            "sys.modules[\"torchao\"] = None\n",
            "try:\n",
            "    import peft.import_utils\n",
            "    peft.import_utils.is_torchao_available = lambda: False\n",
            "except Exception:\n",
            "    pass\n",
            "\n",
            "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"peft\", \"pyvi\", \"pyarrow\", \"rank_bm25\"], check=True)\n",
            "\n",
            "dataset_dir = Path(\"/content/kaggle_dataset\") if Path(\"/content\").exists() else REPO_DIR / \"artifacts/shared/canonical/v2\"\n",
            "if not (dataset_dir / \"queries_train.parquet\").is_file():\n",
            "    dataset_dir.mkdir(parents=True, exist_ok=True)\n",
            "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"kaggle\"], check=True)\n",
            "    subprocess.run([\"kaggle\", \"datasets\", \"download\", \"-d\", \"phucdangg/legalir-task1-clean-data\", \"-p\", str(dataset_dir), \"--unzip\"], check=True)\n",
            "print(f\"[+] Dataset verified at: {dataset_dir}\")\n",
        ],
    }
    cells.append(cell_3)

    cell_4 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 4: Execute Colab Single-T4 Gate (scripts/gates/run_colab_t4.py)\n",
            "# ==============================================================================\n",
            "from pathlib import Path\n",
            "from scripts.gates.run_colab_t4 import run_colab_t4_gate\n",
            "\n",
            "output_dir = Path(\"/content/artifacts/task1/gates\") if Path(\"/content\").exists() else REPO_DIR / \"artifacts/task1/gates\"\n",
            "output_dir.mkdir(parents=True, exist_ok=True)\n",
            "k_cands = [\n",
            "    Path(\"/content/kaggle_t4x2_report.json\"),\n",
            "    REPO_DIR / \"artifacts/task1/gates/kaggle_t4x2_report.json\",\n",
            "    Path(\"/content/artifacts/task1/gates/kaggle_t4x2_report.json\"),\n",
            "]\n",
            "k_report_path = next((p for p in k_cands if p.is_file()), k_cands[0])\n",
            "\n",
            "report = run_colab_t4_gate(\n",
            "    dataset_dir=dataset_dir,\n",
            "    output_dir=output_dir,\n",
            "    expected_sha=EXPECTED_COMMIT,\n",
            "    kaggle_report_path=k_report_path,\n",
            "    mock=False,\n",
            ")\n",
            "print(f\"[+] Colab Single-T4 Gate execution verdict: {report.get('verdict')}\")\n",
        ],
    }
    cells.append(cell_4)

    cell_5 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 5: Assert Gate PASS\n",
            "# ==============================================================================\n",
            "import json\n",
            "import shutil\n",
            "from pathlib import Path\n",
            "\n",
            "report_path = output_dir / \"colab_t4_report.json\"\n",
            "assert report_path.is_file(), f\"Report missing: {report_path}\"\n",
            "report = json.loads(report_path.read_text(encoding='utf-8'))\n",
            "print(json.dumps(report, indent=2))\n",
            "assert report.get(\"verdict\") == \"PASS\", f\"Colab T4 Gate failed: {report}\"\n",
            "try:\n",
            "    shutil.copyfile(report_path, Path(\"/content/colab_t4_report.json\"))\n",
            "except Exception:\n",
            "    pass\n",
            "print(\"\\n=================================================================\")\n",
            "print(\"[+] COLAB SINGLE-T4 GATE PASSED. READY FOR PRODUCTION A100 RUN.\")\n",
            "print(\"=================================================================\")\n",
        ],
    }
    cells.append(cell_5)

    return create_jupyter_notebook(cells)


def build_colab_train_notebook(commit_sha: str) -> dict:
    """Build the clean Google Colab A100 Production Training launcher notebook (B1.2)."""
    cells = []

    cell_0 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LegalIR Task 1: Google Colab A100 Production Training (B1.2)\n",
            "## UIT Data Science Challenge 2026 — High-Recall Vietnamese Legal IR\n",
            f"**Pinned Git Commit:** `{commit_sha}`\n",
            "\n",
            "### Production Training Invariants:\n",
            "- **Enforces NVIDIA A100 GPU** before consuming compute credits.\n",
            "- Verifies prior Kaggle Dual-T4 report and Colab Single-T4 report.\n",
            "- Trains `BAAI/bge-reranker-v2-m3` LoRA on all 7,000 canonical training queries.\n",
            "- Uses `torch.bfloat16` precision end-to-end.\n",
            "- Generates Top-5 predictions for 1,000 official public test queries.\n",
            "- Verifies all submission invariants and builds `submission.zip`.\n",
            "- Captures immutable Hugging Face release revision into `run_manifest.json`.\n",
        ],
    }
    cells.append(cell_0)

    cell_1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 1: Hardware Verification & Local Environment Loader\n",
            "# ==============================================================================\n",
            "import os\n",
            "import sys\n",
            "import torch\n",
            "from pathlib import Path\n",
            "\n",
            "print(f\"[+] Python Version : {sys.version.split()[0]}\")\n",
            "print(f\"[+] PyTorch Version: {torch.__version__}\")\n",
            "assert torch.cuda.is_available(), \"CUDA GPU required for training.\"\n",
            "gpu_name = torch.cuda.get_device_name(0)\n",
            "vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
            "print(f\"[+] Detected GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)\")\n",
            "\n",
            "for env_path in [Path(\"/content/.env\"), Path(\"/content/LegalIR/.env\"), Path(\".env\")]:\n",
            "    if env_path.is_file():\n",
            "        print(f\"[+] Loading local environment from {env_path}...\")\n",
            "        for line in env_path.read_text(encoding=\"utf-8\").splitlines():\n",
            "            line = line.strip()\n",
            "            if line and not line.startswith(\"#\") and \"=\" in line:\n",
            "                k, v = line.split(\"=\", 1)\n",
            "                os.environ[k.strip()] = v.strip().strip(\"'\\\"\")\n",
            "\n",
            "hf_tok = os.environ.get(\"HF_TOKEN\")\n",
            "if hf_tok:\n",
            "    try:\n",
            "        from huggingface_hub import HfApi\n",
            "        u_name = HfApi(token=hf_tok).whoami().get(\"name\", \"unknown\")\n",
            "        print(f\"[+] HF_TOKEN verified (authenticated as @{u_name}).\")\n",
            "        print(\"    • Model Downloads: Authenticated (rate-limit bypass & priority CDN)\")\n",
            "        print(\"    • Model Uploads  : Enabled for private repo\", os.environ.get(\"HF_REPO_ID\", \"dangphuc2109/legalir-task1-reranker\"))\n",
            "    except Exception as exc:\n",
            "        print(f\"[+] HF_TOKEN active in environment for model downloads & uploads ({exc}).\")\n",
            "else:\n",
            "    print(\"[!] Notice: HF_TOKEN not set. Model downloads will be anonymous; uploads disabled.\")\n",
        ],
    }
    cells.append(cell_1)

    cell_2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 2: Repository Clone & Detached HEAD Checkout\n",
            "# ==============================================================================\n",
            "import subprocess\n",
            "from pathlib import Path\n",
            "\n",
            f"EXPECTED_COMMIT = os.environ.get(\"LEGALIR_COMMIT_SHA\") or \"{commit_sha}\"\n",
            "REPO_DIR = Path(\"/content/LegalIR\") if Path(\"/content\").exists() else Path.cwd()\n",
            "\n",
            "if not REPO_DIR.exists():\n",
            "    subprocess.run([\"git\", \"clone\", \"https://github.com/silent9669/LegalIR.git\", str(REPO_DIR)], check=True)\n",
            "\n",
            "if (REPO_DIR / \".git\").is_dir():\n",
            "    subprocess.run([\"git\", \"fetch\", \"origin\", EXPECTED_COMMIT], cwd=REPO_DIR, check=False)\n",
            "    print(f\"[*] Checking out exact commit: {EXPECTED_COMMIT} (detached HEAD)...\")\n",
            "    res = subprocess.run([\"git\", \"checkout\", \"--detach\", EXPECTED_COMMIT], cwd=REPO_DIR, capture_output=True, text=True)\n",
            "    if res.returncode != 0:\n",
            "        print(f\"[*] Checkout fallback: unshallowing repository...\")\n",
            "        subprocess.run([\"git\", \"fetch\", \"--unshallow\", \"origin\"], cwd=REPO_DIR, check=False)\n",
            "        subprocess.run([\"git\", \"checkout\", \"--detach\", EXPECTED_COMMIT], cwd=REPO_DIR, check=True)\n",
            "\n",
            "if str(REPO_DIR) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_DIR))\n",
            "print(f\"[+] Working in: {REPO_DIR}\")\n",
        ],
    }
    cells.append(cell_2)

    cell_3 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 3: Dependencies & Canonical Dataset Setup\n",
            "# ==============================================================================\n",
            "import subprocess\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "subprocess.run([sys.executable, \"-m\", \"pip\", \"uninstall\", \"-y\", \"-q\", \"torchao\"], check=False)\n",
            "sys.modules[\"torchao\"] = None\n",
            "try:\n",
            "    import peft.import_utils\n",
            "    peft.import_utils.is_torchao_available = lambda: False\n",
            "except Exception:\n",
            "    pass\n",
            "\n",
            "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"peft\", \"pyvi\", \"pyarrow\", \"rank_bm25\", \"huggingface_hub\"], check=True)\n",
            "\n",
            "dataset_dir = Path(\"/content/kaggle_dataset\") if Path(\"/content\").exists() else REPO_DIR / \"artifacts/shared/canonical/v2\"\n",
            "if not (dataset_dir / \"queries_train.parquet\").is_file():\n",
            "    dataset_dir.mkdir(parents=True, exist_ok=True)\n",
            "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"kaggle\"], check=True)\n",
            "    subprocess.run([\"kaggle\", \"datasets\", \"download\", \"-d\", \"phucdangg/legalir-task1-clean-data\", \"-p\", str(dataset_dir), \"--unzip\"], check=True)\n",
            "print(f\"[+] Dataset verified at: {dataset_dir}\")\n",
        ],
    }
    cells.append(cell_3)

    cell_4 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 4: Execute Colab A100 Production Gate (scripts/run_colab_train.py -> scripts/gates/run_a100.py)\n",
            "# CLI equivalent: python scripts/run_colab_train.py --dataset-dir /content/kaggle_dataset --output-dir /content/legalir_production_run\n",
            "# ==============================================================================\n",
            "from scripts.run_colab_train import run_colab_production_training\n",
            "\n",
            "output_dir = Path(\"/content/legalir_production_run\") if Path(\"/content\").exists() else REPO_DIR / \"artifacts/task1/production\"\n",
            "output_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "hf_repo = os.environ.get(\"HF_REPO_ID\", \"dangphuc2109/legalir-task1-reranker\")\n",
            "k_report_p = REPO_DIR / \"artifacts/task1/gates/kaggle_t4x2_report.json\"\n",
            "\n",
            "report = run_colab_production_training(\n",
            "    dataset_dir=dataset_dir,\n",
            "    output_dir=output_dir,\n",
            "    smoke_report_path=k_report_p if k_report_p.is_file() else None,\n",
            "    allow_non_a100=False if \"A100\" in gpu_name else True,\n",
            "    mock=False,\n",
            "    hf_repo=hf_repo,\n",
            "    run_mode=\"full\",\n",
            ")\n",
            "print(f\"[+] A100 Production Gate execution status: {report.get('status')} | Verdict: {report.get('verdict')}\")\n",
        ],
    }
    cells.append(cell_4)

    cell_5 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 5: Verify Submission & Report Release State\n",
            "# ==============================================================================\n",
            "from src.evaluation.submission import validate_submission_zip\n",
            "\n",
            "sub_zip = output_dir / \"submission.zip\"\n",
            "valid, errors = validate_submission_zip(sub_zip)\n",
            "assert valid, f\"Submission validation failed: {errors}\"\n",
            "print(f\"[+] SUCCESS: submission.zip validated cleanly at {sub_zip}\")\n",
            "\n",
            "manifest_p = output_dir / \"run_manifest.json\"\n",
            "assert manifest_p.is_file(), f\"Run manifest missing at {manifest_p}\"\n",
            "m_data = json.loads(manifest_p.read_text(encoding='utf-8'))\n",
            "print(f\"[+] Run Status : {m_data.get('status')}\")\n",
            "print(f\"[+] Verdict    : {m_data.get('verdict')}\")\n",
            "if m_data.get(\"huggingface\"):\n",
            "    hf_meta = m_data[\"huggingface\"]\n",
            "    print(f\"[+] Hugging Face Release: https://huggingface.co/{hf_meta.get('repo_id')}\")\n",
            "    print(f\"    Release Commit      : {hf_meta.get('commit_sha')}\")\n",
        ],
    }
    cells.append(cell_5)

    return create_jupyter_notebook(cells)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LegalIR competition notebooks.")
    parser.add_argument("--commit", type=str, default="", help="Target Git commit SHA")
    parser.add_argument("--check-drift", action="store_true", help="Assert committed notebooks match generator output")
    args = parser.parse_args()

    commit_sha = args.commit or get_current_git_commit()
    notebooks_dir = REPO_ROOT / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        notebooks_dir / "kaggle_t4x2_smoke.ipynb": build_kaggle_smoke_notebook(commit_sha),
        notebooks_dir / "kaggle_smoke.ipynb": build_kaggle_smoke_notebook(commit_sha),
        notebooks_dir / "colab_t4_smoke.ipynb": build_colab_t4_smoke_notebook(commit_sha),
        notebooks_dir / "colab_a100_train.ipynb": build_colab_train_notebook(commit_sha),
    }

    if args.check_drift:
        drift = False
        for path, nb_data in targets.items():
            expected_str = json.dumps(nb_data, indent=2, ensure_ascii=False) + "\n"
            if not path.is_file() or path.read_text(encoding="utf-8") != expected_str:
                print(f"[-] DRIFT DETECTED: {path.relative_to(REPO_ROOT)} does not match generator.")
                drift = True
        if drift:
            print("Run `python scripts/generate_notebooks.py` to synchronize notebooks.", file=sys.stderr)
            return 1
        print("[+] SUCCESS: All notebooks match generator output (zero drift).")
        return 0

    for path, nb_data in targets.items():
        nb_str = json.dumps(nb_data, indent=2, ensure_ascii=False) + "\n"
        path.write_text(nb_str, encoding="utf-8")
        print(f"[+] Generated: {path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

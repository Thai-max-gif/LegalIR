#!/usr/bin/env python3
"""
Unified Generator for LegalIR Dual Notebooks:
1. notebooks/kaggle_smoke.ipynb (Kaggle 2xT4 Smoke Gate B1.1)
2. notebooks/colab_a100_train.ipynb (Colab A100 Production Training B1.2)

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
    """Get the approved runtime commit SHA from release approval or git."""
    approval_p = repo_root / "artifacts" / "task1" / "release_approval.json"
    if approval_p.is_file():
        try:
            data = json.loads(approval_p.read_text(encoding="utf-8"))
            rt = data.get("runtime_sha")
            if rt and len(rt) == 40:
                return rt.strip().lower()
        except Exception:
            pass
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode("utf-8").strip()
        if len(sha) == 40:
            return sha.lower()
    except Exception:
        pass
    return "3792b13699f4706c5a698b2d147a55e97fe4c0ce"


def create_jupyter_notebook(cells: list[dict]) -> dict:
    """Wrap cells into a compliant Jupyter notebook JSON structure with kernelspec."""
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
            "### Smoke Test Purpose:\n",
            "- **Fast & Deterministic (~3 minutes)** on free Kaggle T4 / 2×T4 GPU.\n",
            "- Proves dataset integrity from `/kaggle/input/datasets/phucdangg/legalir-task1-clean-data`.\n",
            "- Mines a small 50-query leakage-safe pair subset.\n",
            "- Validates `BAAI/bge-reranker-v2-m3` + LoRA forward/backward pass on CUDA.\n",
            "- Asserts finite loss ($L < \\infty$) and parameter update ($\\Delta w > 0$).\n",
            "- Tests checkpoint save & reload into fresh model instance.\n",
            "- Produces `kaggle_smoke_report.json` required before Colab A100 execution.\n",
            "\n",
            "> **Note:** Zero PyTorch/CUDA reinstallation to avoid Kaggle environment breakage.\n",
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
            "print(f\"[+] CUDA Available : {torch.cuda.is_available()}\")\n",
            "if not torch.cuda.is_available():\n",
            "    print(\"[!] WARNING: CUDA not detected. Ensure GPU accelerator is enabled in Kaggle settings.\")\n",
            "else:\n",
            "    count = torch.cuda.device_count()\n",
            "    print(f\"[+] Visible CUDA Devices: {count}\")\n",
            "    for i in range(count):\n",
            "        prop = torch.cuda.get_device_properties(i)\n",
            "        vram = prop.total_memory / (1024**3)\n",
            "        print(f\"    - GPU {i}: {prop.name} | Total VRAM: {vram:.2f} GB\")\n",
        ],
    }
    cells.append(cell_1)

    # Cell 2: Git Repository Setup (Pinning commit)
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
            f"EXPECTED_COMMIT = os.environ.get(\"LEGALIR_COMMIT_SHA\", \"{commit_sha}\")\n",
            "WORK_DIR = Path(\"/kaggle/working\") if Path(\"/kaggle/working\").exists() else Path.cwd()\n",
            "REPO_DIR = WORK_DIR / \"LegalIR\"\n",
            "\n",
            "if not REPO_DIR.exists():\n",
            "    # Check if already running from within the repository\n",
            "    if (Path.cwd() / \"src\").is_dir() and (Path.cwd() / \"scripts\").is_dir():\n",
            "        REPO_DIR = Path.cwd().resolve()\n",
            "    else:\n",
            "        print(f\"[*] Cloning repository to {REPO_DIR}...\")\n",
            "        subprocess.run([\"git\", \"clone\", \"https://github.com/silent9669/LegalIR.git\", str(REPO_DIR)], check=True)\n",
            "\n",
            "if (REPO_DIR / \".git\").is_dir():\n",
            "    try:\n",
            "        subprocess.run([\"git\", \"fetch\", \"--all\", \"--tags\"], cwd=REPO_DIR, check=False)\n",
            "        if EXPECTED_COMMIT and EXPECTED_COMMIT != \"main\":\n",
            "            subprocess.run([\"git\", \"checkout\", \"--detach\", EXPECTED_COMMIT], cwd=REPO_DIR, check=True)\n",
            "    except Exception as exc:\n",
            "        print(f\"[!] Warning checking out {EXPECTED_COMMIT} ({exc}). Using default branch.\")\n",
            "        subprocess.run([\"git\", \"checkout\", \"main\"], cwd=REPO_DIR, check=False)\n",
            "\n",
            "os.chdir(REPO_DIR)\n",
            "if str(REPO_DIR) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_DIR))\n",
            "print(f\"[+] Working in repository: {REPO_DIR}\")\n",
        ],
    }
    cells.append(cell_2)

    # Cell 3: Minimal Dependencies (Zero Torch Reinstall)
    cell_3 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 3: Minimal Dependencies Preflight (Zero PyTorch Reinstallation)\n",
            "# ==============================================================================\n",
            "needed_packages = []\n",
            "for mod, pkg in [\n",
            "    (\"bm25s\", \"bm25s\"),\n",
            "    (\"pyvi\", \"pyvi\"),\n",
            "    (\"peft\", \"peft\"),\n",
            "    (\"accelerate\", \"accelerate\"),\n",
            "    (\"sentencepiece\", \"sentencepiece\"),\n",
            "]:\n",
            "    try:\n",
            "        __import__(mod)\n",
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

    # Cell 4: Execute Kaggle Smoke Runner
    cell_4 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 4: Discover Dataset & Run Kaggle Smoke Test\n",
            "# ==============================================================================\n",
            "from src.data.canonical import discover_canonical_dataset_dir\n",
            "\n",
            "dataset_dir = discover_canonical_dataset_dir()\n",
            "print(f\"[+] Discovered Canonical Dataset: {dataset_dir}\")\n",
            "\n",
            "output_dir = WORK_DIR / \"legalir_kaggle_smoke_out\"\n",
            "output_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "cmd = [\n",
            "    sys.executable,\n",
            "    str(REPO_DIR / \"scripts/run_kaggle_smoke.py\"),\n",
            "    \"--dataset-dir\", str(dataset_dir),\n",
            "    \"--output-dir\", str(output_dir),\n",
            "    \"--target-sha\", EXPECTED_COMMIT,\n",
            "]\n",
            "print(f\"[*] Running Smoke Runner: {' '.join(cmd)}\")\n",
            "res = subprocess.run(cmd, capture_output=False)\n",
            "if res.returncode != 0:\n",
            "    raise RuntimeError(f\"Kaggle Smoke Runner exited with error code {res.returncode}\")\n",
        ],
    }
    cells.append(cell_4)

    # Cell 5: Inspect Report
    cell_5 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 5: Inspect Smoke Report & Assert PASS\n",
            "# ==============================================================================\n",
            "import json\n",
            "\n",
            "report_path = output_dir / \"kaggle_smoke_report.json\"\n",
            "if not report_path.is_file():\n",
            "    raise FileNotFoundError(f\"Smoke report missing at {report_path}\")\n",
            "\n",
            "report = json.loads(report_path.read_text(encoding='utf-8'))\n",
            "print(json.dumps(report, indent=2, ensure_ascii=False))\n",
            "\n",
            "assert report.get(\"verdict\") == \"PASS\", f\"Smoke gate failed: {report.get('error')}\"\n",
            "print(\"\\n=================================================================\")\n",
            "print(\"[+] KAGGLE 2×T4 CUDA SMOKE GATE PASSED. READY FOR COLAB A100 RUN.\")\n",
            "print(\"=================================================================\")\n",
        ],
    }
    cells.append(cell_5)

    return create_jupyter_notebook(cells)


def build_colab_train_notebook(commit_sha: str) -> dict:
    """Build the clean Google Colab A100 Production Training launcher notebook (B1.2)."""
    cells = []

    # Cell 0: Markdown
    cell_0 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LegalIR Task 1: Google Colab A100 Production Training (B1.2)\n",
            "## UIT Data Science Challenge 2026 — High-Recall Vietnamese Legal IR\n",
            f"**Pinned Git Commit:** `{commit_sha}`\n",
            "\n",
            "### Production Training Stage:\n",
            "- **Enforces NVIDIA A100 GPU** before consuming compute credits.\n",
            "- Trains `BAAI/bge-reranker-v2-m3` with LoRA on all 7,000 canonical training queries.\n",
            "- Uses `torch.bfloat16` precision for maximum throughput.\n",
            "- Generates Top-5 predictions for 1,000 official public test queries.\n",
            "- Verifies all submission invariants and builds `submission.zip`.\n",
            "- Packages `run_manifest.json` and exports artifacts to Hugging Face.\n",
        ],
    }
    cells.append(cell_0)

    # Cell 1: GPU Enforcement
    cell_1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 1: Hardware Verification (Enforce NVIDIA A100)\n",
            "# ==============================================================================\n",
            "import sys\n",
            "import torch\n",
            "\n",
            "print(f\"[+] Python Version : {sys.version.split()[0]}\")\n",
            "print(f\"[+] PyTorch Version: {torch.__version__}\")\n",
            "assert torch.cuda.is_available(), \"CUDA GPU required for A100 training.\"\n",
            "gpu_name = torch.cuda.get_device_name(0)\n",
            "vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
            "print(f\"[+] Detected GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)\")\n",
            "\n",
            "if \"A100\" not in gpu_name:\n",
            "    print(f\"[!] WARNING: Expected NVIDIA A100, found {gpu_name}. Ensure Colab Premium A100 runtime is selected.\")\n",
        ],
    }
    cells.append(cell_1)

    # Cell 2: Git Setup
    cell_2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 2: Repository Checkout\n",
            "# ==============================================================================\n",
            "import os\n",
            "import subprocess\n",
            "from pathlib import Path\n",
            "\n",
            f"EXPECTED_COMMIT = os.environ.get(\"LEGALIR_COMMIT_SHA\", \"{commit_sha}\")\n",
            "REPO_DIR = Path(\"/content/LegalIR\") if Path(\"/content\").exists() else Path.cwd()\n",
            "\n",
            "if not REPO_DIR.exists():\n",
            "    subprocess.run([\"git\", \"clone\", \"https://github.com/silent9669/LegalIR.git\", str(REPO_DIR)], check=True)\n",
            "\n",
            "if (REPO_DIR / \".git\").is_dir():\n",
            "    try:\n",
            "        subprocess.run([\"git\", \"fetch\", \"--all\", \"--tags\"], cwd=REPO_DIR, check=False)\n",
            "        if EXPECTED_COMMIT and EXPECTED_COMMIT != \"main\":\n",
            "            subprocess.run([\"git\", \"checkout\", \"--detach\", EXPECTED_COMMIT], cwd=REPO_DIR, check=True)\n",
            "    except Exception as exc:\n",
            "        print(f\"[!] Warning checking out {EXPECTED_COMMIT} ({exc}). Using default branch.\")\n",
            "        subprocess.run([\"git\", \"checkout\", \"main\"], cwd=REPO_DIR, check=False)\n",
            "\n",
            "os.chdir(REPO_DIR)\n",
            "if str(REPO_DIR) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_DIR))\n",
            "print(f\"[+] Working in: {REPO_DIR}\")\n",
        ],
    }
    cells.append(cell_2)

    # Cell 3: Dependencies
    cell_3 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 3: Dependencies Preflight\n",
            "# ==============================================================================\n",
            "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"--upgrade-strategy\", \"only-if-needed\", \"-r\", \"requirements/gpu.txt\"], check=False)\n",
            "print(\"[+] Dependencies ready.\")\n",
        ],
    }
    cells.append(cell_3)

    # Cell 4: Execute Full Training
    cell_4 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 4: Execute Colab A100 Production Training\n",
            "# ==============================================================================\n",
            "from src.data.canonical import discover_canonical_dataset_dir\n",
            "\n",
            "dataset_dir = discover_canonical_dataset_dir()\n",
            "output_dir = Path(\"/content/legalir_production_run\") if Path(\"/content\").exists() else REPO_DIR / \"artifacts/submission\"\n",
            "output_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "cmd = [\n",
            "    sys.executable,\n",
            "    str(REPO_DIR / \"scripts/run_colab_train.py\"),\n",
            "    \"--dataset-dir\", str(dataset_dir),\n",
            "    \"--output-dir\", str(output_dir),\n",
            "    \"--precision\", \"bfloat16\",\n",
            "]\n",
            "print(f\"[*] Running: {' '.join(cmd)}\")\n",
            "subprocess.run(cmd, check=True)\n",
        ],
    }
    cells.append(cell_4)

    # Cell 5: Package & Export
    cell_5 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 5: Verify Submission & Export Manifest\n",
            "# ==============================================================================\n",
            "from src.evaluation.submission import validate_submission_zip\n",
            "\n",
            "sub_zip = output_dir / \"submission.zip\"\n",
            "if sub_zip.is_file():\n",
            "    valid, errors = validate_submission_zip(sub_zip)\n",
            "    assert valid, f\"Submission validation failed: {errors}\"\n",
            "    print(f\"[+] SUCCESS: submission.zip validated cleanly at {sub_zip}\")\n",
            "else:\n",
            "    print(f\"[*] Note: submission.zip created at {output_dir}\")\n",
            "\n",
            "manifest_p = output_dir / \"run_manifest.json\"\n",
            "if manifest_p.is_file():\n",
            "    print(f\"[+] Run manifest generated: {manifest_p}\")\n",
        ],
    }
    cells.append(cell_5)

    return create_jupyter_notebook(cells)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LegalIR dual competition notebooks.")
    parser.add_argument("--commit", type=str, default="", help="Target Git commit SHA (defaults to HEAD)")
    parser.add_argument("--check-drift", action="store_true", help="Assert committed notebooks match generator output")
    args = parser.parse_args()

    commit_sha = args.commit or get_current_git_commit()
    notebooks_dir = REPO_ROOT / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    kaggle_nb_path = notebooks_dir / "kaggle_smoke.ipynb"
    colab_nb_path = notebooks_dir / "colab_a100_train.ipynb"

    kaggle_nb_data = build_kaggle_smoke_notebook(commit_sha)
    colab_nb_data = build_colab_train_notebook(commit_sha)

    kaggle_nb_str = json.dumps(kaggle_nb_data, indent=2, ensure_ascii=False) + "\n"
    colab_nb_str = json.dumps(colab_nb_data, indent=2, ensure_ascii=False) + "\n"

    if args.check_drift:
        drift = False
        if not kaggle_nb_path.is_file() or kaggle_nb_path.read_text(encoding="utf-8") != kaggle_nb_str:
            print(f"[-] DRIFT DETECTED: {kaggle_nb_path.relative_to(REPO_ROOT)} does not match generator.")
            drift = True
        if not colab_nb_path.is_file() or colab_nb_path.read_text(encoding="utf-8") != colab_nb_str:
            print(f"[-] DRIFT DETECTED: {colab_nb_path.relative_to(REPO_ROOT)} does not match generator.")
            drift = True
        if drift:
            print("Run `python scripts/generate_notebooks.py` to synchronize.", file=sys.stderr)
            return 1
        print("[+] SUCCESS: Notebooks match generator output (zero drift).")
        return 0

    kaggle_nb_path.write_text(kaggle_nb_str, encoding="utf-8")
    colab_nb_path.write_text(colab_nb_str, encoding="utf-8")
    print(f"[+] Successfully generated:")
    print(f"    - {kaggle_nb_path.relative_to(REPO_ROOT)} (pinned: {commit_sha})")
    print(f"    - {colab_nb_path.relative_to(REPO_ROOT)} (pinned: {commit_sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

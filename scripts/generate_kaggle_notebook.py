#!/usr/bin/env python3
"""
Generate production-grade thin Kaggle Final orchestrator notebooks.
Wraps scripts/run_kaggle_final.py.
Zero PyTorch reinstallation on Kaggle.
"""

from __future__ import annotations

import argparse
import json
import re
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
    return "33a0930e7d1ca1ee58000efdc28a79cf7107b9bf"


def build_legalir_notebook(expected_commit: str | None = None) -> dict:
    """Build the clean, thin Kaggle Final production notebook wrapping scripts/run_kaggle_final.py."""
    commit_sha = expected_commit or get_current_git_commit()
    cells = []

    # Cell 0: Markdown - Title, objective, competition constraints
    cell_0 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LegalIR Task 1: High-Recall Vietnamese Legal Information Retrieval\n",
            "## UIT Data Science Challenge 2026 — Kaggle Final Production Runner\n",
            f"**Pinned Runtime Commit:** `{commit_sha}`\n",
            "\n",
            "### Production Pipeline:\n",
            "1. **Verification**: Exact runtime SHA & canonical dataset v2 verification.\n",
            "2. **Immutable Bundle**: Cryptographic & semantic verification of production bundle.\n",
            "3. **Training**: Train one final `BAAI/bge-reranker-v2-m3` LoRA adapter on all 7,000 queries.\n",
            "4. **Public Rerank & Fusion**: Rerank public candidates under frozen fusion winner.\n",
            "5. **Submission**: Strict validation ($1 \\le |answer| \\le 5$, unique official doc IDs) and `submission.zip` packaging.\n",
            "\n",
            "### Invariants:\n",
            "- Learned Parameter Budget: < 4,000,000,000 (4B)\n",
            "- Zero PyTorch reinstallation\n",
            "- Exactly 1,000 public test queries covered\n"
        ]
    }
    cells.append(cell_0)

    # Cell 1: Hardware & Environment Preflight
    cell_1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 1: Hardware & Environment Preflight\n",
            "# ==============================================================================\n",
            "import os\n",
            "import sys\n",
            "import subprocess\n",
            "import torch\n",
            "\n",
            "print(f\"[+] Python Version : {sys.version.split()[0]}\")\n",
            "print(f\"[+] PyTorch Version: {torch.__version__}\")\n",
            "print(f\"[+] CUDA Available : {torch.cuda.is_available()}\")\n",
            "RUN_MODE = os.environ.get(\"LEGALIR_RUN_MODE\", \"full\")  # 'full' or 'smoke'\n",
            "print(f\"[*] Execution Run Mode: {RUN_MODE}\")\n",
            "if torch.cuda.is_available():\n",
            "    for i in range(torch.cuda.device_count()):\n",
            "        prop = torch.cuda.get_device_properties(i)\n",
            "        vram_gb = prop.total_memory / (1024**3)\n",
            "        print(f\"    - GPU {i}: {prop.name} | Total VRAM: {vram_gb:.2f} GB\")\n",
            "else:\n",
            "    print(\"[!] Running on CPU (testing/smoke mode).\")\n"
        ]
    }
    cells.append(cell_1)

    # Cell 2: Repository Bootstrap & Minimal Dependencies Installation (Zero Torch Reinstall)
    cell_2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 2: Repository Bootstrap & Minimal Dependencies (Zero Torch Reinstall)\n",
            "# ==============================================================================\n",
            "from pathlib import Path\n",
            "\n",
            "CWD = Path.cwd()\n",
            "possible_repo_paths = [\n",
            "    CWD,\n",
            "    CWD / \"LegalIR\",\n",
            "    Path(\"/kaggle/working/LegalIR\"),\n",
            "    Path(\"/kaggle/working\"),\n",
            "]\n",
            "\n",
            f"# Pinned runtime: git checkout {commit_sha}\n",
            f"EXPECTED_COMMIT = os.environ.get(\"LEGALIR_COMMIT_SHA\", \"{commit_sha}\")\n",
            "REPO_ROOT = None\n",
            "for p in possible_repo_paths:\n",
            "    if (p / \"scripts\" / \"run_kaggle_final.py\").exists():\n",
            "        REPO_ROOT = p.resolve()\n",
            "        break\n",
            "\n",
            "if REPO_ROOT is None:\n",
            "    print(\"[*] Cloning LegalIR repository into /kaggle/working/LegalIR...\")\n",
            "    target_dir = Path(\"/kaggle/working/LegalIR\")\n",
            "    if not target_dir.exists():\n",
            "        subprocess.run([\"git\", \"clone\", \"https://github.com/silent9669/LegalIR.git\", str(target_dir)], check=True)\n",
            "    if EXPECTED_COMMIT and EXPECTED_COMMIT != \"main\":\n",
            "        subprocess.run([\"git\", \"fetch\", \"--all\"], cwd=target_dir, check=True)\n",
            "        subprocess.run([\"git\", \"checkout\", \"--detach\", EXPECTED_COMMIT], cwd=target_dir, check=True)\n",
            "    REPO_ROOT = target_dir.resolve()\n",
            "\n",
            "actual_commit = subprocess.check_output([\"git\", \"rev-parse\", \"HEAD\"], cwd=REPO_ROOT).decode(\"utf-8\").strip()\n",
            "if EXPECTED_COMMIT and EXPECTED_COMMIT != \"main\" and actual_commit != EXPECTED_COMMIT:\n",
            "    subprocess.run([\"git\", \"fetch\", \"--all\"], cwd=REPO_ROOT, check=True)\n",
            "    subprocess.run([\"git\", \"checkout\", \"--detach\", EXPECTED_COMMIT], cwd=REPO_ROOT, check=True)\n",
            "    actual_commit = subprocess.check_output([\"git\", \"rev-parse\", \"HEAD\"], cwd=REPO_ROOT).decode(\"utf-8\").strip()\n",
            "    if actual_commit != EXPECTED_COMMIT:\n",
            "        raise RuntimeError(f\"Fail-closed commit pin violation: expected {EXPECTED_COMMIT}, got {actual_commit}\")\n",
            "\n",
            "if str(REPO_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_ROOT))\n",
            "\n",
            "# Install minimal dependencies without touching PyTorch\n",
            "required_pkgs = []\n",
            "for mod, pkg in [\n",
            "    (\"lightgbm\", \"lightgbm\"),\n",
            "    (\"sentencepiece\", \"sentencepiece\"),\n",
            "    (\"bm25s\", \"bm25s\"),\n",
            "    (\"pyvi\", \"pyvi\"),\n",
            "    (\"peft\", \"peft\"),\n",
            "    (\"accelerate\", \"accelerate\"),\n",
            "    (\"faiss\", \"faiss-cpu\"),\n",
            "    (\"psutil\", \"psutil\"),\n",
            "]:\n",
            "    try:\n",
            "        __import__(mod)\n",
            "    except ImportError:\n",
            "        required_pkgs.append(pkg)\n",
            "\n",
            "if required_pkgs:\n",
            "    print(f\"[*] Installing missing packages: {required_pkgs}\")\n",
            "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"--no-warn-script-location\"] + required_pkgs, check=True)\n",
            "print(f\"[+] Repositories and dependencies ready on commit {actual_commit}.\")\n"
        ]
    }
    cells.append(cell_2)

    # Cell 3: Execute Kaggle Final Production Runner
    cell_3 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 3: Canonical Dataset Discovery & Preflight Metadata Check (pq.ParquetFile)\n",
            "# ==============================================================================\n",
            "import json\n",
            "import pyarrow.parquet as pq\n",
            "from src.pipeline.kaggle_train import discover_data_dir, discover_public_test_file\n",
            "\n",
            "possible_datasets = [\n",
            "    Path(\"/kaggle/input/task1-canonical-v2\"),\n",
            "    Path(\"/kaggle/input/legalir-task1-clean-data\"),\n",
            "    REPO_ROOT / \"data/task1_canonical_v2\",\n",
            "    Path(\"data/task1_canonical_v2\"),\n",
            "]\n",
            "data_dir = next((p for p in possible_datasets if (p / \"documents.parquet\").exists() or (p / \"public-official.json\").exists()), possible_datasets[0])\n",
            "public_file = discover_public_test_file(repo_root=REPO_ROOT)\n",
            "\n",
            "docs_rows = pq.ParquetFile(data_dir / \"documents.parquet\").metadata.num_rows if (data_dir / \"documents.parquet\").exists() else 0\n",
            "public_data = json.loads(public_file.read_text(encoding=\"utf-8\")) if (public_file and public_file.exists()) else {}\n",
            "public_rows = len(public_data)\n",
            "print(f\"[+] Canonical Data Directory: {data_dir}\")\n",
            "print(f\"[+] Documents Count         : {docs_rows:,} (expected 8,532)\")\n",
            "print(f\"[+] Public Queries Count    : {public_rows:,} (expected 1,000)\")\n",
            "if public_rows != 1000 and RUN_MODE == \"full\":\n",
            "    raise ValueError(f\"Dataset identity mismatch: public queries count is {public_rows}, expected 1,000\")\n",
            "\n",
            "possible_bundles = [\n",
            "    Path(\"/kaggle/input/legalir-production-bundle\"),\n",
            "    Path(\"/kaggle/input/production-bundle\"),\n",
            "    REPO_ROOT / \"artifacts/bundle/production\",\n",
            "    Path(\"artifacts/bundle/production\"),\n",
            "]\n",
            "bundle_dir = next((p for p in possible_bundles if (p / \"production_lock.json\").exists()), possible_bundles[0])\n",
            "\n",
            "output_dir = Path(\"/kaggle/working\") if Path(\"/kaggle/working\").exists() else REPO_ROOT / \"artifacts/submission\"\n",
            "\n",
            "cmd = [\n",
            "    sys.executable,\n",
            "    str(REPO_ROOT / \"scripts/run_kaggle_final.py\"),\n",
            "    \"--dataset-dir\", str(data_dir),\n",
            "    \"--bundle-dir\", str(bundle_dir),\n",
            "    \"--output-dir\", str(output_dir),\n",
            "]\n",
            "if RUN_MODE in (\"smoke\", \"gpu_smoke\", \"mock\"):\n",
            "    cmd.append(\"--mock\")\n",
            "print(f\"[*] Executing Kaggle Final: {' '.join(cmd)}\")\n",
            "subprocess.run(cmd, check=True)\n"
        ]
    }
    cells.append(cell_3)

    # Cell 4: Verify Final Submission Artifact
    cell_4 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# Cell 4: Verify Final Submission Artifact\n",
            "# ==============================================================================\n",
            "zip_candidates = [\n",
            "    Path(\"/kaggle/working/submission.zip\"),\n",
            "    output_dir / \"submission.zip\",\n",
            "]\n",
            "zip_path = next((p for p in zip_candidates if p.is_file()), None)\n",
            "if zip_path is None or not zip_path.is_file():\n",
            "    raise FileNotFoundError(\"submission.zip was not found!\")\n",
            "\n",
            "print(f\"[+] SUCCESS: Valid submission found at {zip_path} ({zip_path.stat().st_size:,} bytes).\")\n",
            "print(\"[+] Ready for official UIT competition submission.\")\n"
        ]
    }
    cells.append(cell_4)

    return {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def generate_and_save_notebooks(
    repo_root: Path | None = None,
    expected_commit: str | None = None,
) -> tuple[Path, Path]:
    """Generate canonical thin competition notebooks with byte-for-byte parity across all surfaces.

    Returns (root_nb, kernel_nb) for test backward compatibility while writing all distributed notebooks.
    """
    root = repo_root or REPO_ROOT
    commit = expected_commit or get_current_git_commit(root)

    nb_dict = build_legalir_notebook(expected_commit=commit)
    nb_content = json.dumps(nb_dict, indent=1, ensure_ascii=False) + "\n"

    target_paths = [
        root / "legalir_training.ipynb",
        root / "kaggle_kernel_task1" / "legalir_training.ipynb",
        root / "kaggle_kernel" / "legalir_training.ipynb",
        root / "kaggle_kernel" / "legalqa_gpu_pipeline.ipynb",
        root / "notebooks" / "kaggle_final.ipynb",
    ]

    for p in target_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(nb_content, encoding="utf-8")

    return target_paths[0], target_paths[1]


def main():
    parser = argparse.ArgumentParser(description="Generate canonical thin Kaggle Final notebooks.")
    parser.add_argument("--commit", type=str, default=None, help="Target git commit SHA to pin")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root path")
    args = parser.parse_args()

    root_nb, kernel_nb = generate_and_save_notebooks(repo_root=args.repo_root, expected_commit=args.commit)
    print(f"[+] Generated canonical notebooks pinned to {args.commit or get_current_git_commit(args.repo_root)}:")
    print(f"    - {root_nb.relative_to(args.repo_root)}")
    print(f"    - {kernel_nb.relative_to(args.repo_root)}")
    print("    - and all distributed mirror surfaces.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate the reproducible Google Colab Single-T4 Contract Smoke Notebooks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COLAB_DIR = REPO_ROOT / "colab"
DEFAULT_NOTEBOOK_PATH = COLAB_DIR / "legalir_t4_smoke.ipynb"
REFRESH_NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "colab_t4_smoke.ipynb"


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


def build_notebook_data(target_commit: str | None = None) -> dict:
    commit_sha = target_commit or get_current_git_commit()
    if len(commit_sha) != 40:
        raise ValueError(f"target_commit must be a 40-character Git commit SHA, got: '{commit_sha}'")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# LegalIR Single-T4 Contract Smoke Gate (Google Colab)\n",
                f"**Pinned Runtime Commit:** `{commit_sha}`\n",
                "\n",
                "This notebook executes the official single-T4 sequential contract smoke on real canonical v2 data using production modules.\n",
                "Prerequisites: Commit must have a **GREEN** `LegalIR CI` GitHub Actions run before execution.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [1] Hardware Preflight: Verify GPU availability and Tesla T4 contract\n",
                "!nvidia-smi\n",
                "\n",
                "import torch\n",
                "assert torch.cuda.is_available(), 'CUDA is not available! Change runtime type to GPU.'\n",
                "gpu_name = torch.cuda.get_device_name(0)\n",
                "print(f'CUDA Device: {gpu_name}')\n",
                "if 'T4' not in gpu_name:\n",
                "    print(f'[!] WARNING: Active GPU {gpu_name} is not a Tesla T4. Smoke will run with non-T4 override.')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [2] Mount Google Drive and retrieve Colab Secrets\n",
                "import os\n",
                "from google.colab import drive, userdata\n",
                "\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# Retrieve HF_TOKEN securely from Colab Secrets (Never print token values)\n",
                "try:\n",
                "    HF_TOKEN = userdata.get('HF_TOKEN')\n",
                "    if HF_TOKEN:\n",
                "        os.environ['HF_TOKEN'] = HF_TOKEN\n",
                "        print('[+] HF_TOKEN loaded securely from Colab Secrets.')\n",
                "    else:\n",
                "        print('[!] Warning: HF_TOKEN not found in Secrets. Some private assets may not download.')\n",
                "except Exception as exc:\n",
                "    print(f'[!] userdata notice: {exc}')\n",
                "\n",
                "# Optional GITHUB_TOKEN for high-rate-limit GitHub CI status checking\n",
                "try:\n",
                "    GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')\n",
                "    if GITHUB_TOKEN:\n",
                "        os.environ['GITHUB_TOKEN'] = GITHUB_TOKEN\n",
                "        print('[+] GITHUB_TOKEN loaded securely from Colab Secrets.')\n",
                "except Exception:\n",
                "    pass\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [3] User Configuration: Target Commit SHA and Official Data Path\n",
                f"TARGET_SHA = '{commit_sha}'\n",
                "DATA_DIR = '/content/drive/MyDrive/legalir-task1-clean-data'\n",
                "OUTPUT_DIR = '/content/drive/MyDrive/legalir-smoke-runs/colab_t4_smoke'\n",
                "\n",
                "print(f'Target SHA : {TARGET_SHA}')\n",
                "print(f'Data Dir   : {DATA_DIR}')\n",
                "print(f'Output Dir : {OUTPUT_DIR}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [4] Clone repository and checkout exact TARGET_SHA in detached HEAD\n",
                "!rm -rf /content/LegalIR\n",
                "!git clone https://github.com/silent9669/LegalIR.git /content/LegalIR\n",
                "%cd /content/LegalIR\n",
                "!git checkout {TARGET_SHA}\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [5] Install minimal Python dependencies without reinstalling torch\n",
                "import subprocess, sys\n",
                "required_pkgs = []\n",
                "for mod, pkg in [('lightgbm', 'lightgbm'), ('sentencepiece', 'sentencepiece'), ('bm25s', 'bm25s'), ('pyvi', 'pyvi'), ('peft', 'peft'), ('accelerate', 'accelerate'), ('faiss', 'faiss-cpu'), ('psutil', 'psutil')]:\n",
                "    try:\n",
                "        __import__(mod)\n",
                "    except ImportError:\n",
                "        required_pkgs.append(pkg)\n",
                "if required_pkgs:\n",
                "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-warn-script-location'] + required_pkgs, check=True)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [6] Verify GitHub CI Status Gate for TARGET_SHA\n",
                "!python scripts/verify_github_ci.py --repo silent9669/LegalIR --sha {TARGET_SHA}\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [7] Execute Official Single-T4 Contract Smoke Gate with all required CLI args\n",
                "!python scripts/run_colab_t4_smoke.py \\\n",
                "    --data-dir '{DATA_DIR}' \\\n",
                "    --work-dir '{OUTPUT_DIR}' \\\n",
                "    --target-sha '{TARGET_SHA}' \\\n",
                "    --allow-non-t4\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [8] Display and inspect Colab Smoke Report\n",
                "import json\n",
                "from pathlib import Path\n",
                "\n",
                "report_file = Path(OUTPUT_DIR) / 'colab_smoke_report.json'\n",
                "if report_file.exists():\n",
                "    report = json.loads(report_file.read_text(encoding='utf-8'))\n",
                "    print(json.dumps(report, indent=2))\n",
                "    print('=' * 65)\n",
                "    print(f'FINAL COLAB SMOKE VERDICT: {report.get(\"result\")}')\n",
                "    print('=' * 65)\n",
                "else:\n",
                "    print(f'[-] Error: Smoke report not found at {report_file}')\n",
            ],
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def generate_colab_notebook(output_path: Path | str = DEFAULT_NOTEBOOK_PATH, target_commit: str | None = None) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nb_data = build_notebook_data(target_commit=target_commit)
    out.write_text(json.dumps(nb_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Colab smoke notebooks.")
    parser.add_argument("--commit", type=str, default=None, help="40-char commit SHA to pin")
    args = parser.parse_args()

    commit_sha = args.commit or get_current_git_commit()
    out1 = generate_colab_notebook(DEFAULT_NOTEBOOK_PATH, target_commit=commit_sha)
    out2 = generate_colab_notebook(REFRESH_NOTEBOOK_PATH, target_commit=commit_sha)
    print(f"[+] Successfully generated Colab smoke notebooks at {out1} and {out2} with commit pin: {commit_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

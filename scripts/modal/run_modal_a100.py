import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

# Define the Modal App
app = modal.App("legalir-a100-production")

# Define the environment image
# Pinned to match requirements-colab.txt (verified: transformers 5.15.1 exists).
# Python 3.11 to match torch cp311 wheels. torch>=2.5 is REQUIRED:
# transformers 5.x lazy-loads model classes only when torch>=2.5 is importable
# (torch 2.1.2 passes raw `import torch` but fails the backend gate with the
# misleading "requires the PyTorch library but it was not found" error).
# PyTorch is installed with CUDA 12.4 support, which is suitable for A100.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers==5.15.1",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "huggingface-hub==1.28.0",
        "numpy>=1.24,<3",
        "pandas>=2,<3",
        "pyarrow>=14",
        "pyyaml>=6,<7",
        "scikit-learn>=1.3,<2",
        "lightgbm>=4,<5",
        "bm25s>=0.2,<1",
        "pyvi>=0.1.1,<1",
        "sentencepiece>=0.1.99",
        "faiss-cpu>=1.7",
        "psutil>=5.9",
        "kaggle>=1.8,<3",
        "tqdm>=4.65"
    )
)

# Persistent volume: without this a 5h timeout kill loses everything
# (5-fold OOF + final LoRA has no checkpoint-resume; trainer saves only at end).
# Pipeline outputs are written directly into a unique attempt directory on the
# Volume from the beginning. Background commits improve durability while the
# process runs, but they do not guarantee every final byte survives a kill:
# the last uncommitted bytes and the current unfinished training loop may be
# lost. There is no checkpoint-resume; a timeout kill still requires a full
# re-run (Volume holds forensics only).
volume = modal.Volume.from_name("legalir-production", create_if_missing=True)
VOLUME_MOUNT = "/root/legalir_volume"

# 5 hours timeout (5 * 60 * 60 = 18000 seconds) to prevent excessive billing.
# Preserved unless a separately reviewed and approved change is made.
# Never automatically relaunch. The timeout caps duration, not spend.
TIMEOUT_SECONDS = 18000

_SHA_RE = re.compile(r"[0-9a-f]{40}")


def create_attempt_dir(volume_root: Path, expected_sha: str) -> Path:
    """Create a unique Volume-backed attempt directory for one run.

    Layout: <volume_root>/<exact-release-sha>/attempts/<uuid4-hex>/.
    A new UUID is used per attempt; prior runs are never overwritten and
    resume is never inferred from old files.
    """
    sha = str(expected_sha or "").strip()
    if not _SHA_RE.fullmatch(sha):
        raise ValueError("Expected an exact lowercase 40-character Git SHA")
    path = Path(volume_root) / sha / "attempts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_launcher_state(attempt_dir: Path, state: dict) -> None:
    """Write sanitized launcher_state.json atomically via temp file + rename.

    Only small private operational metadata is stored: attempt ID, release
    SHA, UTC timestamps, phase, outcome, and exception class. Never exception
    text, token values, or environment dictionaries.
    """
    allowed = {
        "attempt_id": state.get("attempt_id"),
        "expected_sha": state.get("expected_sha"),
        "phase": state.get("phase"),
        "outcome": state.get("outcome"),
        "exception_class": state.get("exception_class"),
        "started_utc": state.get("started_utc"),
        "updated_utc": state.get("updated_utc"),
    }
    tmp = attempt_dir / "launcher_state.json.tmp"
    tmp.write_text(json.dumps(allowed, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, attempt_dir / "launcher_state.json")


def _try_commit_best_effort() -> None:
    """Best-effort intermediate Volume commit; warn and continue on failure."""
    try:
        volume.commit()
    except Exception as exc:  # noqa: BLE001 - durability warning only
        print(f"[!] Intermediate Volume commit failed: {type(exc).__name__}", flush=True)


def _resolve_repo_dir() -> Path:
    # Test hook: LEGALIR_MODAL_REPO_DIR overrides the container checkout path
    # so offline orchestration tests never touch /root. Production default
    # remains /root/LegalIR.
    return Path(os.environ.get("LEGALIR_MODAL_REPO_DIR", "/root/LegalIR"))


def _resolve_dataset_dir() -> Path:
    # Test hook: LEGALIR_MODAL_DATASET_DIR overrides /root/kaggle_dataset.
    return Path(os.environ.get("LEGALIR_MODAL_DATASET_DIR", "/root/kaggle_dataset"))


def _resolve_volume_mount() -> Path:
    # VOLUME_MOUNT is the production constant; tests monkeypatch the module
    # attribute directly. This helper keeps that contract explicit.
    import sys as _sys

    mod = _sys.modules.get(__name__)
    override = getattr(mod, "VOLUME_MOUNT", VOLUME_MOUNT) if mod is not None else VOLUME_MOUNT
    return Path(override)

@app.function(
    image=image,
    gpu="A100",
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_MOUNT: volume},
    secrets=[
        modal.Secret.from_name("kaggle-secret"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
def run_production_training(expected_sha: str, hf_allow_public_repo: bool = False):
    """
    Executes the A100 production training pipeline within a Modal container.
    Pre-A100 hardware gate: Kaggle dual-T4 report (B1.1), strictly enforced.

    Durability: the pipeline's output_dir is a unique attempt directory on the
    mounted Volume from the beginning. Completed adapter files and
    recovery.tar.gz do not depend on a final copy allowlist. On graceful
    return or exception, the Volume is explicitly committed after the wrapper
    has built its archive. Background commits improve durability while the
    process runs, but the last uncommitted bytes and the current unfinished
    training loop may still be lost on a hard kill. No resume is implied.
    """
    import sys

    # Validate SHA before constructing any paths. Fail closed on bad input.
    sha = str(expected_sha or "").strip()
    if not _SHA_RE.fullmatch(sha):
        raise ValueError("Expected an exact lowercase 40-character Git SHA")

    # Unique Volume-backed attempt directory; pipeline writes here directly.
    # training.log is written to this path while training runs by the
    # pipeline's own logger (do not wrap training_log around itself).
    attempt_dir = create_attempt_dir(_resolve_volume_mount(), sha)
    attempt_id = attempt_dir.name
    output_dir = attempt_dir
    started_utc = _utc_now_iso()

    def _update_state(phase: str, outcome: str = "running", exc_class=None) -> None:
        # Best-effort: metadata recording must never mask a primary failure or
        # skip the final persistence attempt. Failures are logged with their
        # class only (no exception text, tokens, or environment).
        try:
            _write_launcher_state(
                attempt_dir,
                {
                    "attempt_id": attempt_id,
                    "expected_sha": sha,
                    "phase": phase,
                    "outcome": outcome,
                    "exception_class": exc_class,
                    "started_utc": started_utc,
                    "updated_utc": _utc_now_iso(),
                },
            )
        except Exception as state_exc:  # noqa: BLE001
            print(
                f"[!] launcher_state write failed: {type(state_exc).__name__} "
                f"(phase={phase} outcome={outcome})",
                flush=True,
            )

    def _final_commit_after_failure(primary_class: str) -> None:
        # A final persistence attempt independent of metadata recording.
        try:
            volume.commit()
            print(f"[*] Committed Volume attempt after failure: {attempt_dir}", flush=True)
        except Exception as commit_exc:  # noqa: BLE001
            # Preserve the original failure; report commit problem separately.
            print(
                f"[!] Volume commit failed: {type(commit_exc).__name__} at {attempt_dir}; "
                f"primary failure was {primary_class}",
                flush=True,
            )

    # Initial metadata before clone/preflight and before expensive work.
    _update_state("checkout")
    _try_commit_best_effort()

    def _run_attempt_body() -> dict:
        # Attempt body: checkout, provenance, HF preflight, dataset, training.
        # Raises on failure; the lifecycle finalization below guarantees
        # best-effort failure metadata plus a final Volume commit for ANY
        # graceful failure across these phases, then re-raises the original
        # exception. State updates never mask the primary failure.
        # 1. Clone the repository and checkout the exact expected SHA
        _update_state("checkout")
        repo_dir = _resolve_repo_dir()
        if not repo_dir.exists():
            print(f"[*] Cloning repository into {repo_dir}...")
            subprocess.run(["git", "clone", "https://github.com/silent9669/LegalIR.git", str(repo_dir)], check=True)

        print(f"[*] Checking out exact commit: {sha}")
        subprocess.run(["git", "fetch", "origin", sha], cwd=repo_dir, check=False)
        res = subprocess.run(["git", "checkout", "--detach", sha], cwd=repo_dir, capture_output=True, text=True)
        if res.returncode != 0:
            print("[*] Checkout fallback: unshallowing repository...")
            subprocess.run(["git", "fetch", "--unshallow", "origin"], cwd=repo_dir, check=False)
            subprocess.run(["git", "checkout", "--detach", sha], cwd=repo_dir, check=True)

        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        os.chdir(repo_dir)

        # 2. CPU provenance gate before any expensive work (Kaggle T4x2 gate).
        from scripts.colab.bootstrap import prepare_dataset, verify_launch

        # We must point to the verified artifact paths in the repo
        kaggle_report = repo_dir / "artifacts/task1/gates/kaggle_t4x2_report.json"
        freeze_file = repo_dir / "artifacts/task1/freeze/production_freeze.json"

        _update_state("provenance")
        print("[*] Verifying production launch constraints (Kaggle T4x2 gate)...")
        verify_launch(sha, kaggle_report, freeze_file)
        _try_commit_best_effort()

        # 3. Preflight Hugging Face Access BEFORE expensive dataset acquisition.
        from scripts.gates.run_a100 import preflight_huggingface_access
        hf_repo = os.environ.get("HF_REPO_ID", "dangphuc2109/legalir-task1-reranker")
        _update_state("hf_preflight")
        print(f"[*] Verifying Hugging Face write access to {hf_repo}...")
        if hf_allow_public_repo:
            print("[!] OPERATOR OVERRIDE: public HF repos permitted for this launch (recorded in manifest).", flush=True)
        hf_ok, hf_detail = preflight_huggingface_access(hf_repo, allow_public_repo=hf_allow_public_repo)
        if not hf_ok:
            raise RuntimeError(f"Hugging Face preflight failed: {hf_detail}")
        print(f"[+] {hf_detail}")
        _try_commit_best_effort()

        # 4. Acquire Kaggle dataset (outside the output tree; pipeline caches may
        # reside within the working dir but stay excluded from HF/recovery by
        # release_files()).
        _update_state("dataset")
        dataset_dir = _resolve_dataset_dir()
        print(f"[*] Downloading and verifying canonical dataset at {dataset_dir}...")
        prepare_dataset(dataset_dir, freeze_file)
        _try_commit_best_effort()

        # 5. Execute the pipeline directly into the Volume-backed attempt dir.
        # Keep existing run_colab_production_training() archive generation; the
        # archive is built inside output_dir and committed below, not via a
        # filtered duplicate snapshot.
        from scripts.run_colab_train import run_colab_production_training

        _update_state("training")
        print(f"[*] Starting A100 production training pipeline at {output_dir}...")
        print("[*] Durable attempt path on Volume; background commits do not guarantee final bytes on kill.", flush=True)
        return run_colab_production_training(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            smoke_report_path=kaggle_report,
            expected_sha=sha,
            precision="bf16",
            allow_non_a100=False,
            mock=False,
            hf_repo=hf_repo,
            freeze_file_path=freeze_file,
            run_mode="full",
            hf_allow_public_repo=hf_allow_public_repo,
        )

    # Lifecycle-wide finalization: any graceful failure after the attempt
    # directory exists records best-effort failure metadata and attempts a
    # final Volume commit before the original exception propagates. Neither
    # step may raise or mask the primary failure.
    try:
        _attempt_report = _run_attempt_body()
    except BaseException as primary_exc:
        primary_class = type(primary_exc).__name__
        _update_state("failed", outcome="failed", exc_class=primary_class)
        _final_commit_after_failure(primary_class)
        raise

    _update_state("completed", outcome="completed")
    try:
        volume.commit()
        print(f"[*] Committed Volume attempt: {attempt_dir}", flush=True)
    except Exception as commit_exc:  # noqa: BLE001
        # Training succeeded but final commit failed: do not return success.
        # Do not erase an already genuine HF receipt; report delivery success
        # and local durability failure separately.
        hf_info = _attempt_report.get("huggingface", {}) if isinstance(_attempt_report, dict) else {}
        hf_ok_flag = bool(hf_info.get("uploaded")) if isinstance(hf_info, dict) else False
        print(
            f"[!] Volume commit failed: {type(commit_exc).__name__} at {attempt_dir}; "
            f"HF delivery success={hf_ok_flag}",
            flush=True,
        )
        raise RuntimeError(
            f"Volume persistence failed ({type(commit_exc).__name__}) at {attempt_dir}; "
            f"training report exists with HF uploaded={hf_ok_flag}. Inspect the Volume path."
        ) from None

    print(f"[+] Training completed. Status: {_attempt_report.get('status')} | Verdict: {_attempt_report.get('verdict')}")

    # The pipeline internally handles Hugging Face artifact upload and cleanup.
    return _attempt_report

@app.local_entrypoint()
def main(hf_allow_public_repo: bool = False):
    import sys
    # Try to grab the SHA from local git if we are in the repo, or from env
    expected_sha = os.environ.get("LEGALIR_COMMIT_SHA")
    if not expected_sha:
        try:
            expected_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            print("Error: Could not determine expected Git SHA. Please set LEGALIR_COMMIT_SHA.", file=sys.stderr)
            sys.exit(1)

    # Defense in depth: repeat CPU provenance validation before .remote().
    # The wrapper script is the recommended entrypoint because invoking Modal
    # can build an image before main executes; local checks alone cannot
    # validate remote secret values or GPU.
    try:
        from scripts.colab.bootstrap import verify_launch as _verify

        _repo = Path(__file__).resolve().parents[2]
        _verify(
            expected_sha,
            _repo / "artifacts/task1/gates/kaggle_t4x2_report.json",
            _repo / "artifacts/task1/freeze/production_freeze.json",
            repo_root=_repo,
        )
    except Exception as exc:
        print(f"[!] Local CPU provenance preflight failed before Modal dispatch: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"[*] Dispatching A100 training job to Modal for commit: {expected_sha}")
    print("[*] This process will run remotely on an A100 GPU and automatically terminate after 5 hours max.")
    print("[*] Durable outputs use /root/legalir_volume/<sha>/attempts/<id>/ on the 'legalir-production' Volume.")
    print("[*] NOTE: 5h caps duration, not spend — retries/re-runs bill extra. No checkpoint-resume:")
    print("    a timeout kill still requires a full re-run (Volume holds forensics only).")
    print("[*] Ensure you have created 'kaggle-secret' and 'huggingface-secret' in the Modal dashboard!")
    if hf_allow_public_repo:
        print("[!] OPERATOR OVERRIDE: public HF repos permitted for this launch.", flush=True)

    result = run_production_training.remote(expected_sha, hf_allow_public_repo=hf_allow_public_repo)
    print(f"[*] Remote job finished with result: {result}")

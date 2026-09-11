from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def hardware_report() -> dict[str, Any]:
    gpus = []
    for idx in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(idx)
        gpus.append({"index": idx, "name": prop.name, "capability": f"{prop.major}.{prop.minor}", "total_memory_bytes": prop.total_memory})
    packages = {}
    for name in ["torch", "transformers", "peft", "accelerate", "pyarrow", "bm25s", "pyvi", "huggingface-hub"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda, "gpu_count": torch.cuda.device_count(), "gpus": gpus, "packages": packages, "disk": {p: shutil.disk_usage(p)._asdict() for p in ["/kaggle/working", "/content", "."] if Path(p).exists()}, "git_sha": git_sha()}


def report_status(checks: list[dict[str, Any]]) -> str:
    statuses = {item.get("status") for item in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "NOT_RUN" in statuses:
        return "NOT_RUN"
    if "SKIPPED" in statuses:
        return "SKIPPED"
    return "PASS"


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(dest)


class NvmlPeakSampler:
    """Samples device memory; it reports observed peak, not a guarantee of every instantaneous peak."""
    def __init__(self, index: int, interval_seconds: float = 0.1):
        self.index, self.interval_seconds, self.peak_bytes = index, interval_seconds, None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        except Exception as exc:
            self.error = repr(exc)
            return
        def sample() -> None:
            while not self._stop.is_set():
                used = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
                self.peak_bytes = max(self.peak_bytes or 0, used)
                self._stop.wait(self.interval_seconds)
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread: self._thread.join(timeout=2)
        if self.error:
            return {"status": "NOT_RUN", "reason": self.error, "sampling_interval_ms": self.interval_seconds * 1000}
        return {"status": "PASS", "observed_peak_used_bytes": self.peak_bytes, "sampling_interval_ms": self.interval_seconds * 1000}

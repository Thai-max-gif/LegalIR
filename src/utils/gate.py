from __future__ import annotations

from typing import Any


def gate_checks(config: dict[str, Any], run: dict[str, Any]) -> list[dict[str, Any]]:
    checks = list(run.get("checks", []))
    training = run.get("training", {})
    if training.get("optimizer_updates", 0) < 5:
        checks.append({"name": "optimizer_updates", "status": "FAIL", "reason": "fewer than 5 completed optimizer updates"})
    if not training.get("parameter_changed"):
        checks.append({"name": "parameter_delta", "status": "FAIL", "reason": "no observed trainable parameter update"})
    vram = run.get("vram", {})
    threshold = config["gate"]["vram_threshold_bytes"]
    for device, peak in vram.get("peak_device_bytes", {}).items():
        checks.append({"name": f"vram_device_{device}", "status": "PASS" if peak <= threshold else "FAIL", "peak_bytes": peak, "threshold_bytes": threshold, "peak_gib": peak / 2**30})
    if config["gate"].get("require_end_to_end") and run.get("retrieval", {}).get("status") != "PASS":
        checks.append({"name": "end_to_end_legalir", "status": "BLOCKED", "reason": "retrieval/fusion/rerank/document aggregation not fully verified"})
    if config["gate"].get("require_approval") and not run.get("approval_record"):
        checks.append({"name": "lead_approval", "status": "BLOCKED", "reason": "no Lead approval record configured"})
    return checks

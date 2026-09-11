from __future__ import annotations

from typing import Any


def logical_parameter_count(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        pointer = parameter.data_ptr()
        if pointer and pointer not in seen:
            seen.add(pointer)
            total += parameter.numel()
    return total


def audit_learned_stack(entries: list[dict[str, Any]], cap: int = 4_000_000_000) -> dict[str, Any]:
    totals = []
    for entry in entries:
        if entry.get("model") is None:
            totals.append({"name": entry.get("name"), "status": "BLOCKED", "reason": "model not loaded; tensor metadata unavailable"})
        else:
            model = entry["model"]
            totals.append({"name": entry["name"], "total_logical_parameters": logical_parameter_count(model), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "revision": entry.get("revision"), "license": entry.get("license")})
    known = sum(x.get("total_logical_parameters", 0) for x in totals)
    return {"models": totals, "known_total_logical_parameters": known, "cap": cap, "under_cap": known < cap if all("total_logical_parameters" in x for x in totals) else None}

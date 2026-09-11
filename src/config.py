from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    config = load_config(path.parent / parent) if parent else {}
    config = _merge(config, raw)
    if overrides:
        config = _merge(config, overrides)
    validate_config(config)
    return config


def validate_config(c: dict[str, Any]) -> None:
    required = [("data", "dataset_root"), ("model", "reranker_id"), ("experiment", "output_dir")]
    for group, key in required:
        if not c.get(group, {}).get(key):
            raise ValueError(f"Missing required config: {group}.{key}")
    if c["model"].get("num_labels") != 1:
        raise ValueError("LegalIR reranker must use num_labels=1")
    if c["model"].get("max_length", 0) > 512:
        raise ValueError("max_length must be <= 512 total tokens")
    if c["lora"].get("r") != 16 or c["lora"].get("alpha") != 32:
        raise ValueError("LoRA r=16 and alpha=32 are required by the approved profile")
    if c["training"].get("loss_type") not in {"auto", "pairwise", "bce"}:
        raise ValueError("training.loss_type must be auto, pairwise, or bce")
    if c["training"].get("precision") not in {"fp16", "bf16", "fp32"}:
        raise ValueError("Unsupported precision")
    if c["retrieval"].get("document_aggregation") not in {"max", "mean"}:
        raise ValueError("retrieval.document_aggregation must be max or mean")
    if c["training"].get("eval_steps", 0) <= 0 or c["training"].get("save_steps", 0) <= 0:
        raise ValueError("eval_steps and save_steps must be positive")


def canonical_yaml(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, allow_unicode=True, sort_keys=True)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_yaml(config).encode("utf-8")).hexdigest()


def write_resolved_config(config: dict[str, Any], out_dir: str | Path) -> Path:
    dest = Path(out_dir) / "resolved_config.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(canonical_yaml(config), encoding="utf-8")
    return dest


def dotted_override(items: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value: {item}")
        path, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        target = result
        parts = path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result

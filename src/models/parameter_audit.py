"""LegalIR Parameter Budget Auditor & Compliance Validator (<4B Parameters).

Official Competition Rule:
The total learned parameter count across all models used by the Task 1 system must be
strictly below 4,000,000,000 parameters (< 4B).
LoRA, quantization, 8-bit/4-bit loading, pruning, and other memory optimizations do NOT
reduce the model's parameter count for competition-rule purposes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import yaml

MAX_PARAMETER_BUDGET: int = 4_000_000_000  # 4B strict upper bound

# Pinned exact parameter counts for offline preflight / zero-download environments
KNOWN_PARAM_COUNTS: dict[str, int] = {
    "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2": 134_998_272,
    "BAAI/bge-reranker-v2-m3": 567_755_777,
    "BAAI/bge-m3": 567_756_802,
    "Qwen/Qwen3-Embedding-0.6B": 594_000_000,
    "Qwen/Qwen2.5-Coder-0.5B": 494_032_896,
    "bert-base-uncased": 109_482_240,
    "roberta-base": 124_644_864,
    "xlm-roberta-base": 278_043_648,
    "xlm-roberta-large": 559_890_432,
}

DEFAULT_PIPELINE_MODELS: list[dict[str, str]] = [
    {
        "name": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        "role": "dense_embedding",
    },
    {
        "name": "BAAI/bge-reranker-v2-m3",
        "role": "cross_encoder_reranker",
    },
]


class ParameterBudgetExceededError(ValueError):
    """Raised when the total system parameter count reaches or exceeds the 4B limit."""

    pass


def estimate_transformer_parameters(config_or_dict: Any) -> int:
    """Analytically estimate parameter count from a transformer config or dictionary.

    Used as an offline fallback when meta-device instantiation is unavailable.
    """
    if hasattr(config_or_dict, "to_dict"):
        d = config_or_dict.to_dict()
    elif isinstance(config_or_dict, dict):
        d = config_or_dict
    else:
        d = {}

    hidden_size = d.get("hidden_size") or d.get("d_model", 768)
    vocab_size = d.get("vocab_size", 30522)
    num_layers = d.get("num_hidden_layers") or d.get("n_layer") or d.get("num_layers", 12)
    num_heads = d.get("num_attention_heads") or d.get("n_head", 12)
    num_kv_heads = d.get("num_key_value_heads", num_heads)
    inter_size = d.get("intermediate_size") or d.get("ffn_dim") or (4 * hidden_size)
    max_pos = d.get("max_position_embeddings") or d.get("n_positions", 512)
    type_vocab = d.get("type_vocab_size", 0)
    model_type = str(d.get("model_type", "")).lower()
    head_dim = d.get("head_dim", hidden_size // max(1, num_heads))
    hidden_act = str(d.get("hidden_act", "")).lower()
    is_gated = "silu" in hidden_act or "swiglu" in hidden_act or "llama" in model_type or "qwen" in model_type

    # 1. Embeddings
    emb = vocab_size * hidden_size
    if "rope" not in model_type and "llama" not in model_type and "qwen" not in model_type:
        emb += max_pos * hidden_size
    if type_vocab > 0:
        emb += type_vocab * hidden_size
    emb += 2 * hidden_size  # Embeddings LayerNorm

    # 2. Transformer layers
    q_proj = hidden_size * (num_heads * head_dim) + (hidden_size if "qwen" not in model_type else 0)
    k_proj = hidden_size * (num_kv_heads * head_dim) + (hidden_size if "qwen" not in model_type else 0)
    v_proj = hidden_size * (num_kv_heads * head_dim) + (hidden_size if "qwen" not in model_type else 0)
    out_proj = (num_heads * head_dim) * hidden_size + hidden_size
    attn_ln = 2 * hidden_size

    if is_gated:
        ffn = 3 * hidden_size * inter_size
    else:
        ffn = 2 * hidden_size * inter_size + inter_size + hidden_size
    ffn_ln = 2 * hidden_size

    layer_params = q_proj + k_proj + v_proj + out_proj + attn_ln + ffn + ffn_ln
    total = emb + num_layers * layer_params + 2 * hidden_size

    # Optional pooler / classifier
    if "bert" in model_type:
        total += hidden_size * hidden_size + hidden_size
    num_labels = d.get("num_labels", 0)
    if num_labels and num_labels > 0:
        total += hidden_size * num_labels + num_labels

    return int(total)


def count_parameters_from_config(
    config_or_name: Any,
    offline_fallback: bool = True,
) -> int:
    """Calculate exact parameter count from a Hugging Face PretrainedConfig, dict, or model name.

    Uses meta-device instantiation (zero memory/zero download) with fallback to known counts
    and analytical architecture estimation.
    """
    model_name_str = None
    if isinstance(config_or_name, (str, Path)):
        model_name_str = str(config_or_name)
        if model_name_str in KNOWN_PARAM_COUNTS:
            return KNOWN_PARAM_COUNTS[model_name_str]

        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name_str)
        except Exception:
            if offline_fallback and model_name_str in KNOWN_PARAM_COUNTS:
                return KNOWN_PARAM_COUNTS[model_name_str]
            if offline_fallback and Path(model_name_str).is_dir():
                cfg_path = Path(model_name_str) / "config.json"
                if cfg_path.exists():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        return estimate_transformer_parameters(json.load(f))
            if offline_fallback:
                # Fail-closed: silent 0 would undercount the <4B budget.
                # Unknown model names must be pinned in KNOWN_PARAM_COUNTS.
                raise ValueError(
                    f"Unknown model '{model_name_str}' has no pinned parameter count; "
                    f"add it to KNOWN_PARAM_COUNTS to keep the <4B audit fail-closed."
                )
            raise
    else:
        config = config_or_name

    # Try meta-device instantiation via transformers
    if config is not None:
        try:
            from transformers import AutoModel, AutoModelForSequenceClassification, PretrainedConfig

            if isinstance(config, dict):
                from transformers import PretrainedConfig
                config_obj = PretrainedConfig.from_dict(config)
            else:
                config_obj = config

            arch = getattr(config_obj, "architectures", []) or []
            is_seq_cls = any(
                "SequenceClassification" in a or "Reranker" in a or "ForSequenceClassification" in a
                for a in arch
            )
            with torch.device("meta"):
                if is_seq_cls:
                    model = AutoModelForSequenceClassification.from_config(config_obj)
                else:
                    model = AutoModel.from_config(config_obj)
                return sum(p.numel() for p in model.parameters())
        except Exception:
            # Fall back to analytical estimation
            return estimate_transformer_parameters(config)

    if model_name_str and model_name_str in KNOWN_PARAM_COUNTS:
        return KNOWN_PARAM_COUNTS[model_name_str]

    return 0


def count_parameters(
    model_or_config: Any,
    trainable_only: bool = False,
    offline_fallback: bool = True,
) -> int:
    """Accurately count learned parameters for PyTorch models, PEFT models, HF configs, state dicts, or model names.

    For PEFT / LoRA models:
    Returns the total base model parameters + adapter parameters (not just trainable LoRA params),
    ensuring competition compliance (LoRA does not reduce base architecture parameter budget).
    """
    if model_or_config is None:
        return 0

    # 1. PyTorch Module (including PEFT models)
    if isinstance(model_or_config, nn.Module):
        if trainable_only:
            return sum(p.numel() for p in model_or_config.parameters() if p.requires_grad)
        # Note: model.parameters() in a PeftModel includes both base model params (requires_grad=False)
        # and adapter params (requires_grad=True). This correctly sums the full base architecture.
        return sum(p.numel() for p in model_or_config.parameters())

    # 2. State dict or tensor dictionary
    if isinstance(model_or_config, dict):
        # Check if it's a state_dict of tensors
        if any(isinstance(v, torch.Tensor) for v in model_or_config.values()):
            return sum(v.numel() for v in model_or_config.values() if isinstance(v, torch.Tensor))
        # Otherwise treat as config dict
        return count_parameters_from_config(model_or_config, offline_fallback=offline_fallback)

    # 3. Hugging Face PretrainedConfig or model name string/Path
    if hasattr(model_or_config, "to_dict") or isinstance(model_or_config, (str, Path)):
        return count_parameters_from_config(model_or_config, offline_fallback=offline_fallback)

    # 4. Fallback object with parameters()
    if hasattr(model_or_config, "parameters") and callable(model_or_config.parameters):
        params = list(model_or_config.parameters())
        if trainable_only:
            return sum(p.numel() for p in params if getattr(p, "requires_grad", True))
        return sum(p.numel() for p in params)

    return 0


def audit_model_parameters(
    model_or_config: Any,
    name: str | None = None,
    role: str | None = None,
    offline_fallback: bool = True,
) -> dict[str, Any]:
    """Generate a detailed parameter report for a single model component."""
    inferred_name = name or str(getattr(model_or_config, "name_or_path", getattr(model_or_config, "__class__.__name__", "model")))
    if isinstance(model_or_config, (str, Path)):
        inferred_name = str(model_or_config)

    total_p = count_parameters(model_or_config, trainable_only=False, offline_fallback=offline_fallback)
    if inferred_name in KNOWN_PARAM_COUNTS and total_p < KNOWN_PARAM_COUNTS[inferred_name]:
        total_p = KNOWN_PARAM_COUNTS[inferred_name]

    trainable_p = count_parameters(model_or_config, trainable_only=True, offline_fallback=offline_fallback) if isinstance(model_or_config, nn.Module) else total_p

    is_peft = False
    base_p = total_p
    adapter_p = 0

    if isinstance(model_or_config, nn.Module):
        # Strong PEFT / LoRA detection (avoid marking plain HF models as PEFT just because base_model or mixins exist)
        has_peft_config = bool(getattr(model_or_config, "peft_config", None))
        has_lora_params = any("lora_" in n.lower() for n, _ in model_or_config.named_parameters())
        class_name = model_or_config.__class__.__name__
        is_peft_class = "PeftModel" in class_name or getattr(model_or_config, "is_peft_model", False)

        if has_peft_config or has_lora_params or is_peft_class:
            is_peft = True
            lora_p = sum(p.numel() for n, p in model_or_config.named_parameters() if "lora_" in n.lower() or (has_peft_config and p.requires_grad))
            adapter_p = lora_p
            base_p = max(0, total_p - adapter_p)
            trainable_p = lora_p

    return {
        "model_name": inferred_name,
        "role": role or "unknown",
        "parameters": total_p,
        "parameters_billions": round(total_p / 1e9, 6),
        "trainable_parameters": trainable_p,
        "is_peft_lora": is_peft,
        "base_parameters": base_p,
        "adapter_parameters": adapter_p,
    }


def extract_models_from_config(config_path_or_dict: str | Path | dict[str, Any]) -> list[dict[str, str]]:
    """Extract model definitions and roles from a YAML configuration file or dict."""
    if isinstance(config_path_or_dict, (str, Path)):
        path = Path(config_path_or_dict)
        if not path.exists():
            return list(DEFAULT_PIPELINE_MODELS)
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = config_path_or_dict or {}

    extracted: list[dict[str, str]] = []
    seen = set()

    # 1. Pipeline format (configs/pipeline.yaml)
    dense_cfg = cfg.get("retrieval", {}).get("dense_macro", {})
    dense_name = dense_cfg.get("model_name")
    if dense_name and dense_name not in seen:
        extracted.append({"name": str(dense_name), "role": "dense_embedding"})
        seen.add(dense_name)

    reranker_cfg = cfg.get("ranking", {}).get("reranker", {})
    if reranker_cfg.get("enabled", True):
        reranker_name = reranker_cfg.get("model_name")
        if reranker_name and reranker_name not in seen:
            extracted.append({"name": str(reranker_name), "role": "cross_encoder_reranker"})
            seen.add(reranker_name)

    # 2. Models format (configs/models.yaml)
    emb_cfg = cfg.get("embedding", {})
    emb_name = emb_cfg.get("name")
    if emb_name and emb_name not in seen:
        extracted.append({"name": str(emb_name), "role": "dense_embedding"})
        seen.add(emb_name)

    rrk_cfg = cfg.get("reranker", {})
    rrk_name = rrk_cfg.get("name")
    if rrk_name and rrk_name not in seen:
        extracted.append({"name": str(rrk_name), "role": "cross_encoder_reranker"})
        seen.add(rrk_name)

    if not extracted:
        return list(DEFAULT_PIPELINE_MODELS)

    return extracted


def audit_system_parameters(
    models: Sequence[Any] | dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    output_json: str | Path | None = "parameter_audit.json",
    raise_on_violation: bool = True,
    offline_fallback: bool = True,
    require_loaded_models: bool = False,
) -> dict[str, Any]:
    """Audit the total learned parameters of the LegalIR system and verify compliance (<4B budget).

    Args:
        models: List of model objects, configs, model names, or dictionary mapping name -> model.
        config_path: Path to pipeline.yaml/models.yaml to extract models from if models is None.
        output_json: Path to save parameter_audit.json (set None to skip file writing).
        raise_on_violation: If True, raises ParameterBudgetExceededError when total >= 4B.
        offline_fallback: If True, uses pinned known counts and offline config estimation when needed.
        require_loaded_models: If True, strictly requires instantiated PyTorch modules and active adapters.

    Returns:
        Structured audit report dictionary with per-model breakdown and compliance status.
    """
    model_entries: list[tuple[Any, str, str]] = []

    if models is not None:
        if isinstance(models, dict):
            for name, m in models.items():
                model_entries.append((m, str(name), "model"))
        else:
            for item in models:
                if isinstance(item, dict) and "name" in item:
                    model_entries.append((item["name"], item["name"], item.get("role", "model")))
                elif isinstance(item, tuple) and len(item) >= 2:
                    model_entries.append((item[0], str(item[1]), item[2] if len(item) > 2 else "model"))
                else:
                    name_str = str(getattr(item, "name_or_path", getattr(item, "__class__.__name__", str(item))))
                    model_entries.append((item, name_str, "model"))
    else:
        # Load from config or default models
        configs_to_check = [config_path] if config_path else ["configs/pipeline.yaml", "configs/models.yaml"]
        extracted = []
        for cp in configs_to_check:
            if cp and Path(cp).exists():
                extracted = extract_models_from_config(cp)
                if extracted:
                    break
        if not extracted:
            extracted = list(DEFAULT_PIPELINE_MODELS)

        for entry in extracted:
            model_entries.append((entry["name"], entry["name"], entry.get("role", "model")))

    if require_loaded_models:
        for model_obj, name, role in model_entries:
            if not isinstance(model_obj, nn.Module):
                raise RuntimeError(
                    f"Strict parameter audit requires loaded nn.Module, but component '{name}' ({role}) is {type(model_obj)}"
                )

    model_reports: dict[str, Any] = {}
    total_params = 0

    for model_obj, name, role in model_entries:
        report = audit_model_parameters(
            model_or_config=model_obj,
            name=name,
            role=role,
            offline_fallback=offline_fallback,
        )
        model_reports[name] = report
        total_params += report["parameters"]

    is_compliant = total_params < MAX_PARAMETER_BUDGET
    verdict = "PASS" if is_compliant else "FAIL"
    utilization_pct = round((total_params / MAX_PARAMETER_BUDGET) * 100, 4)
    headroom = MAX_PARAMETER_BUDGET - total_params

    audit_report = {
        "is_compliant": is_compliant,
        "verdict": verdict,
        "total_learned_parameters": total_params,
        "total_parameters_billions": round(total_params / 1e9, 6),
        "budget_limit": MAX_PARAMETER_BUDGET,
        "budget_limit_billions": 4.0,
        "budget_utilization_pct": utilization_pct,
        "headroom_parameters": headroom,
        "headroom_billions": round(headroom / 1e9, 6),
        "models": model_reports,
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rule_compliance": {
            "max_allowed_budget": MAX_PARAMETER_BUDGET,
            "passed": is_compliant,
            "rule": "Total learned parameters across all models in Task 1 system must be < 4,000,000,000 (<4B)",
        },
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2, ensure_ascii=False)

    if raise_on_violation and not is_compliant:
        raise ParameterBudgetExceededError(
            f"Total parameter count {total_params:,} ({total_params / 1e9:.4f}B) reaches or exceeds "
            f"the strict maximum competition budget of {MAX_PARAMETER_BUDGET:,} (4.0B). System is non-compliant!"
        )

    return audit_report


def validate_parameter_budget(
    models: Sequence[Any] | None = None,
    total_params: int | None = None,
    raise_on_violation: bool = True,
) -> bool:
    """Validate whether total parameters are strictly below 4B budget limit."""
    if total_params is None:
        report = audit_system_parameters(models=models, output_json=None, raise_on_violation=raise_on_violation)
        return bool(report["is_compliant"])

    is_compliant = total_params < MAX_PARAMETER_BUDGET
    if raise_on_violation and not is_compliant:
        raise ParameterBudgetExceededError(
            f"Total parameter count {total_params:,} reaches or exceeds "
            f"the strict maximum competition budget of {MAX_PARAMETER_BUDGET:,} (4.0B)."
        )
    return is_compliant

from __future__ import annotations

from typing import Any

import torch


def load_reranker_with_lora(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    m = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(m["reranker_id"], revision=m.get("reranker_revision"), use_fast=True)
    tokenizer.padding_side = "right"
    kwargs: dict[str, Any] = {"num_labels": 1, "revision": m.get("reranker_revision")}
    if config["training"]["precision"] == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif config["training"]["precision"] == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    if config["training"].get("attn_implementation") in {"eager", "sdpa"}:
        kwargs["attn_implementation"] = config["training"]["attn_implementation"]
    model = AutoModelForSequenceClassification.from_pretrained(m["reranker_id"], **kwargs)
    if config["training"].get("gradient_checkpointing"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    requested = set(config["lora"]["target_modules"])
    actual = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in requested and not name.startswith("classifier.")]
    if not actual:
        raise RuntimeError(f"No non-classifier linear LoRA targets matched {sorted(requested)}")
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"], lora_dropout=config["lora"]["dropout"], bias="none", target_modules=actual, modules_to_save=config["lora"]["modules_to_save"])
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    audit = {"lora_target_modules": actual, "lora_target_count": len(actual), "trainable_parameters": trainable, "total_parameters": sum(p.numel() for p in model.parameters())}
    return model, tokenizer, audit


def reload_adapter(adapter_path: str, base_model_id: str, revision: str | None, device: str):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification
    base = AutoModelForSequenceClassification.from_pretrained(base_model_id, revision=revision, num_labels=1)
    return PeftModel.from_pretrained(base, adapter_path).to(device).eval()

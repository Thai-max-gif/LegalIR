from __future__ import annotations

from typing import Any


class FrozenDenseEncoder:
    def __init__(self, model_id: str, revision: str | None, device: str):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, texts: list[str], **kwargs: Any):
        raise RuntimeError("Dense encoding is intentionally blocked until index manifest supplies verified pooling/preprocessing and compatible index format")

import re

from pathlib import Path
from src.models.bootstrap import MODEL_REGISTRY, required_model_files
from src.models.device import resolve_device


def test_required_files_exclude_onnx_and_openvino():
    files = required_model_files("BAAI/bge-m3")
    assert "pytorch_model.bin" in files or "*.safetensors" in files
    assert not any(name.startswith("onnx/") for name in files)
    assert not any(name.startswith("openvino/") for name in files)


def test_dek21_v2_registry_is_pinned_for_local_bootstrap():
    model_name = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    metadata = MODEL_REGISTRY[model_name]

    assert re.fullmatch(r"[0-9a-f]{40}", metadata["revision"])
    assert model_name in MODEL_REGISTRY
    assert required_model_files(model_name) == metadata["allow_patterns"]


def test_dek21_allowlist_contains_phobert_tokenizer_files():
    files = required_model_files("CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2")

    assert "bpe.codes" in files
    assert "*.json" in files
    assert "tokenizer*" in files


def test_explicit_cpu_is_preserved():
    assert resolve_device("cpu") == "cpu"


def test_auto_device_resolves():
    dev = resolve_device("auto")
    assert dev in {"mps", "cuda", "cpu"}

"""Regression: adapter reload must preserve the training base-model revision.

Training resolves an immutable BGE revision, but evaluation/final inference
previously reloaded the adapter base without passing that revision.
"""
import json
from pathlib import Path
from unittest.mock import patch

from src.models.bootstrap import MODEL_REGISTRY
from src.ranking.reranker import CrossEncoderReranker

PINNED = MODEL_REGISTRY["BAAI/bge-reranker-v2-m3"]["revision"]


def _make_adapter_dir(tmp_path: Path, *, base_model="BAAI/bge-reranker-v2-m3", revision=PINNED) -> Path:
    d = tmp_path / "adapter"
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model}), encoding="utf-8"
    )
    manifest: dict = {"base_model": base_model, "status": "completed"}
    if revision is not None:
        manifest["base_model_revision"] = revision
    (d / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"fake-weights")
    return d


def _install_hf_mocks(monkeypatch, calls: dict):
    import transformers

    class _DummyTokenizer:
        model_max_length = 512

        def __call__(self, *a, **k):
            raise AssertionError("tokenizer should not be invoked in reload test")

    class _DummyConfig:
        max_position_embeddings = 512
        model_type = "xlm-roberta"
        pad_token_id = 1

    class _DummyModel:
        config = _DummyConfig()

        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_tok_from_pretrained(name_or_path, **kwargs):
        calls.setdefault("tokenizer", []).append((str(name_or_path), dict(kwargs)))
        if str(name_or_path).endswith("adapter") or Path(str(name_or_path)).is_dir():
            # Simulate: adapter dir has no tokenizer files, force fallback.
            raise OSError("no tokenizer in adapter dir")
        return _DummyTokenizer()

    def fake_model_from_pretrained(name_or_path, **kwargs):
        calls.setdefault("model", []).append((str(name_or_path), dict(kwargs)))
        return _DummyModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tok_from_pretrained)
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification, "from_pretrained", fake_model_from_pretrained
    )

    import peft

    def fake_peft_from_pretrained(base_model, adapter_path, **kwargs):
        calls.setdefault("peft", []).append((str(adapter_path), dict(kwargs)))
        return base_model

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", fake_peft_from_pretrained)


def test_adapter_reload_uses_manifest_revision(tmp_path, monkeypatch):
    adapter_dir = _make_adapter_dir(tmp_path)
    calls: dict = {}
    _install_hf_mocks(monkeypatch, calls)

    r = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        adapter_path=adapter_dir,
        device="cpu",
    )
    r.ensure_loaded()

    assert calls["model"], "base model must be loaded"
    base_source, base_kwargs = calls["model"][0]
    assert base_source == "BAAI/bge-reranker-v2-m3"
    assert base_kwargs.get("revision") == PINNED, f"got {base_kwargs}"

    # Tokenizer fallback (adapter dir has no tokenizer) must use same revision.
    assert len(calls["tokenizer"]) == 2
    _, fallback_kwargs = calls["tokenizer"][1]
    assert fallback_kwargs.get("revision") == PINNED


def test_explicit_revision_overrides_manifest(tmp_path, monkeypatch):
    adapter_dir = _make_adapter_dir(tmp_path, revision="0" * 40)
    calls: dict = {}
    _install_hf_mocks(monkeypatch, calls)

    r = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        adapter_path=adapter_dir,
        device="cpu",
        revision=PINNED,
    )
    r.ensure_loaded()

    _, base_kwargs = calls["model"][0]
    assert base_kwargs.get("revision") == PINNED


def test_registry_fallback_when_manifest_lacks_revision(tmp_path, monkeypatch):
    adapter_dir = _make_adapter_dir(tmp_path, revision=None)
    calls: dict = {}
    _install_hf_mocks(monkeypatch, calls)

    r = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        adapter_path=adapter_dir,
        device="cpu",
    )
    r.ensure_loaded()

    _, base_kwargs = calls["model"][0]
    assert base_kwargs.get("revision") == PINNED


def test_direct_model_load_pins_revision(monkeypatch):
    calls: dict = {}
    _install_hf_mocks(monkeypatch, calls)

    r = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3", device="cpu")
    # Force non-adapter path with no local model_path.
    r.model_path = None
    r.ensure_loaded()

    _, base_kwargs = calls["model"][0]
    assert base_kwargs.get("revision") == PINNED


def test_mock_never_passes_revision():
    r = CrossEncoderReranker(model_name="mock", device="cpu")
    assert r._resolve_base_revision("mock", PINNED) is None
    # Local dirs never use revision.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        assert r._resolve_base_revision(td, PINNED) is None


def test_train_reranker_records_portable_revision(tmp_path, monkeypatch):
    import pandas as pd

    pairs = tmp_path / "pairs.parquet"
    pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "query_text": ["q", "q"],
            "doc_id": ["dA", "dB"],
            "label": [1.0, 0.0],
            "evidence_text": ["eA", "eB"],
        }
    ).to_parquet(pairs)

    # Avoid real HF downloads: stub tokenizer/model constructors.
    import transformers

    class _Tok:
        def save_pretrained(self, d):
            Path(d).mkdir(parents=True, exist_ok=True)

    class _Cfg:
        max_position_embeddings = 512
        model_type = "bert"
        pad_token_id = 0

    class _Model:
        config = _Cfg()

        def parameters(self):
            import torch

            m = torch.nn.Linear(4, 1)
            return m.parameters()

        def named_modules(self):
            import torch

            m = torch.nn.Linear(4, 1)
            return m.named_modules()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok())
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *a, **k: _Model(),
    )

    # Stub the heavy trainer: record that training ran, write minimal manifest.
    import importlib
    import sys
    tr_mod = sys.modules.get("src.training.train_reranker")
    if tr_mod is None:
        tr_mod = importlib.import_module("src.training.train_reranker")
    # When src.training.__init__ shadows the submodule with the function,
    # fall back to the file-backed module spec.
    if not hasattr(tr_mod, "RerankerTrainer"):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_train_reranker_mod",
            str(Path("src/training/train_reranker.py").resolve()),
        )
        assert spec and spec.loader
        tr_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tr_mod)

    class _StubTrainer:
        def __init__(self, *a, **k):
            self.batch_size = 2
            self.gradient_accumulation_steps = 8

        def train(self, output_dir=None):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "adapter_model.safetensors").write_bytes(b"w")
            return {"status": "completed", "global_steps": 1}

    monkeypatch.setattr(tr_mod, "RerankerTrainer", _StubTrainer)

    out_dir = tmp_path / "out"
    report = tr_mod.train_reranker(
        pairs_file=pairs,
        output_dir=out_dir,
        config_path={"base_model_name": "BAAI/bge-reranker-v2-m3", "batch_size": 2,
                     "gradient_accumulation_steps": 8, "max_steps": 1},
        base_model_name="BAAI/bge-reranker-v2-m3",
        enforce_full_coverage_steps=False,
    )
    assert report["base_model"] == "BAAI/bge-reranker-v2-m3"
    assert report["base_model_revision"] == PINNED
    assert "artifacts" not in report["base_model"]
    manifest = json.loads((out_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_model_revision"] == PINNED

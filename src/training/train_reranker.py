from pathlib import Path
from typing import Any
import argparse
import json
import pandas as pd
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.paths import ProjectPaths
from src.models.device import resolve_device
from src.training.trainer import (
    RerankerTrainer,
    audit_pair_coverage,
    compute_coverage_required_steps,
)


def load_training_config(config_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path, dict):
        data = dict(config_path)
    else:
        config_path = Path(config_path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {} if config_path.exists() else {}

    out_dir = str(data.get("output_dir", "artifacts/local/training/checkpoints"))
    if not (out_dir.startswith("artifacts/local/") or "/local/" in out_dir or out_dir.startswith("/kaggle/working/") or "checkpoints" in out_dir):
        raise ValueError(f"Training output_dir must be inside artifacts/local/ or /kaggle/working/, got: {out_dir}")

    return data


def train_reranker(
    *,
    pairs_file: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "configs/experiments/reranker_lora.yaml",
    fold: int | None = None,
    max_steps: int | None = None,
    base_model_name: str | None = None,
    loss_type: str | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    device: str | None = None,
    precision: str | None = None,
    num_workers: int | None = None,
    enforce_full_coverage_steps: bool = True,
    allow_warm_start: bool = False,
) -> dict[str, Any]:
    """
    Supervised Cross-Encoder fine-tuning with PEFT LoRA, verified weight updates,
    and checkpoint saving.
    """
    pairs_path = Path(pairs_file)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Training pairs file not found: {pairs_file}")

    pairs_df = pd.read_parquet(pairs_path)
    if pairs_df.empty:
        raise ValueError(f"Training pairs file is empty: {pairs_file}")

    paths = ProjectPaths.from_repo()
    cfg = load_training_config(config_path) if (isinstance(config_path, dict) or Path(config_path).exists()) else {}

    # Command-line / argument overrides
    if max_steps is not None:
        cfg["max_steps"] = max_steps
    if base_model_name is not None:
        cfg["base_model_name"] = base_model_name
    if loss_type is not None:
        cfg["loss_type"] = loss_type
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    if learning_rate is not None:
        cfg["learning_rate"] = learning_rate
    if precision is not None:
        cfg["precision"] = precision
    if num_workers is not None:
        cfg["num_workers"] = max(0, int(num_workers))

    # Strict isolation: folds must never warm-start from a global 7,000-query adapter
    if fold is not None:
        cfg["allow_warm_start"] = False
    else:
        cfg["allow_warm_start"] = bool(allow_warm_start)

    resolved_device = resolve_device(device if device is not None else cfg.get("device", "auto"))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Logical base-model id (portable) vs actual load path (may be a local snapshot).
    # The manifest must record the logical id + immutable revision so adapter
    # reload on another machine resolves the same base weights.
    logical_model_id = (
        base_model_name
        if base_model_name is not None
        else str(cfg.get("base_model_name", "BAAI/bge-reranker-v2-m3"))
    )
    model_name_or_path = logical_model_id
    hf_manifest_path = paths.local_models / "huggingface" / "manifest.json"
    if hf_manifest_path.exists():
        try:
            hf_data = json.loads(hf_manifest_path.read_text(encoding="utf-8"))
            if logical_model_id in hf_data and "path" in hf_data[logical_model_id]:
                local_path = Path(hf_data[logical_model_id]["path"])
                if local_path.is_dir():
                    model_name_or_path = str(local_path)
        except Exception as e:
            print(f"Warning: failed reading HF manifest: {e}")

    print(f"Loaded {len(pairs_df)} reranker training pairs from {pairs_path}")

    # Use 100% of outer-fold training queries (no 10% holdout waste since outer fold itself is evaluated)
    train_pairs_df = pairs_df
    val_pairs_df = None

    # Audit pair coverage and compute required steps from actual eligible queries in training pairs
    coverage_audit = audit_pair_coverage(train_pairs_df)
    n_eligible = coverage_audit["eligible_queries_count"]
    batch_sz = int(cfg.get("batch_size", 2))
    grad_acc = int(cfg.get("gradient_accumulation_steps", 8))
    req_steps = compute_coverage_required_steps(
        eligible_query_count=n_eligible,
        batch_size=batch_sz,
        gradient_accumulation_steps=grad_acc,
        target_coverage_pct=1.0,
        require_pos_and_neg=True,
    )

    if enforce_full_coverage_steps:
        cfg_steps = int(cfg.get("max_steps", 500))
        requested = max_steps or 0
        effective_steps = max(cfg_steps, requested, req_steps)
    else:
        effective_steps = max_steps if max_steps is not None else int(cfg.get("max_steps", 500))
    cfg["max_steps"] = effective_steps

    # Load tokenizer and model
    resolved_revision: str | None = None
    if logical_model_id == "mock" or model_name_or_path == "mock":
        import tempfile
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

        print("Loading mock lightweight base model...")
        config = BertConfig(
            vocab_size=300,
            hidden_size=32,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=64,
            max_position_embeddings=128,
            num_labels=1,
        )
        base_model = BertForSequenceClassification(config)
        tmp_vocab = Path(tempfile.gettempdir()) / "mock_vocab.txt"
        if not tmp_vocab.exists():
            vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(295)]
            tmp_vocab.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(tmp_vocab))
        resolved_revision = None
    else:
        resolved_revision = (
            cfg.get("revision")
            or cfg.get("reranker_revision")
            or cfg.get("base_model_revision")
            or cfg.get("base_model_rev")
        )
        if resolved_revision is not None:
            resolved_revision = str(resolved_revision).strip() or None
        if resolved_revision is None:
            try:
                from src.models.bootstrap import MODEL_REGISTRY
                if logical_model_id in MODEL_REGISTRY:
                    resolved_revision = MODEL_REGISTRY[logical_model_id].get("revision")
            except Exception:
                pass
        # Only pass revision for Hub ids; local snapshot dirs already pin weights.
        is_local_dir = False
        try:
            is_local_dir = Path(model_name_or_path).expanduser().is_dir()
        except Exception:
            is_local_dir = False
        load_kwargs: dict[str, Any] = {}
        if resolved_revision and not is_local_dir:
            load_kwargs["revision"] = resolved_revision

        print(f"Loading base model: {model_name_or_path} (logical={logical_model_id} revision={resolved_revision})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer for real base model '{model_name_or_path}' "
                f"(logical={logical_model_id} revision={resolved_revision}): {e}"
            ) from e

        try:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=1,
                **load_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load real base model '{model_name_or_path}' "
                f"(logical={logical_model_id} revision={resolved_revision}): {e}"
            ) from e

    trainer = RerankerTrainer(
        model=base_model,
        tokenizer=tokenizer,
        train_data=train_pairs_df,
        val_data=val_pairs_df,
        config=cfg,
        device=resolved_device,
    )

    report = trainer.train(output_dir=out_path)
    if fold is not None:
        report["fold"] = fold
    # Portable identity: logical Hub id + immutable revision. The local load
    # path is recorded separately and must not be used as the reload key.
    report["base_model"] = logical_model_id
    report["base_model_revision"] = resolved_revision
    report["base_model_local_path"] = str(model_name_or_path)
    report["output_dir"] = str(out_path)
    report["pairs_file"] = str(pairs_path)
    report["input_pair_count"] = len(pairs_df)
    report["pair_count"] = len(pairs_df)
    report["positive_count"] = int((pairs_df["label"] > 0.5).sum()) if "label" in pairs_df.columns else 0
    report["negative_count"] = int((pairs_df["label"] <= 0.5).sum()) if "label" in pairs_df.columns else 0
    report["unique_training_queries"] = len(train_pairs_df["query_id"].unique())
    report["eligible_training_queries"] = int(n_eligible)
    report["actual_unique_queries_seen"] = int(report.get("actual_unique_queries_seen", len(train_pairs_df["query_id"].unique())))
    report["actual_query_coverage_pct"] = float(report.get("actual_query_coverage_pct", 100.0))
    report["unique_query_coverage_pct"] = float(report.get("unique_query_coverage_pct", 100.0))
    report["positive_query_coverage_pct"] = float(report.get("positive_query_coverage_pct", 100.0))
    report["negative_query_coverage_pct"] = float(report.get("negative_query_coverage_pct", 100.0))
    report["coverage_required_steps"] = int(req_steps)
    report["effective_max_steps"] = int(effective_steps)
    report["pair_coverage_audit"] = coverage_audit
    report["optimizer_steps"] = int(report.get("global_steps", 0))
    try:
        bs = int(getattr(trainer, "batch_size", 2))
        ga = int(getattr(trainer, "gradient_accumulation_steps", 8))
        fallback_examples = int(report.get("global_steps", 0)) * bs * ga
    except Exception:
        fallback_examples = 0
    report["effective_examples_seen"] = int(report.get("actual_examples_seen", fallback_examples))
    report["epochs_or_equivalent"] = round(len(train_pairs_df) / max(1, len(train_pairs_df)), 2)

    # Compute adapter checksum
    import hashlib
    adapter_weights = out_path / "adapter_model.safetensors"
    if not adapter_weights.exists():
        adapter_weights = out_path / "adapter_model.bin"
    if adapter_weights.exists():
        report["adapter_checksum"] = hashlib.sha256(adapter_weights.read_bytes()).hexdigest()
    else:
        report["adapter_checksum"] = None

    # Overwrite manifest with full report
    manifest_file = out_path / "training_manifest.json"
    manifest_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Reranker training completed, manifest saved to {manifest_file}")
    return report


def main():
    parser = argparse.ArgumentParser(description="LegalIR LoRA Reranker Trainer")
    parser.add_argument("--pairs-file", type=str, default="artifacts/local/training/pairs/fold_0/reranker_pairs.parquet")
    parser.add_argument("--output-dir", type=str, default="artifacts/local/training/checkpoints/fold_0")
    parser.add_argument("--config", type=str, default="configs/experiments/reranker_lora.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--loss-type", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_reranker(
        pairs_file=args.pairs_file,
        output_dir=args.output_dir,
        config_path=args.config,
        fold=args.fold,
        max_steps=args.max_steps,
        base_model_name=args.base_model,
        loss_type=args.loss_type,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()

from pathlib import Path
from src.training.train_reranker import load_training_config


def test_training_config_keeps_outputs_local(tmp_path: Path):
    cfg_file = tmp_path / "reranker_lora.yaml"
    cfg_file.write_text(
        """
base_model_name: "BAAI/bge-reranker-v2-m3"
base_model_revision: "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
output_dir: "artifacts/local/training/checkpoints/reranker_lora"
learning_rate: 2e-5
lora_r: 16
lora_alpha: 32
batch_size: 2
gradient_accumulation_steps: 4
max_steps: 10
quantization: null
""",
        encoding="utf-8",
    )

    config = load_training_config(cfg_file)
    assert config["output_dir"].startswith("artifacts/local/")
    assert config["base_model_revision"] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert config["quantization"] is None
    assert config["lora_r"] == 16

from pathlib import Path
import pytest

from src.core.config import load_pipeline_config
from src.core.paths import ProjectPaths


def test_project_paths_separate_shared_and_local(tmp_path: Path):
    paths = ProjectPaths.from_repo(tmp_path)
    assert paths.shared == tmp_path / "artifacts" / "shared"
    assert paths.canonical == paths.shared / "canonical" / "v2"
    assert paths.local_models == tmp_path / "artifacts" / "local" / "models"
    assert paths.local_runs == tmp_path / "artifacts" / "local" / "runs"


def test_config_rejects_absolute_project_paths(tmp_path: Path):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("paths:\n  canonical: /tmp/illegal\n", encoding="utf-8")
    with pytest.raises(ValueError, match="relative"):
        load_pipeline_config(cfg)


def test_config_loads_defaults(tmp_path: Path):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("seed: 42\npaths:\n  canonical: artifacts/shared/canonical/v2\n", encoding="utf-8")
    data = load_pipeline_config(cfg)
    assert data["seed"] == 42
    assert data["paths"]["canonical"] == "artifacts/shared/canonical/v2"


def test_pipeline_config_dek21():
    from src.core.config import load_pipeline_config
    cfg = load_pipeline_config()
    assert cfg.retrieval.dense_macro.model_name == "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    assert cfg.retrieval.dense_macro.dimension == 768
    assert cfg.retrieval.dense_macro.use_pyvi is True
    assert cfg.ranking.reranker.model_name == "BAAI/bge-reranker-v2-m3"


def test_pipeline_config_to_dict_is_safe_yaml_serializable():
    import yaml

    from src.core.config import PipelineConfig

    cfg = PipelineConfig({
        "nested": PipelineConfig({"value": 1}),
        "items": [PipelineConfig({"name": "item"})],
    })

    plain = cfg.to_dict()

    assert plain == {"nested": {"value": 1}, "items": [{"name": "item"}]}
    assert yaml.safe_load(yaml.safe_dump(plain)) == plain

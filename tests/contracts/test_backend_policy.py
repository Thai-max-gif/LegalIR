import pytest
from src.pipeline.kaggle_train import run_kaggle_pipeline


def test_backend_policy_rejects_kaggle_full(tmp_path):
    """Kaggle backend must fail closed on FULL mode per QUALITY_RUNTIME_PLAN.md Section 1."""
    with pytest.raises(RuntimeError, match="Backend policy violation: Kaggle backend is restricted"):
        run_kaggle_pipeline(
            working_dir=tmp_path / "work",
            run_mode="full",
            backend="kaggle",
        )


def test_backend_policy_allows_colab_or_modal_full(monkeypatch, tmp_path):
    """Modal and Colab are permitted production backends."""
    # Mocking discover_data_dir to ensure the policy check passes before data loading
    def fake_discover(*args, **kwargs):
        raise StopIteration("Policy passed")

    monkeypatch.setattr("src.pipeline.kaggle_train.discover_data_dir", fake_discover)

    with pytest.raises(StopIteration, match="Policy passed"):
        run_kaggle_pipeline(
            working_dir=tmp_path / "work",
            run_mode="full",
            backend="colab",
        )

    with pytest.raises(StopIteration, match="Policy passed"):
        run_kaggle_pipeline(
            working_dir=tmp_path / "work",
            run_mode="full",
            backend="modal",
        )

from pathlib import Path
import pytest
from src.artifacts.cli import plan_cleanup, apply_cleanup


def test_cleanup_deletes_verified_duplicate_but_preserves_unknown_file(tmp_path: Path):
    keep = tmp_path / "artifacts" / "shared" / "canonical" / "v2" / "documents.parquet"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_bytes(b"canonical v2 content")

    duplicate = tmp_path / "data" / "task1_canonical" / "v1" / "documents.parquet"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    duplicate.write_bytes(b"legacy content")

    unknown = tmp_path / "my_custom_notes.txt"
    unknown.write_text("important user notes", encoding="utf-8")

    actions = plan_cleanup(tmp_path)
    action_paths = [a["path"] for a in actions]

    assert any(str(duplicate).startswith(a) for a in action_paths)
    assert str(unknown) not in action_paths

    with pytest.raises(ValueError, match="confirmation token"):
        apply_cleanup(actions, confirmation_token="")

    deleted = apply_cleanup(actions, confirmation_token="CONFIRM_CLEANUP")
    assert not duplicate.exists()
    assert keep.exists()
    assert unknown.exists()

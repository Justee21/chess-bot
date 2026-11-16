import hashlib
import json

import pytest

from chessbot.frozen import read_config, verify_artifacts


def test_verify_artifacts_detects_changed_checkpoint(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"original weights")
    (tmp_path / "table.json").write_bytes(b"{}")
    cfg = {"checkpoint": "model.pt", "table": "table.json",
           "checkpoint_sha256": hashlib.sha256(b"original weights").hexdigest(),
           "table_sha256": hashlib.sha256(b"{}").hexdigest()}
    verify_artifacts(cfg, tmp_path)
    (tmp_path / "model.pt").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_artifacts(cfg, tmp_path)


def test_missing_artifact_is_an_error_not_a_base_model_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_artifacts"):
        verify_artifacts({"checkpoint": "missing.pt"}, tmp_path)


def test_config_paths_resolve_from_repository_not_working_directory(tmp_path):
    path = tmp_path / "config" / "final.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"rules": {"resign": False, "offer_or_accept_draws": False, "search": False}}))
    _, root = read_config(path)
    assert root == tmp_path


def test_enabling_search_in_config_is_rejected(tmp_path):
    path = tmp_path / "final.json"
    path.write_text(json.dumps({"rules": {"resign": False, "offer_or_accept_draws": False, "search": True}}))
    with pytest.raises(ValueError, match="unsupported policy rules"):
        read_config(path)

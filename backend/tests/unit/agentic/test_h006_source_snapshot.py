import hashlib
import json

import pytest

from ml.agentic.training.train_sft import validate_source_snapshot_manifest


def manifest(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    (repo / "code.py").write_text("value = 1\n", encoding="utf-8")
    (data / "train.jsonl").write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": "h006-source-snapshot.v1", "status": "frozen_internal_run_input",
        "origin_git_commit": "a" * 40, "promotion_eligible": False,
        "online_replacement_authorized": False,
        "code_files": {"code.py": hashlib.sha256((repo / "code.py").read_bytes()).hexdigest()},
        "dataset_dir": str(data),
        "dataset_files": {"train.jsonl": hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest()},
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return repo, data, path


def test_source_snapshot_detects_code_and_data_changes(tmp_path):
    repo, data, path = manifest(tmp_path)
    assert validate_source_snapshot_manifest(path, repo)["code_files"] == 1
    (repo / "code.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="code.py"):
        validate_source_snapshot_manifest(path, repo)
    repo, data, path = manifest(tmp_path / "fresh")
    (data / "train.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train.jsonl"):
        validate_source_snapshot_manifest(path, repo)


def test_source_snapshot_rejects_path_escape(tmp_path):
    repo, _, path = manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["code_files"] = {"../outside.py": "0" * 64}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        validate_source_snapshot_manifest(path, repo)


def test_source_snapshot_accepts_generic_internal_sft_schema(tmp_path):
    repo, _, path = manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "internal-sft-source-snapshot.v1"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_source_snapshot_manifest(path, repo)["code_files"] == 1

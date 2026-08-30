from scripts.check_staged_secrets import is_text_candidate, scan_paths, scan_text


def test_scan_text_redacts_openai_compatible_key():
    token = "sk-proj-" + "abcdefghijklmnopqrstuvwxyz123456"

    findings = scan_text(".env.production", f"DEEPSEEK_API_KEY={token}\n")

    assert findings == [
        {
            "path": ".env.production",
            "line": 1,
            "rule": "openai_compatible_api_key",
            "match": "[REDACTED]",
        }
    ]
    assert token not in str(findings)


def test_scan_text_allows_documented_placeholders():
    assert scan_text(".env.example", "OPENAI_API_KEY=sk-your-key-here\n") == []


def test_env_variants_and_source_files_are_scanned():
    assert is_text_candidate(".env.production") is True
    assert is_text_candidate("backend/settings.py") is True
    assert is_text_candidate("weights/model.safetensors") is False


def test_scan_paths_uses_requested_tracked_file_list(tmp_path):
    token = "sk-proj-" + "abcdefghijklmnopqrstuvwxyz123456"
    tracked = tmp_path / "tracked.py"
    tracked.write_text(f'API_KEY = "{token}"\n', encoding="utf-8")

    report = scan_paths(tmp_path, ["tracked.py"], source="tracked")

    assert report["scanned_files"] == 1
    assert report["passed"] is False
    assert token not in str(report)

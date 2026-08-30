from pathlib import Path

from scripts.build_stage26_evidence_freeze import (
    FROZEN_BASELINE,
    parse_history_scan,
    scan_secrets,
    summarize_git_porcelain,
    validate_baseline,
)


def test_validate_baseline_requires_stage24_and_stage25_to_match():
    stage24 = {
        "schema_version": "travel-agent-stage24-final-evaluation.v1",
        "status": "passed",
        "headline": dict(FROZEN_BASELINE),
    }
    stage25 = {
        "schema_version": "travel-agent-stage25-showcase.v1",
        "status": "ready",
        "headline_metrics": {
            "strict_tool_decisions": FROZEN_BASELINE["strict_success"],
            "multi_turn_tasks": FROZEN_BASELINE["rollout_success"],
            "mean_reward": FROZEN_BASELINE["mean_reward"],
            "teacher_task_share": FROZEN_BASELINE["teacher_call_share"],
            "token_reduction_vs_all_teacher_percent": FROZEN_BASELINE[
                "token_reduction_vs_all_teacher_percent"
            ],
            "model_latency_reduction_vs_all_teacher_percent": FROZEN_BASELINE[
                "latency_reduction_vs_all_teacher_percent"
            ],
        },
    }

    assert validate_baseline(stage24, stage25)["passed"] is True
    stage25["headline_metrics"]["mean_reward"] = 1.0
    assert validate_baseline(stage24, stage25)["passed"] is False


def test_git_summary_never_persists_paths():
    result = summarize_git_porcelain(" M secret.env\nM  source.py\n?? private.txt\n")

    assert result["clean"] is False
    assert result["entries"] == 3
    assert result["index_changes"] == 1
    assert result["worktree_changes"] == 1
    assert result["untracked"] == 1
    assert "secret.env" not in str(result)


def test_secret_scan_redacts_values(tmp_path: Path):
    leaked = tmp_path / "settings.env"
    token = "sk-proj-" + "abcdefghijklmnopqrstuvwxyz123456"
    leaked.write_text(f"OPENAI_API_KEY={token}\n", encoding="utf-8")

    result = scan_secrets(tmp_path, [leaked])

    assert result["passed"] is False
    assert result["findings"] == [
        {
            "path": "settings.env",
            "line": 1,
            "rule": "openai_api_key",
            "match": "[REDACTED]",
        }
    ]
    assert token not in str(result)


def test_history_scan_only_retains_commits_and_paths():
    commits, paths = parse_history_scan(
        "commit:abc123\n.env.example\n\ncommit:def456\n.env.example\nconfig/dev.env\n"
    )

    assert commits == ["abc123", "def456"]
    assert paths == [".env.example", "config/dev.env"]

import json

import pytest

from scripts.build_stage24_final_report import (
    _pct_delta,
    summarize_http_benchmark,
    summarize_rollout_candidates,
)


def test_summarize_http_benchmark_checks_runs_and_families(tmp_path):
    report = {
        "schema_version": "vllm-http-benchmark.v1",
        "model": "student",
        "cases": 2,
        "repetitions": 1,
        "concurrency": 2,
        "summary": {
            "runs": 2,
            "successful_runs": 1,
            "action_mismatches": 1,
            "argument_mismatches": 0,
            "http_errors": 0,
            "request_throughput_per_second": 4.0,
            "inference": {
                "completion_tokens": {"mean": 5.0},
                "request_latency_ms": {"mean": 10.0, "p50": 9.0, "p95": 12.0},
                "ttft_ms": {"mean": 3.0},
                "tpot_ms": {"mean": 1.0},
            },
        },
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    rows = [
        {"case_id": "a", "family": "search", "success": True},
        {"case_id": "b", "family": "tradeoff", "success": False},
    ]
    (tmp_path / "runs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    result = summarize_http_benchmark(tmp_path)

    assert result["success_rate"] == 0.5
    assert result["family_success"]["search"]["success_rate"] == 1.0
    assert result["family_success"]["tradeoff"]["success_rate"] == 0.0
    assert len(result["source"]["runs_sha256"]) == 64


def test_summarize_rollout_candidates_aggregates_model_only_metrics(tmp_path):
    manifest = {
        "model": "student",
        "candidate_rollouts": 2,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        {
            "task_id": "a",
            "family": "search",
            "score": {
                "successful": True,
                "episode_reward": 1.0,
                "policy_steps": 2,
                "completion_tokens": 10,
                "request_latency_ms": 100.0,
            },
        },
        {
            "task_id": "b",
            "family": "tradeoff",
            "score": {
                "successful": False,
                "episode_reward": 0.0,
                "policy_steps": 1,
                "completion_tokens": 5,
                "request_latency_ms": 50.0,
            },
        },
    ]
    (tmp_path / "teacher_candidates.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    result = summarize_rollout_candidates(tmp_path)

    assert result["success_rate"] == 0.5
    assert result["mean_episode_reward"] == 0.5
    assert result["policy_steps"] == 3
    assert result["completion_tokens"] == 15
    assert result["mean_episode_request_latency_ms"] == 75.0


def test_pct_delta_is_signed_and_rejects_zero_reference():
    assert _pct_delta(90, 100) == -10.0
    assert _pct_delta(110, 100) == 10.0
    with pytest.raises(ValueError, match="non-zero"):
        _pct_delta(1, 0)

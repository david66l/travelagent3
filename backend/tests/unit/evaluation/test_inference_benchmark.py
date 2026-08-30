from evaluation.inference_benchmark import (
    StreamAccumulator,
    VLLMBenchmarkCase,
    build_chat_payload,
    compare_curriculum_profiles,
    summarize_benchmark_runs,
)


def test_stream_accumulator_extracts_ttft_usage_and_tool_call():
    accumulator = StreamAccumulator()
    accumulator.consume(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "search_", "arguments": '{"key'},
                            }
                        ]
                    }
                }
            ]
        },
        elapsed_ms=40.0,
    )
    accumulator.consume(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "pois", "arguments": 'words":[]}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 6,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        },
        elapsed_ms=90.0,
    )
    metrics = accumulator.inference_metrics(
        model="Qwen3-8B", total_latency_ms=100.0, thinking_enabled=False
    )

    assert accumulator.tool_names == {0: "search_pois"}
    assert accumulator.tool_arguments == {0: '{"keywords":[]}'}
    assert metrics.ttft_ms == 40.0
    assert metrics.tpot_ms == 12.0
    assert metrics.cached_prompt_tokens == 80


def test_payload_hard_disables_thinking_and_requires_one_tool():
    case = VLLMBenchmarkCase(
        case_id="search-1",
        messages=[{"role": "user", "content": "search"}],
        tools=[{"type": "function", "function": {"name": "search_pois"}}],
        allowed_actions=["search_pois"],
    )
    payload = build_chat_payload(
        case,
        model="Qwen3-8B",
        max_tokens=128,
        temperature=0.1,
        thinking_enabled=False,
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False


def test_benchmark_summary_keeps_correctness_and_latency_together():
    metrics = {
        "model": "Qwen3-8B",
        "backend": "vllm-http",
        "thinking_mode": "disabled",
        "streaming": True,
        "completion_tokens": 4,
        "request_latency_ms": 80,
        "ttft_ms": 20,
        "tpot_ms": 20,
    }
    report = summarize_benchmark_runs(
        [
            {"success": True, "inference_metrics": metrics},
            {
                "success": False,
                "action_mismatch": True,
                "argument_mismatch": True,
                "inference_metrics": metrics,
            },
        ]
    )

    assert report["runs"] == 2
    assert report["successful_runs"] == 1
    assert report["action_mismatches"] == 1
    assert report["argument_mismatches"] == 1
    assert report["reasoning_runs"] == 0
    assert report["inference"]["ttft_ms"]["measured_requests"] == 2


def test_benchmark_summary_separates_frozen_label_contract_conflicts():
    metrics = {
        "model": "student",
        "backend": "vllm-http",
        "thinking_mode": "disabled",
        "streaming": True,
        "completion_tokens": 4,
        "request_latency_ms": 80,
    }
    report = summarize_benchmark_runs(
        [
            {
                "success": False,
                "policy_contract_success": True,
                "label_contract_conflict": True,
                "inference_metrics": metrics,
            },
            {
                "success": True,
                "policy_contract_success": True,
                "label_contract_conflict": False,
                "inference_metrics": metrics,
            },
        ]
    )

    assert report["successful_runs"] == 1
    assert report["label_contract_conflicts"] == 1
    assert report["contract_consistent_runs"] == 1
    assert report["contract_consistent_successful_runs"] == 1
    assert report["policy_contract_successful_runs"] == 2


def test_profile_comparison_separates_round_length_and_compute_signals():
    def report(*, requests, completion_tokens, request_ms, rollout_ms):
        return {
            "behavior_gate": {"successful_rollouts": 8, "rollouts": 8},
            "rollout_latency": {"mean_ms": rollout_ms, "p95_ms": rollout_ms},
            "inference_metrics": {
                "requests": requests,
                "prompt_tokens": {"total": 1000},
                "completion_tokens": {"total": completion_tokens},
                "request_latency_ms": {"mean": request_ms, "p95": request_ms},
            },
        }

    comparison = compare_curriculum_profiles(
        report(requests=10, completion_tokens=100, request_ms=100, rollout_ms=200),
        report(requests=12, completion_tokens=150, request_ms=180, rollout_ms=350),
    )

    assert comparison["changes_percent"]["requests"] == 20.0
    assert comparison["changes_percent"]["completion_tokens"] == 50.0
    assert set(comparison["diagnostic_signals"]) == {
        "MORE_POLICY_ROUNDS",
        "MORE_COMPLETION_TOKENS",
        "SLOWER_REQUEST_COMPUTE_PROXY",
    }

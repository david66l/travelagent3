from core.inference_metrics import InferenceMetrics, percentile, summarize_inference_metrics


def test_percentile_uses_linear_interpolation():
    assert percentile([100.0, 200.0, 300.0], 0.50) == 200.0
    assert percentile([100.0, 200.0], 0.95) == 195.0


def test_summary_keeps_missing_ttft_explicit():
    report = summarize_inference_metrics(
        [
            InferenceMetrics(
                model="Qwen3-8B",
                backend="transformers",
                thinking_mode="disabled",
                prompt_tokens=100,
                completion_tokens=10,
                request_latency_ms=200,
            ),
            InferenceMetrics(
                model="Qwen3-8B",
                backend="vllm",
                thinking_mode="disabled",
                streaming=True,
                prompt_tokens=120,
                completion_tokens=8,
                cached_prompt_tokens=80,
                request_latency_ms=120,
                ttft_ms=40,
                tpot_ms=10,
            ),
        ]
    )

    assert report["requests"] == 2
    assert report["completion_tokens"]["total"] == 18
    assert report["cached_prompt_tokens"]["total"] == 80
    assert report["ttft_ms"]["measured_requests"] == 1
    assert report["thinking_modes"] == ["disabled"]

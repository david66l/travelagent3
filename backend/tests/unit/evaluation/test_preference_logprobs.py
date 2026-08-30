import json

from scripts.evaluate_preference_logprobs import _load_pairs, summarize_preference_scores


def test_preference_logprob_summary_reports_accuracy_margin_and_family():
    rows = [
        {
            "family": "search",
            "mean_logprob_margin": 0.4,
            "chosen_mean_logprob": -0.1,
            "rejected_mean_logprob": -0.5,
            "chosen_sum_logprob": -1.0,
            "rejected_sum_logprob": -2.0,
        },
        {
            "family": "search",
            "mean_logprob_margin": -0.2,
            "chosen_mean_logprob": -0.6,
            "rejected_mean_logprob": -0.4,
            "chosen_sum_logprob": -3.0,
            "rejected_sum_logprob": -2.0,
        },
        {
            "family": "tradeoff",
            "mean_logprob_margin": 0.3,
            "chosen_mean_logprob": -0.2,
            "rejected_mean_logprob": -0.5,
            "chosen_sum_logprob": -1.0,
            "rejected_sum_logprob": -3.0,
        },
    ]

    result = summarize_preference_scores(rows)

    assert result["overall"]["chosen_preferred"] == 2
    assert result["overall"]["preference_accuracy"] == 0.666667
    assert result["overall"]["mean_logprob_margin"] == 0.166667
    assert result["overall"]["sequence_preference_accuracy"] == 0.666667
    assert result["overall"]["mean_sequence_logprob_margin"] == 0.666667
    assert result["families"]["tradeoff"]["preference_accuracy"] == 1.0


def test_preference_loader_is_standalone_and_validates_required_fields(tmp_path):
    pair = {
        "pair_id": "pair-1",
        "family": "single_action_abort",
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "request"},
        ],
        "tools": [{"type": "function", "function": {"name": "abort"}}],
        "chosen": {"role": "assistant", "tool_calls": []},
        "rejected": {"role": "assistant", "tool_calls": []},
    }
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(pair) + "\n", encoding="utf-8")

    assert _load_pairs(path) == [pair]

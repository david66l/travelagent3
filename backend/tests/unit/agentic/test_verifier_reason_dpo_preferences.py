from agentic.reason_quality import (
    build_grounded_repair_reason,
    repair_reason_rationale_prefixes,
    verifier_reason_quality_checks,
)
from ml.agentic.training.train_dpo import (
    SAME_ACTION_REASON_QUALITY_POLICY,
    _same_action_reason_quality_errors,
    validate_preference_dataset,
)
from scripts.build_verifier_reason_dpo_preferences import _auc, has_target_rationale_prefix


EVIDENCE = "宽窄巷子尚未结束，人民公园重叠45分钟。"
PHRASES = ["宽窄巷子尚未结束", "人民公园重叠45分钟"]


def _pair() -> dict:
    action = "retry_solve"
    chosen_reason = build_grounded_repair_reason(EVIDENCE, action)
    return {
        "pair_id": "reason-quality:ok",
        "action": action,
        "negative_type": "A_evidence_only",
        "evidence": EVIDENCE,
        "grounding_phrases": PHRASES,
        "chosen": {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": action, "arguments": {"reason": chosen_reason}}}
            ],
        },
        "rejected": {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": action, "arguments": {"reason": EVIDENCE}}}
            ],
        },
    }


def test_same_action_reason_validator_accepts_a_correct_pair():
    assert _same_action_reason_quality_errors(_pair()) == []


def test_same_action_reason_validator_rejects_swapped_action_or_evidence():
    swapped = _pair()
    swapped["rejected"]["tool_calls"][0]["function"]["name"] = "abort"
    assert any("SAME_ACTION_CHANGED" in item for item in _same_action_reason_quality_errors(swapped))

    changed_evidence = _pair()
    changed_evidence["rejected"]["tool_calls"][0]["function"]["arguments"]["reason"] = "另一条证据"
    assert any("A_REASON_NOT_EVIDENCE_EOS" in item for item in _same_action_reason_quality_errors(changed_evidence))


def test_wrong_action_rationale_is_detected_as_not_target_rationale():
    wrong = build_grounded_repair_reason(EVIDENCE, "abort")
    assert not has_target_rationale_prefix(wrong, "retry_solve")
    assert has_target_rationale_prefix(build_grounded_repair_reason(EVIDENCE, "retry_solve"), "retry_solve")


def test_same_action_reason_validator_rejects_b_with_the_target_rationale():
    row = _pair()
    row["negative_type"] = "B_wrong_action_rationale"
    row["rejected"]["tool_calls"][0]["function"]["arguments"]["reason"] = build_grounded_repair_reason(
        EVIDENCE, "retry_solve"
    )
    assert any(
        "B_TARGET_RATIONALE_NOT_FAILED" in item
        for item in _same_action_reason_quality_errors(row)
    )


def test_length_auc_helper_is_tie_aware_and_balanced():
    # Chosen and rejected have identical multisets of lengths: length alone is
    # no better than chance even though every pair itself can differ.
    assert _auc([2, 4, 4, 2], [1, 1, 0, 0]) == 0.5


def _write_train_only_dataset(tmp_path):
    import hashlib
    import json

    actions = ("retry_solve", "propose_tradeoff", "abort")
    negatives = ("A_evidence_only", "B_wrong_action_rationale")
    rows = []
    for index in range(100):
        for action in actions:
            for negative in negatives:
                evidence = f"第{index + 1}天的可见验证证据。"
                reason = evidence
                if negative == "B_wrong_action_rationale":
                    reason = next(
                        candidate
                        for wrong_action in actions
                        if wrong_action != action
                        for prefix in repair_reason_rationale_prefixes(wrong_action)
                        for candidate in [f"{prefix}：{evidence}"]
                        if not verifier_reason_quality_checks(
                            reason=candidate,
                            target_action=action,
                            grounding_phrases=[evidence],
                            evidence=evidence,
                        )["reason_action_rationale_match"]
                    )
                messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": f"go {index}"}]
                tools = [{"type": "function", "function": {"name": action, "parameters": {}}}]
                source_task_id = f"source-{(index * 3 + actions.index(action)) % 40}"
                rows.append(
                    {
                        "pair_id": f"{action}:{negative}:{index}",
                        "schema_version": SAME_ACTION_REASON_QUALITY_POLICY,
                        "family": f"{action}:{negative}",
                        "reason_codes": ["SAME_ACTION_REASON_QUALITY_CONTRACT"],
                        "source_task_id": source_task_id,
                        "action": action,
                        "negative_type": negative,
                        "context_hash": hashlib.sha256(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                        "evidence": evidence,
                        "grounding_phrases": [evidence],
                        "messages": messages,
                        "tools": tools,
                        "chosen": {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": action, "arguments": {"reason": build_grounded_repair_reason(evidence, action)}}}]},
                        "rejected": {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": action, "arguments": {"reason": reason}}}]},
                    }
                )
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "passed", "dataset_version": "unit", "preference_evidence_policy": SAME_ACTION_REASON_QUALITY_POLICY, "split_policy": "train-only", "audit": {"ready": True, "mean_chosen_minus_rejected_completion_tokens": 0.0, "length_only_auc": 0.5}}), encoding="utf-8")
    return rows


def test_train_only_validator_accepts_contract_and_rejects_action_or_evidence_changes(tmp_path):
    rows = _write_train_only_dataset(tmp_path)
    assert validate_preference_dataset(tmp_path, 600)["ready"] is True

    rows[0]["rejected"]["tool_calls"][0]["function"]["name"] = "abort"
    import json
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert any("SAME_ACTION_CHANGED" in error for error in validate_preference_dataset(tmp_path, 600)["errors"])

    rows = _write_train_only_dataset(tmp_path)
    rows[0]["rejected"]["tool_calls"][0]["function"]["arguments"]["reason"] = "different evidence"
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert any("A_REASON_NOT_EVIDENCE_EOS" in error for error in validate_preference_dataset(tmp_path, 600)["errors"])

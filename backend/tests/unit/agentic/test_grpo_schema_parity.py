from scripts.audit_grpo_schema_parity import build_report


def test_schema_parity_report_uses_one_strict_contract_and_rejection_code():
    report = build_report(
        transformers_schema=lambda method: {
            "type": "function",
            "function": {
                "name": method.__name__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                    },
                    "required": ["reason"],
                },
            },
        },
        transformers_version="test",
    )

    assert report["passed"] is True
    assert report["checks"]["trainer_equals_production"] is True
    assert report["checks"]["frozen_audit_equals_production"] is True
    assert report["checks"]["unexpected_properties_forbidden"] is True
    assert report["rejections"]["production_validator"]["rejection_code"] == (
        "UNEXPECTED_ARGUMENT:candidates"
    )
    assert report["rejections"]["trl_environment_method"]["rejection_code"] == (
        "UNEXPECTED_ARGUMENT:candidates"
    )

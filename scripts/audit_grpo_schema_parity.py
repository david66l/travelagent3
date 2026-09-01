"""Prove Trainer, frozen-audit, and runtime policy argument contract parity."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.policy_actions import (  # noqa: E402
    PolicyArgumentValidationError,
    model_visible_policy_actions,
    policy_action_schema,
    policy_action_schemas,
    policy_action_schemas_for_state,
    validate_policy_arguments,
)
from agentic.grpo_training import (  # noqa: E402
    VERIFIER_REPAIR_ACTIONS_BY_ROUTE,
    VERIFIER_REPAIR_SCHEMA_CAPABILITY_BY_ROUTE,
)
from agentic.trl_environment import (  # noqa: E402
    build_trl_environment_factories,
    canonical_trl_tool_schemas,
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _rejection(call: Callable[[], Any]) -> dict[str, Any]:
    try:
        call()
    except PolicyArgumentValidationError as exc:
        return {
            "exception": type(exc).__name__,
            "rejection_code": exc.rejection_code,
            "validation_errors": exc.validation_errors,
        }
    except Exception as exc:  # pragma: no cover - a parity failure in the pinned runtime
        return {"exception": type(exc).__name__, "message": str(exc)}
    return {"exception": None}


def build_report(
    *,
    transformers_schema: Callable[[Callable[..., Any]], dict[str, Any]],
    transformers_version: str,
) -> dict[str, Any]:
    factories = build_trl_environment_factories("react")
    environments = {
        route: factories[route](audit_enabled=False)
        for route in VERIFIER_REPAIR_ACTIONS_BY_ROUTE
    }
    environment = environments["decision_verifier_repair_retry"]
    method = environment.retry_solve
    raw_arguments = {
        "strategy": "greedy",
        "reason": "verifier-grounded retry",
        "candidates": ["untrusted-poi"],
    }
    production = policy_action_schema("retry_solve")
    frozen_audit = policy_action_schemas(["retry_solve"])[0]
    route_reports: dict[str, dict[str, Any]] = {}
    for route, expected_actions in VERIFIER_REPAIR_ACTIONS_BY_ROUTE.items():
        route_environment = environments[route]
        all_public_tools = {
            name: member
            for name, member in inspect.getmembers(
                route_environment,
                predicate=inspect.ismethod,
            )
            if name not in {"reset", "get_reward"} and not name.startswith("_")
        }
        capability = VERIFIER_REPAIR_SCHEMA_CAPABILITY_BY_ROUTE[route]
        visible_names = model_visible_policy_actions(
            tuple(all_public_tools),
            capability=capability,
        )
        public_tools = {name: all_public_tools[name] for name in visible_names}
        trainer_schemas = canonical_trl_tool_schemas(
            list(all_public_tools.values()),
            action_order=expected_actions,
            capability=capability,
        )
        production_schemas = policy_action_schemas_for_state(
            expected_actions,
            capability=capability,
        )
        route_checks = {
            "public_tool_surface_exact": set(public_tools) == set(expected_actions)
            and len(public_tools) == len(expected_actions),
            "trainer_schema_order_exact": [
                schema["function"]["name"] for schema in trainer_schemas
            ]
            == list(expected_actions),
            "trainer_schemas_equal_production": trainer_schemas
            == production_schemas,
            "execution_tools_are_bound_methods": all(
                inspect.ismethod(tool) and tool.__self__ is route_environment
                for tool in public_tools.values()
            ),
        }
        route_reports[route] = {
            "expected_actions": list(expected_actions),
            "public_actions": list(public_tools),
            "trainer_model_visible": trainer_schemas,
            "production_validator": production_schemas,
            "schema_set_sha256": _canonical_sha256(trainer_schemas),
            "checks": route_checks,
            "passed": all(route_checks.values()),
        }
    trainer_visible = route_reports["decision_verifier_repair_retry"][
        "trainer_model_visible"
    ][0]
    permissive_signature = transformers_schema(method)
    validator_rejection = _rejection(
        lambda: validate_policy_arguments("retry_solve", raw_arguments)
    )
    environment_rejection = _rejection(lambda: method(**raw_arguments))
    parameters = trainer_visible["function"]["parameters"]
    checks = {
        "legacy_broad_factory_absent": "decision_verifier_repair" not in factories,
        "all_narrow_routes_registered": set(VERIFIER_REPAIR_ACTIONS_BY_ROUTE)
        <= set(factories),
        "all_route_contracts_pass": all(
            route_report["passed"] for route_report in route_reports.values()
        ),
        "trainer_equals_production": trainer_visible == production,
        "frozen_audit_equals_production": frozen_audit == production,
        "only_model_owned_reason": set(parameters["properties"]) == {"reason"},
        "required_equal": parameters.get("required") == ["reason"],
        "reason_min_length_equal": parameters["properties"]["reason"].get("minLength")
        == 1,
        "unexpected_properties_forbidden": parameters.get("additionalProperties") is False,
        "validator_rejection_is_stable": validator_rejection.get("rejection_code")
        == "UNEXPECTED_ARGUMENT:candidates",
        "environment_rejection_is_stable": environment_rejection.get("rejection_code")
        == "UNEXPECTED_ARGUMENT:candidates",
        "validator_and_environment_rejections_equal": validator_rejection.get(
            "rejection_code"
        )
        == environment_rejection.get("rejection_code"),
    }
    return {
        "schema_version": "grpo-policy-schema-parity.v1",
        "scope": "read-only contract audit; no model generation or optimizer step",
        "transformers_version": transformers_version,
        "action": "retry_solve",
        "raw_invalid_arguments": raw_arguments,
        "schemas": {
            "trainer_model_visible": trainer_visible,
            "frozen_checkpoint_audit": frozen_audit,
            "production_validator": production,
            "transformers_signature_baseline": permissive_signature,
            "routes": route_reports,
        },
        "canonical_schema_sha256": _canonical_sha256(production),
        "aggregate_schema_set_sha256": _canonical_sha256(
            {
                route: report["trainer_model_visible"]
                for route, report in route_reports.items()
            }
        ),
        "rejections": {
            "production_validator": validator_rejection,
            "trl_environment_method": environment_rejection,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "known_historical_gap": (
            "The completed diagnostic run retained the rejected key name in its TypeError, "
            "but not the full original candidates value; this audit does not reconstruct it."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import transformers
    from transformers.utils import get_json_schema

    report = build_report(
        transformers_schema=get_json_schema,
        transformers_version=transformers.__version__,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

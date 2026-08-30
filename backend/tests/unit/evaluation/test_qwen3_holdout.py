import importlib.util
from pathlib import Path

from agentic.grpo_training import GRPOCorpusRow


SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "build_qwen3_holdout.py"
SPEC = importlib.util.spec_from_file_location("build_qwen3_holdout", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _row(*, missing=False, feasible=True, failed_search=False) -> GRPOCorpusRow:
    payload = {
        "task": {
            "task_id": f"task-{missing}-{feasible}-{failed_search}",
            "template_family": "test",
            "difficulty": "L2",
            "seed": 7,
            "user_request": "请规划旅行",
            "missing_slots": ["budget"] if missing else [],
            "feasibility_report": {"feasible": feasible},
        },
        "snapshot": {
            "environment_version": "test.v1",
            "snapshot_version": "snapshot.v1",
            "state_id": "state-1",
            "tool_responses": {
                "search_pois": [
                    {
                        "data_source": "unavailable" if failed_search else "built_in",
                        "error_code": "UPSTREAM" if failed_search else None,
                        "data": {},
                    }
                ]
            },
        },
    }
    return GRPOCorpusRow(**payload)


def test_family_and_route_cover_policy_decisions():
    assert module.route_and_actions(_row(missing=True)) == ("clarification", ["ask_user"])
    assert module.route_and_actions(_row(feasible=False)) == (
        "tradeoff",
        ["propose_tradeoff", "abort"],
    )
    assert module.task_family(_row(failed_search=True)) == "recovery"


def test_benchmark_case_hashes_model_visible_payload():
    case, payload_hash, signature = module.benchmark_case(_row(missing=True))

    assert case.expected_action == "ask_user"
    assert case.allowed_actions == ["ask_user"]
    assert len(payload_hash) == 64
    assert len(signature) == 64

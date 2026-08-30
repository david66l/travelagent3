import json

from agentic.policy_actions import policy_action_schemas
from agentic.sft_dataset import SFTExample, SFTMessage, SFTToolCall, SFTToolFunction
from scripts.build_sft_benchmark_cases import build


def test_sft_benchmark_preserves_exact_tool_arguments(tmp_path):
    source = tmp_path / "test.jsonl"
    output = tmp_path / "cases" / "test.jsonl"
    example = SFTExample(
        example_id="example-1",
        scenario_id="scenario-1",
        trajectory_id="trajectory-1",
        step_index=0,
        split="test",
        quality_label="validated_plan",
        source="teacher",
        environment_version="env-v1",
        policy_name="teacher",
        policy_version="v1",
        messages=[
            SFTMessage(role="system", content="policy"),
            SFTMessage(role="user", content="context"),
            SFTMessage(
                role="assistant",
                tool_calls=[
                    SFTToolCall(
                        function=SFTToolFunction(
                            name="search_pois",
                            arguments={"keywords": ["博物馆"]},
                        )
                    )
                ],
            ),
        ],
        tools=policy_action_schemas(["search_pois"]),
    )
    source.write_text(example.model_dump_json() + "\n", encoding="utf-8")

    manifest = build(source, output)
    case = json.loads(output.read_text(encoding="utf-8"))

    assert manifest["cases"] == 1
    assert case["expected_action"] == "search_pois"
    assert case["expected_arguments"] == {"keywords": ["博物馆"]}

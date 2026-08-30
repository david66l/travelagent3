"""Tests for dependency-light Agent Policy training preflight."""

from agentic.sft_dataset import DatasetManifest, SFTExample, SFTMessage
from agentic.sft_dataset import SFTToolCall, SFTToolFunction
from agentic.policy_actions import policy_action_schemas
from agentic.training import (
    check_training_dependencies,
    preflight_sft_model,
    preflight_sft_dataset,
    preflight_sft_termination_boundaries,
    select_sft_smoke_rows,
    to_conversational_prompt_completion,
)


def test_dependency_check_reports_installed_version_conflicts(monkeypatch):
    import agentic.training as training_module

    real_find_spec = training_module.importlib.util.find_spec
    real_version = training_module.importlib.metadata.version
    real_requires = training_module.importlib.metadata.requires

    def fake_find_spec(name):
        if name in {"trl", "transformers"}:
            return object()
        return real_find_spec(name)

    def fake_version(name):
        if name == "trl":
            return "1.0.0"
        if name == "transformers":
            return "4.57.6"
        return real_version(name)

    def fake_requires(name):
        if name == "trl":
            return ["transformers>=999"]
        return real_requires(name)

    monkeypatch.setattr(training_module.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(training_module.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(training_module.importlib.metadata, "requires", fake_requires)

    dependencies = check_training_dependencies()
    trl = next(item for item in dependencies if item.name == "trl")

    assert trl.incompatible_requirements
    assert trl.incompatible_requirements[0].startswith("transformers>=999")


class _QwenToolTokenizer:
    chat_template = "native-qwen-tool-template"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        tools,
        enable_thinking,
    ):
        assert tools
        assert enable_thinking is False
        roles = [message["role"] for message in messages]
        rendered = "|".join(roles)
        if messages and messages[-1]["role"] == "assistant":
            rendered += "<tool_call>{}</tool_call>"
        elif add_generation_prompt:
            rendered += "|assistant"
        if not tokenize:
            return rendered
        return [ord(character) for character in rendered]


class _BoundaryTokenizer(_QwenToolTokenizer):
    eos_token_id = ord("§")

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        tools,
        enable_thinking,
    ):
        rendered = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        if messages and messages[-1]["role"] == "assistant":
            rendered += "§\n"
        if not tokenize:
            return rendered
        return [ord(character) for character in rendered]

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(chr(token_id) for token_id in token_ids)


def _example(split: str, index: int) -> SFTExample:
    return SFTExample(
        example_id=f"trajectory:{index}",
        scenario_id=f"scenario-{index}",
        trajectory_id=f"trajectory-{index}",
        step_index=0,
        split=split,
        quality_label="validated_plan",
        source="teacher",
        environment_version="env-v1",
        policy_name="teacher",
        policy_version="v1",
        messages=[
            SFTMessage(role="system", content="policy"),
            SFTMessage(role="user", content=f"context-{index}"),
            SFTMessage(
                role="assistant",
                tool_calls=[SFTToolCall(function=SFTToolFunction(name="get_weather"))],
            ),
        ],
        tools=policy_action_schemas(["get_weather"]),
    )


def _dataset(tmp_path, *, overlap: bool = False):
    rows = {
        "train": [_example("train", 1)],
        "validation": [_example("validation", 2)],
        "test": [_example("test", 3)],
    }
    for split, examples in rows.items():
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(item.model_dump_json() for item in examples) + "\n",
            encoding="utf-8",
        )
    manifest = DatasetManifest(
        dataset_version="sft-test",
        candidate_episodes=3,
        accepted_episodes=3,
        rejected_episodes=0,
        exported_examples=3,
        split_examples={"train": 1, "validation": 1, "test": 1},
        source_episodes={"teacher": 3},
        quality_episodes={"validated_plan": 3},
        rejection_codes={},
        environment_versions=["env-v1"],
        policy_versions=["teacher:v1"],
        split_group_overlap=overlap,
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def test_conversational_rows_train_only_on_assistant_completion():
    row = _example("train", 1)

    converted = to_conversational_prompt_completion([row.model_dump(mode="json")])

    assert [message["role"] for message in converted[0]["prompt"]] == ["system", "user"]
    assert converted[0]["completion"][0]["role"] == "assistant"
    assert converted[0]["completion"][0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert converted[0]["completion"][0]["tool_calls"][0]["function"]["arguments"] == {}
    assert converted[0]["tools"][0]["function"]["name"] == "get_weather"


def test_smoke_selection_round_robins_final_actions():
    rows = []
    for action_index, action in enumerate(("search_pois", "ask_user", "propose_tradeoff")):
        for index in range(5):
            example = _example("train", action_index * 10 + index)
            example.messages[-1].tool_calls[0].function.name = action
            rows.append(example.model_dump(mode="json"))

    selected = select_sft_smoke_rows(rows, 6)
    actions = [row["messages"][-1]["tool_calls"][0]["function"]["name"] for row in selected]

    assert actions == [
        "ask_user",
        "propose_tradeoff",
        "search_pois",
        "ask_user",
        "propose_tradeoff",
        "search_pois",
    ]


def test_preflight_can_validate_data_without_gpu_dependencies(tmp_path):
    _dataset(tmp_path)

    report = preflight_sft_dataset(
        tmp_path,
        minimum_train_examples=1,
        require_dependencies=False,
    )

    assert report.ready is True
    assert report.dataset_version == "sft-test"
    assert report.train_examples == 1
    assert report.validation_examples == 1
    assert report.test_examples == 1
    assert report.unique_model_visible_payloads == 3


def test_preflight_blocks_small_or_leaking_dataset(tmp_path):
    _dataset(tmp_path, overlap=True)

    report = preflight_sft_dataset(
        tmp_path,
        minimum_train_examples=3000,
        require_dependencies=False,
    )

    assert report.ready is False
    assert "DATASET_SPLIT_GROUP_OVERLAP" in report.errors
    assert any(error.startswith("TRAIN_EXAMPLES_BELOW_MINIMUM") for error in report.errors)


def test_preflight_blocks_id_only_payload_diversity(tmp_path):
    _dataset(tmp_path)
    train = _example("train", 1)
    duplicate = train.model_copy(
        update={
            "example_id": "different-id",
            "scenario_id": "different-scenario",
            "trajectory_id": "different-trajectory",
            "split": "validation",
        },
        deep=True,
    )
    (tmp_path / "validation.jsonl").write_text(duplicate.model_dump_json() + "\n", encoding="utf-8")

    report = preflight_sft_dataset(
        tmp_path,
        minimum_train_examples=1,
        require_dependencies=False,
    )

    assert report.ready is False
    assert "MODEL_VISIBLE_PAYLOAD_DUPLICATES:1" in report.errors
    assert report.unique_model_visible_payloads == 2


def test_model_preflight_checks_native_tool_template_and_length(tmp_path):
    _dataset(tmp_path)

    report = preflight_sft_model(tmp_path, _QwenToolTokenizer(), max_length=200)

    assert report.ready is True
    assert report.rows_checked == 3
    assert report.tool_envelope_rows == 3
    assert report.over_max_length == 0


def test_model_preflight_rejects_missing_template(tmp_path):
    _dataset(tmp_path)
    tokenizer = _QwenToolTokenizer()
    tokenizer.chat_template = None

    report = preflight_sft_model(tmp_path, tokenizer)

    assert report.ready is False
    assert report.errors == ["MODEL_CHAT_TEMPLATE_MISSING"]


def test_termination_boundary_preflight_finds_immediate_eos(tmp_path):
    _dataset(tmp_path)

    report = preflight_sft_termination_boundaries(tmp_path, _BoundaryTokenizer())

    assert report.ready is True
    assert report.rows_checked == 3
    assert report.boundary_rows == 3
    assert report.termination_token_id == ord("§")


def test_termination_boundary_preflight_rejects_missing_eos(tmp_path):
    _dataset(tmp_path)
    tokenizer = _QwenToolTokenizer()
    tokenizer.eos_token_id = ord("§")
    tokenizer.decode = lambda token_ids, *, skip_special_tokens: "".join(
        chr(token_id) for token_id in token_ids
    )

    report = preflight_sft_termination_boundaries(tmp_path, tokenizer)

    assert report.ready is False
    assert report.missing_termination_rows == 3
    assert "SFT_TERMINATION_TOKEN_MISSING:3" in report.errors

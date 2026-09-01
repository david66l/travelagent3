"""Audit the pinned TRL Trainer's real route-scoped prompt rendering without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import (  # noqa: E402
    VERIFIER_REPAIR_ACTIONS_BY_ROUTE,
    VERIFIER_REPAIR_ROUTE_BY_TARGET,
    VERIFIER_REPAIR_SCHEMA_CAPABILITY_BY_ROUTE,
    load_grpo_corpus,
    to_trl_environment_rows,
    tool_result_suffix_ids,
)
from agentic.chat_template_contract import (  # noqa: E402
    AGENT_CHAT_TEMPLATE_KWARGS,
    AGENT_CHAT_TEMPLATE_SHA256,
    AGENT_RENDER_PROTOCOL_VERSION,
    render_agent_tool_prompt,
)
from agentic.policy_actions import policy_action_schemas_for_state  # noqa: E402
from agentic.trl_environment import build_trl_environment_factories  # noqa: E402
from ml.agentic.training.train_grpo import (  # noqa: E402
    _disable_unused_vllm_import,
    create_stable_tool_suffix_grpo_trainer_class,
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _select_rows(corpus_dir: Path) -> list[Any]:
    rows = load_grpo_corpus(corpus_dir / "train.jsonl")
    selected = []
    for target in ("retry_solve", "propose_tradeoff", "abort"):
        selected.append(
            next(
                row
                for row in rows
                if row.snapshot.hidden_test_facts["grpo_decision_state"][
                    "target_action"
                ]
                == target
            )
        )
    return selected


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise RuntimeError("single prompt tokenization returned a batch")
        value = value[0]
    return list(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import transformers
    from datasets import Dataset
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    _disable_unused_vllm_import()
    import trl
    from trl import GRPOConfig, GRPOTrainer
    from trl.chat_template_utils import is_chat_template_prefix_preserving

    if trl.__version__ != "1.9.2":
        raise RuntimeError(f"trainer render audit requires TRL 1.9.2, found {trl.__version__}")

    source_rows = _select_rows(args.corpus_dir)
    rows = to_trl_environment_rows(source_rows)
    expected_routes = [
        VERIFIER_REPAIR_ROUTE_BY_TARGET[
            row.snapshot.hidden_test_facts["grpo_decision_state"]["target_action"]
        ]
        for row in source_rows
    ]
    if [row["environment"] for row in rows] != expected_routes:
        raise RuntimeError("transported routes differ from the target/action contract")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise RuntimeError("source SFT tokenizer does not provide a chat template")
    native_chat_template_sha256 = hashlib.sha256(
        tokenizer.chat_template.encode("utf-8")
    ).hexdigest()

    # The random tiny model is never forwarded or optimized. It only satisfies
    # the real GRPOTrainer initializer so this audit exercises TRL's own
    # environment discovery and `_tokenize_prompts` implementation.
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=max(len(tokenizer), 128),
            n_positions=1024,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    config = GRPOConfig(
        output_dir=str(args.output.parent / ".trainer-render-init"),
        per_device_train_batch_size=3,
        num_generations=3,
        max_completion_length=512,
        gradient_checkpointing=False,
        report_to=[],
        use_vllm=False,
        chat_template_kwargs=dict(AGENT_CHAT_TEMPLATE_KWARGS),
        bf16=False,
        fp16=False,
    )
    trainer_class = create_stable_tool_suffix_grpo_trainer_class(
        base_trainer_class=GRPOTrainer,
        trl_version=trl.__version__,
    )
    trainer = trainer_class(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(rows, on_mixed_types="use_json"),
        processing_class=tokenizer,
        environment_factory=build_trl_environment_factories("react"),
    )

    actual_schemas = {
        route: trainer._env_tools[route] for route in VERIFIER_REPAIR_ACTIONS_BY_ROUTE
    }
    expected_schemas = {
        route: policy_action_schemas_for_state(
            actions,
            capability=VERIFIER_REPAIR_SCHEMA_CAPABILITY_BY_ROUTE[route],
        )
        for route, actions in VERIFIER_REPAIR_ACTIONS_BY_ROUTE.items()
    }
    trainer._batch_environments = expected_routes
    prompts = [row["prompt"] for row in rows]
    actual_token_ids, images, multimodal_fields = trainer._tokenize_prompts(prompts)

    independent_token_ids = []
    serving_renderer_token_ids = []
    rendered_prompts = []
    for prompt, route in zip(prompts, expected_routes, strict=True):
        schema = expected_schemas[route]
        tokenized = tokenizer.apply_chat_template(
            conversation=[prompt],
            tools=schema,
            chat_template=trainer.chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            **trainer.chat_template_kwargs,
        )
        independent_token_ids.append(tokenized["input_ids"][0])
        serving_renderer_token_ids.append(
            _token_ids(
                render_agent_tool_prompt(
                    tokenizer,
                    prompt,
                    tools=schema,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                )
            )
        )
        rendered_prompts.append(
            tokenizer.apply_chat_template(
                conversation=prompt,
                tools=schema,
                chat_template=trainer.chat_template,
                add_generation_prompt=True,
                tokenize=False,
                **trainer.chat_template_kwargs,
            )
        )

    prefix_checks = []
    for prompt, route in zip(prompts, expected_routes, strict=True):
        schema = expected_schemas[route]
        for boundary in range(1, len(prompt)):
            prefix_ids = _token_ids(
                tokenizer.apply_chat_template(
                    conversation=prompt[:boundary],
                    tools=schema,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    **trainer.chat_template_kwargs,
                )
            )
            extended_ids = _token_ids(
                tokenizer.apply_chat_template(
                    conversation=prompt[: boundary + 1],
                    tools=schema,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    **trainer.chat_template_kwargs,
                )
            )
            prefix_checks.append(extended_ids[: len(prefix_ids)] == prefix_ids)

    tool_messages = [
        {"role": "tool", "name": "abort", "content": '{"done":true}'}
    ]
    suffix_ids = trainer._get_tool_suffix_ids(tool_messages)
    independent_suffix_ids = tool_result_suffix_ids(
        tokenizer,
        tool_messages=tool_messages,
        chat_template=trainer.chat_template,
        chat_template_kwargs=trainer.chat_template_kwargs,
    )

    checks = {
        "trl_version_pinned": trl.__version__ == "1.9.2",
        "legacy_broad_route_absent": "decision_verifier_repair"
        not in trainer.environment_factories,
        "actual_env_tools_equal_production": actual_schemas == expected_schemas,
        "actual_token_ids_equal_independent_render": actual_token_ids
        == independent_token_ids,
        "actual_token_ids_equal_shared_serving_renderer": actual_token_ids
        == serving_renderer_token_ids,
        "shared_render_protocol_active": trainer.agent_render_protocol_version
        == AGENT_RENDER_PROTOCOL_VERSION,
        "shared_template_hash_active": trainer.agent_chat_template_sha256
        == AGENT_CHAT_TEMPLATE_SHA256,
        "no_images": images is None,
        "no_multimodal_fields": multimodal_fields == {},
        "optimizer_not_created": getattr(trainer, "optimizer", None) is None,
        "template_prefix_preserving": is_chat_template_prefix_preserving(tokenizer),
        "all_incremental_prefixes_exact": all(prefix_checks),
        "tool_suffix_boundary_exact": suffix_ids == independent_suffix_ids,
    }
    route_evidence = {}
    for index, (source_row, route) in enumerate(
        zip(source_rows, expected_routes, strict=True)
    ):
        route_evidence[route] = {
            "task_id": source_row.task.task_id,
            "target_action": source_row.snapshot.hidden_test_facts[
                "grpo_decision_state"
            ]["target_action"],
            "actions": list(VERIFIER_REPAIR_ACTIONS_BY_ROUTE[route]),
            "schema_set_sha256": _canonical_sha256(actual_schemas[route]),
            "rendered_prompt_sha256": hashlib.sha256(
                rendered_prompts[index].encode("utf-8")
            ).hexdigest(),
            "token_ids_sha256": _canonical_sha256(actual_token_ids[index]),
            "token_count": len(actual_token_ids[index]),
        }

    report = {
        "schema_version": "grpo-actual-trainer-render-audit.v1",
        "scope": "real pinned Trainer initialization and tokenization; no generation or optimizer",
        "passed": all(checks.values()),
        "checks": checks,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
        },
        "tokenizer": {
            "source": args.tokenizer,
            "class": type(tokenizer).__name__,
            "vocab_sha256": _canonical_sha256(tokenizer.get_vocab()),
            "native_chat_template_sha256": native_chat_template_sha256,
            "chat_template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode("utf-8")
            ).hexdigest(),
            "chat_template_kwargs": trainer.chat_template_kwargs,
        },
        "render_protocol_version": AGENT_RENDER_PROTOCOL_VERSION,
        "aggregate_schema_set_sha256": _canonical_sha256(actual_schemas),
        "aggregate_rendered_prompt_sha256": _canonical_sha256(rendered_prompts),
        "aggregate_token_ids_sha256": _canonical_sha256(actual_token_ids),
        "routes": route_evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

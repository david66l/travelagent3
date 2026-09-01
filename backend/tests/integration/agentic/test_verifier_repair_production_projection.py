"""Production boundary checks for the verifier-repair constraint contract."""

from pathlib import Path

from agentic.grpo_training import load_grpo_corpus
from agentic.runtime import initialize_agent_ledger
from agentic.state import AgentLedgerState
from contracts.constraint_flexibility import ConstraintFlexibilityContract
from models.travel_slots import (
    ConstraintFlexibilityContract as ModelConstraintFlexibilityContract,
)
from models.travel_slots import TravelSlots
from scripts.build_verifier_repair_grpo_corpus import _TEMPLATES, _prepare_variant


def test_offline_contract_matches_production_slot_and_goal_projection():
    source = load_grpo_corpus(
        Path("ml/agentic/datasets/native-react-grpo-v1/train.jsonl")
    )[0]
    row = _prepare_variant(
        source,
        split="validation",
        template=_TEMPLATES["validation"][1],
        ordinal=1,
    )
    offline_contract = row.task.slots["constraint_flexibility"]
    parsed_slots = TravelSlots.model_validate(row.task.slots)
    result = initialize_agent_ledger(
        {
            "user_input": row.task.user_request,
            "slots": parsed_slots.to_flat_dict(),
            "missing_slots": [],
        },
        mode="agent",
    )
    projected = AgentLedgerState(**result["agent_ledger"]).goal.hard_constraints[
        "constraint_flexibility"
    ]

    assert ModelConstraintFlexibilityContract is ConstraintFlexibilityContract
    assert parsed_slots.constraint_flexibility is not None
    assert parsed_slots.constraint_flexibility.model_dump(mode="json") == offline_contract
    assert projected == offline_contract

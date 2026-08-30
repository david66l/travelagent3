import json

from agentic.corpus_generation import CurriculumTeacherPolicy, build_curriculum_case
from agentic.environment import TravelAgentEnvironment
from agentic.policy import ControllerFirstPolicy
from agentic.sft_dataset import EpisodeCandidate
from scripts.build_multiturn_recovery_dataset import build_multiturn_example


async def test_multiturn_recovery_matches_tool_history_contract():
    task, snapshot = build_curriculum_case(7)
    rollout = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(CurriculumTeacherPolicy())
    )
    candidate = EpisodeCandidate(
        scenario_id=task.task_id,
        source="synthetic",
        template_family=task.template_family,
        city="广州",
        episode=rollout.episode,
    )

    example = build_multiturn_example(candidate, split="train")

    assert [item.role for item in example.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert example.messages[3].name == "search_pois"
    transition = json.loads(example.messages[3].content)
    assert transition["done"] is False
    assert transition["last_transition"]["observations"][0]["error_code"] == "UPSTREAM_TIMEOUT"
    assert transition["policy_state"]["failure_summary"][0]["retryable"] is True

"""Verifier-guided teacher selection and preference extraction tests."""

from agentic.trajectory import episode_content_hash


def _failed_variant(rollout):
    failed = rollout.model_copy(deep=True)
    failed.episode.trajectory_id = "rejected-trajectory"
    failed.reward.trajectory_id = "rejected-trajectory"
    for step in failed.episode.steps:
        step.context.trajectory_id = "rejected-trajectory"
    policy_step = next(
        step for step in failed.episode.steps if step.action.decision_source != "controller"
    )
    keywords = list(policy_step.action.arguments.get("keywords") or [])
    policy_step.action.arguments["keywords"] = keywords[:1]
    failed.reward.gate_status = "task_failed"
    failed.reward.components.task = -1
    failed.reward.components.constraint = -1
    failed.reward.episode_reward = -0.25
    failed.reward.audit_metrics["hard_pass"] = False
    failed.reward.audit_metrics["invalid_model_steps"] = 1
    failed.episode.content_hash = episode_content_hash(failed.episode)
    return failed

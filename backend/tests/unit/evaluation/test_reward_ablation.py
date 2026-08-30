from agentic.reward import EpisodeReward, RewardComponents
from evaluation.reward_ablation import build_reward_ablation_report


def _reward(trajectory_id: str, task: float, tool: float, total: float) -> EpisodeReward:
    return EpisodeReward(
        trajectory_id=trajectory_id,
        reward_config_version="hierarchical-b0.v1",
        gate_status="passed",
        components=RewardComponents(
            task=task,
            constraint=1,
            format=1,
            tool=tool,
            grounding=1,
            efficiency=1,
        ),
        terminal_reward=0.8,
        process_reward=0.2,
        quality_reward=0,
        episode_reward=total,
        turn_rewards=[],
    )


def test_reward_ablation_reports_all_six_components():
    report = build_reward_ablation_report([_reward("a", 1, 1, 1), _reward("b", 0.5, -0.5, 0.7)])

    assert [row.component for row in report.rows] == [
        "task",
        "constraint",
        "format",
        "tool",
        "grounding",
        "efficiency",
    ]
    task = next(row for row in report.rows if row.component == "task")
    assert task.ablated_mean_reward < task.full_mean_reward
    assert "does not claim" in report.note

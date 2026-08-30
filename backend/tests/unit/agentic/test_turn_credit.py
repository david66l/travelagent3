import pytest

from agentic.turn_credit import model_token_segments, project_turn_relative_advantages


def test_model_token_segments_separate_tool_observations():
    assert model_token_segments([1, 1, 0, 0, 1, 1, 1, 0]) == [(0, 2), (4, 7)]


def test_one_turn_rollouts_remain_exactly_trajectory_b0():
    values, report = project_turn_relative_advantages(
        [1.0, -1.0, 0.5, -0.5],
        [[0.8], [-0.8], [0.4], [-0.4]],
        [[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0]],
        group_size=4,
    )

    assert values == [
        [1.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.0, 0.0],
    ]
    assert report.eligible_multiturn_trajectories == 0
    assert report.locally_credited_turns == 0
    assert report.effective_nonzero_credited_turns == 0


def test_multiturn_credit_changes_only_model_segments_within_group():
    values, report = project_turn_relative_advantages(
        [1.0, -1.0, 1.0, -1.0],
        [[-0.5, 0.9], [0.5, -0.9], [-0.5, 0.9], [0.5, -0.9]],
        [[1, 1, 0, 0, 1, 0]] * 4,
        group_size=4,
        blend_weight=0.5,
    )

    assert values[0][0] < values[0][4]
    assert values[1][0] > values[1][4]
    assert all(row[2:4] == [0.0, 0.0] for row in values)
    assert report.eligible_multiturn_trajectories == 4
    assert report.locally_credited_turns == 8
    assert report.effective_nonzero_credited_turns == 8


def test_zero_variance_turn_credit_falls_back_to_trajectory_b0():
    values, report = project_turn_relative_advantages(
        [0.0, 0.0],
        [[0.5, 0.8], [0.5, 0.8]],
        [[1, 0, 1], [1, 0, 1]],
        group_size=2,
    )

    assert values == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert report.zero_variance_turn_buckets == 2
    assert report.locally_credited_turns == 0
    assert report.effective_nonzero_credited_turns == 0


def test_unmatched_final_text_segment_is_outside_tool_policy_credit_scope():
    values, report = project_turn_relative_advantages(
        [0.8, -0.8],
        [[0.2], [-0.2]],
        [[1, 0, 1], [1, 0, 1]],
        group_size=2,
    )

    assert values == [[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]
    assert report.invalid_model_turns == 0
    assert report.unmatched_model_turns == 2
    assert report.alignment_rejected_trajectories == 0
    assert report.extra_unmatched_model_turns == 0
    assert report.invalid_action_positive_credit_rate == 0


def test_extra_unmatched_segments_reject_trajectory_from_positional_projection():
    values, report = project_turn_relative_advantages(
        [1.0, -1.0],
        [[0.0, 0.8], [0.0, -0.8]],
        [[1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1]],
        group_size=2,
        policy_turn_validities=[
            ["external_failure", "valid"],
            ["external_failure", "valid"],
        ],
    )

    assert values == [[0.0] * 7, [0.0] * 7]
    assert report.alignment_rejected_trajectories == 2
    assert report.extra_unmatched_model_turns == 2
    assert report.locally_credited_turns == 0


def test_projection_rejects_non_grouped_batch():
    with pytest.raises(ValueError, match="divisible"):
        project_turn_relative_advantages(
            [1.0],
            [[1.0]],
            [[1]],
            group_size=4,
        )


def test_invalid_turn_never_falls_back_to_positive_trajectory_advantage():
    values, report = project_turn_relative_advantages(
        [1.0, 1.0],
        [[-1.0, 0.8], [-1.0, 0.4]],
        [[1, 0, 1], [1, 0, 1]],
        group_size=2,
        policy_turn_validities=[
            ["invalid", "valid"],
            ["invalid", "valid"],
        ],
    )

    assert values[0][0] == -1.0
    assert values[1][0] == -1.0
    assert report.invalid_model_turns == 2
    assert report.invalid_action_positive_credit_count == 0
    assert report.invalid_action_positive_credit_rate == 0


def test_external_failure_turn_is_neutral_and_excluded_from_group_norm():
    values, report = project_turn_relative_advantages(
        [1.0, -1.0],
        [[0.0, 0.8], [0.0, -0.8]],
        [[1, 0, 1], [1, 0, 1]],
        group_size=2,
        policy_turn_validities=[
            ["external_failure", "valid"],
            ["external_failure", "valid"],
        ],
    )

    assert values[0][0] == 0.0
    assert values[1][0] == 0.0
    assert report.external_failure_turns == 2
    assert report.compared_turn_buckets == 1
    assert report.zero_advantage_group_ratio == 0


def test_valid_recovery_is_ranked_against_invalid_same_turn_counterfactual():
    values, report = project_turn_relative_advantages(
        [1.0, -1.0, 1.0, -1.0],
        [[0.0, 0.9], [0.0, -1.0], [0.0, 0.9], [0.0, -1.0]],
        [[1, 0, 1]] * 4,
        group_size=4,
        policy_turn_validities=[
            ["external_failure", "valid"],
            ["external_failure", "invalid"],
            ["external_failure", "valid"],
            ["external_failure", "invalid"],
        ],
    )

    assert values[0][2] > 0
    assert values[1][2] == -1.0
    assert report.compared_turn_buckets == 1
    assert report.zero_variance_turn_buckets == 0
    assert report.effective_nonzero_credited_turns == 2
    assert report.invalid_action_positive_credit_count == 0

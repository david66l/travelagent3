"""Tests for unified planning job input (user_feedback vs input_requirements)."""

from core.conversation_turn import (
    effective_user_feedback,
    feedback_from_input_requirements,
    user_profile_from_job,
)
from models.planning_job import PlanningJob
from tests.support.planning_feedback import feedback_with_trip


def test_feedback_from_input_requirements():
    state = feedback_from_input_requirements(
        {"destination": "杭州", "travel_days": 3, "travelers_count": 2}
    )
    flat = state["profile"]["trip"]
    assert flat["destination"] == "杭州"
    assert flat["travel_days"] == 3
    assert state["phase"] == "planning"


def test_effective_user_feedback_prefers_conversation_state():
    fb = feedback_with_trip("上海", 2)
    resolved = effective_user_feedback(fb, {"destination": "北京", "travel_days": 5})
    assert resolved is fb


def test_effective_user_feedback_falls_back_to_requirements():
    resolved = effective_user_feedback({}, {"destination": "成都", "travel_days": 4})
    assert resolved["profile"]["trip"]["destination"] == "成都"


def test_user_profile_from_job_rest_path():
    job = PlanningJob(
        input_requirements={"destination": "西安", "travel_days": 2, "travelers_count": 1},
        user_feedback={},
    )
    profile = user_profile_from_job(job)
    assert profile.destination == "西安"
    assert profile.travel_days == 2


def test_user_profile_from_job_ws_path():
    job = PlanningJob(user_feedback=feedback_with_trip("广州", 3))
    profile = user_profile_from_job(job)
    assert profile.destination == "广州"
    assert profile.travel_days == 3

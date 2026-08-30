"""Reproducible curriculum tasks executed through the production Agent Loop."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from agentic.environment import EnvironmentSnapshot, EnvironmentTask, SnapshotToolResponse
from agentic.loop import PolicyAction, PolicyContext
from evaluation.validator import ItineraryValidator


CURRICULUM_SCHEMA_VERSION = "travel-curriculum.v2"
_CITIES = (
    ("上海", 31.2304, 121.4737),
    ("北京", 39.9042, 116.4074),
    ("杭州", 30.2741, 120.1551),
    ("南京", 32.0603, 118.7969),
    ("苏州", 31.2989, 120.5853),
    ("成都", 30.5728, 104.0668),
    ("西安", 34.3416, 108.9398),
    ("广州", 23.1291, 113.2644),
)
_FAMILIES = (
    ("history-culture", ["历史", "博物馆"]),
    ("food-exploration", ["美食", "本地文化"]),
    ("family-relaxed", ["亲子", "公园"]),
    ("art-architecture", ["艺术", "建筑"]),
    ("nature-scenic", ["自然", "摄影"]),
    ("classic-first-visit", ["地标", "历史"]),
)


class CurriculumTeacherPolicy:
    """Select the verified happy-path action while retaining policy decisions."""

    async def propose(self, context: PolicyContext) -> PolicyAction:
        capability = context.capability.get("status")
        task_id = str(context.current_subtask.get("task_id") or "")
        if "ask_user" in context.allowed_actions and capability == "needs_user":
            detail = "、".join(context.missing_information) or "缺失信息"
            return PolicyAction(
                action="ask_user",
                arguments={"question": f"请补充{detail}，我再继续规划。"},
            )
        if (
            "abort" in context.allowed_actions
            and capability in {"infeasible", "unsafe", "missing_tool"}
            and context.capability.get("actionable_alternatives") is False
        ):
            evidence = [str(item) for item in context.capability.get("evidence") or []]
            return PolicyAction(
                action="abort",
                arguments={"reason": evidence[0] if evidence else "没有安全可行的替代方案"},
            )
        if "propose_tradeoff" in context.allowed_actions and capability == "infeasible":
            evidence = [str(item) for item in context.capability.get("evidence") or []]
            alternatives = [str(item) for item in context.capability.get("alternatives") or []]
            return PolicyAction(
                action="propose_tradeoff",
                arguments={
                    "reason": evidence[0] if evidence else "当前预算与行程天数冲突",
                    "options": alternatives or ["提高预算", "减少行程天数"],
                },
            )
        if "finish" in context.allowed_actions:
            return PolicyAction(action="finish")
        if task_id == "search_candidates":
            has_candidates = any(
                item.get("artifact_type") == "poi_candidate_set"
                and int(item.get("poi_count") or 0) > 0
                for item in context.relevant_artifacts
            )
            if has_candidates and "accept_candidates" in context.allowed_actions:
                return PolicyAction(action="accept_candidates")
            interests = list(context.soft_preferences.get("interests") or [])
            return PolicyAction(action="search_pois", arguments={"keywords": interests[:2]})
        if task_id == "review_itinerary":
            latest_report = next(
                (
                    item
                    for item in reversed(context.relevant_artifacts)
                    if item.get("artifact_type") == "validation_report"
                ),
                None,
            )
            if latest_report and latest_report.get("hard_pass") is True:
                return PolicyAction(action="accept_itinerary")
            return PolicyAction(
                action="propose_tradeoff",
                arguments={
                    "reason": "验证器发现当前行程仍有硬约束冲突",
                    "options": ["减少景点", "调整预算或日期"],
                },
            )
        action = context.allowed_actions[0]
        arguments: dict[str, Any] = {}
        if action == "search_pois":
            interests = list(context.soft_preferences.get("interests") or [])
            arguments["keywords"] = interests[:2]
        return PolicyAction(action=action, arguments=arguments)


class AdaptiveRecoveryTeacherPolicy(CurriculumTeacherPolicy):
    """Recover from a visible broad-query failure without using hidden facts.

    This policy is a data-generation oracle, but deliberately has access only to
    the same ``PolicyContext`` exposed to the production policy.  It may narrow
    a query only when the tool's failure message names exactly one of the
    user's grounded interests.
    """

    async def propose(self, context: PolicyContext) -> PolicyAction:
        if str(context.current_subtask.get("task_id") or "") == "search_candidates" and any(
            item.get("artifact_type") == "poi_candidate_set" and int(item.get("poi_count") or 0) > 0
            for item in context.relevant_artifacts
        ):
            return PolicyAction(action="accept_candidates")
        if "search_pois" not in context.allowed_actions:
            return await super().propose(context)

        interests = [str(item) for item in context.soft_preferences.get("interests") or []]
        broad_failures = [
            item
            for item in context.failure_summary
            if item.get("code") == "QUERY_TOO_BROAD" and item.get("retryable")
        ]
        if not broad_failures:
            return PolicyAction(action="search_pois", arguments={"keywords": interests[:2]})

        message = str(broad_failures[-1].get("message") or "")
        grounded_targets = [interest for interest in interests if interest and interest in message]
        if len(grounded_targets) != 1:
            raise ValueError(
                "adaptive recovery requires exactly one grounded interest in visible failure feedback"
            )
        return PolicyAction(
            action="search_pois",
            arguments={"keywords": [grounded_targets[0]]},
        )


def build_curriculum_case(index: int) -> tuple[EnvironmentTask, EnvironmentSnapshot]:
    """Build one deterministic, unique task/snapshot pair from an integer seed."""
    city, lat, lng = _CITIES[index % len(_CITIES)]
    family, interests = _FAMILIES[(index // len(_CITIES)) % len(_FAMILIES)]
    days = 1 + (index % 3)
    # Keep model-visible requests diverse beyond the old 1,200-case cycle.
    # Three years of dates and 37 budget bands remain realistic while making
    # large formal corpora semantically distinct rather than ID-only distinct.
    start = date(2027, 1, 4) + timedelta(days=index % 1095)
    budget = float(days * (700 + (index % 37) * 25))
    scenario_kind = (
        "plan",
        "plan",
        "plan",
        "plan",
        "plan",
        "plan",
        "missing",
        "retry",
        "infeasible",
        "plan",
    )[index % 10]
    task_id = f"curriculum-{index:05d}-{city}"
    candidates, itinerary = _build_plan(city, lat, lng, days, index)
    constraints = {
        "travel_days": days,
        "total_budget": budget,
        "must_visit": [],
        "interests": interests,
        "include_restaurant": True,
        "meals_per_day": 2,
    }
    report = ItineraryValidator().validate(
        itinerary,
        constraints=constraints,
        facts=candidates,
    )
    if not report.hard_pass:
        codes = [item.code for item in report.hard_violations]
        raise ValueError(f"curriculum fixture failed validation: {codes}")

    missing_slots = ["budget_range"] if scenario_kind == "missing" else []
    feasible = scenario_kind != "infeasible"
    task = EnvironmentTask(
        task_id=task_id,
        template_family=family,
        difficulty="L1" if days == 1 else ("L2" if days == 2 else "L3"),
        seed=index,
        user_request=(f"请规划{days}天{city}行程，偏好{'、'.join(interests)}，预算{int(budget)}元"),
        slots={
            "destination": city,
            "travel_days": days,
            "start_date": start.isoformat(),
            "end_date": (start + timedelta(days=days - 1)).isoformat(),
            "budget_range": budget,
            "interests": interests,
        },
        profile={"destination": city, "travel_days": days, "interests": interests},
        missing_slots=missing_slots,
        feasibility_report={
            "feasible": feasible,
            "status": (
                "needs_user" if missing_slots else ("solvable" if feasible else "infeasible")
            ),
            "reasons": ([] if feasible else ["预算不足以覆盖指定天数"]),
            "actionable_alternatives": (True if not feasible else None),
            "alternatives": (["提高预算", "减少行程天数"] if not feasible else []),
        },
    )
    attraction_names = [item["name"] for item in candidates if item["category"] == "attraction"][:6]
    # TravelActionExecutor deliberately excludes restaurants/hotels/transport
    # from the bounded POI-detail batch. Mirror that exact selection contract
    # in the immutable snapshot; using the first eight mixed candidates leaves
    # three-day cases without responses for their later attraction requests.
    detail_names = attraction_names[:8]
    matrix_size = len(attraction_names) + 1
    matrix = [
        [0 if row == col else 8 + abs(row - col) * 4 for col in range(matrix_size)]
        for row in range(matrix_size)
    ]
    snapshot = EnvironmentSnapshot(
        environment_version=CURRICULUM_SCHEMA_VERSION,
        # Version identifies the generator contract; state_id identifies one
        # concrete immutable world.
        snapshot_version=CURRICULUM_SCHEMA_VERSION,
        state_id=f"curriculum-state-{index:05d}",
        tool_responses={
            "get_weather": [
                {
                    "data": [
                        {
                            "date": (start + timedelta(days=offset)).isoformat(),
                            "condition": ("晴" if (index + offset) % 4 else "多云"),
                            "temperature": "18-26℃",
                        }
                        for offset in range(days)
                    ]
                }
            ],
            "search_pois": [{"data": candidates}],
            "get_poi_detail": [
                {
                    "data": next(item for item in candidates if item["name"] == name),
                    "expected_arguments": {
                        "poi_name": name,
                        "city": city,
                    },
                }
                for name in detail_names
            ],
            "get_route_matrix": [
                {
                    "data": {
                        "poi_ids": ["__hotel", *attraction_names],
                        "time_minutes": matrix,
                        "transport_cost": [
                            [0.0 if row == col else 3.0 for col in range(matrix_size)]
                            for row in range(matrix_size)
                        ],
                    }
                }
            ],
            "solve_itinerary": [
                {
                    "data": {
                        "status": "optimal",
                        "days": itinerary,
                        "solve_time_ms": 5 + index % 7,
                    }
                }
            ],
            "validate_itinerary": [{"data": report.model_dump(mode="json")}],
        },
        hidden_test_facts={
            "closed_pois": [],
            "validator_report": report.model_dump(mode="json"),
        },
    )
    if scenario_kind == "retry":
        snapshot.tool_responses["search_pois"].insert(
            0,
            SnapshotToolResponse(
                data=None,
                data_source="unavailable",
                fallback_reason="snapshot timeout",
                error_code="UPSTREAM_TIMEOUT",
                retryable=True,
            ),
        )
    return task, snapshot


def _build_plan(
    city: str, lat: float, lng: float, days: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    itinerary: list[dict[str, Any]] = []
    for day_number in range(1, days + 1):
        suffix = f"{seed:05d}-{day_number}"
        names = {
            "morning": f"{city}文化馆-{suffix}",
            "lunch": f"{city}午餐厅-{suffix}",
            "afternoon": f"{city}城市公园-{suffix}",
            "dinner": f"{city}晚餐厅-{suffix}",
        }
        activities = [
            _activity(names["morning"], "attraction", "09:00", "11:00", 120, 40),
            _activity(names["lunch"], "restaurant", "12:00", "13:00", 60, 80),
            _activity(names["afternoon"], "attraction", "14:00", "16:00", 120, 30),
            _activity(names["dinner"], "restaurant", "18:00", "19:00", 60, 100),
        ]
        for offset, activity in enumerate(activities):
            candidates.append(
                {
                    "id": f"poi-{suffix}-{offset}",
                    "name": activity["poi_name"],
                    "category": activity["category"],
                    "score": round(0.75 + ((seed + offset) % 20) / 100, 2),
                    "location": {
                        "lat": lat + (day_number * 4 + offset) * 0.001,
                        "lng": lng + (day_number * 3 + offset) * 0.001,
                    },
                    "ticket_price": activity["cost"],
                    "open_time": "08:00",
                    "close_time": "21:00",
                    "recommended_hours": activity["duration_min"] / 60,
                    "tags": ["本地", activity["category"]],
                }
            )
        itinerary.append(
            {
                "day_number": day_number,
                "activities": activities,
                "total_cost": sum(item["cost"] for item in activities),
                "transport_cost": 30,
            }
        )
    return candidates, itinerary


def _activity(
    name: str, category: str, start: str, end: str, duration: int, cost: float
) -> dict[str, Any]:
    return {
        "poi_id": name,
        "poi_name": name,
        "category": category,
        "start_time": start,
        "end_time": end,
        "duration_min": duration,
        "cost": cost,
    }


__all__ = [
    "AdaptiveRecoveryTeacherPolicy",
    "CURRICULUM_SCHEMA_VERSION",
    "CurriculumTeacherPolicy",
    "build_curriculum_case",
]

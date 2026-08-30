"""Deterministic task-graph planning for the Agent Loop MVP."""

from __future__ import annotations

from agentic.state import GoalLedger, TaskGraph, TaskNode


MANDATORY_GATE_TASKS = frozenset(
    {"capability_check", "solve_itinerary", "validate_itinerary", "review_itinerary"}
)


class DefaultTaskGraphPlanner:
    """Build the safe minimum DAG that a policy may later refine, not delete."""

    def plan(self, goal: GoalLedger, *, plan_version: int = 1) -> TaskGraph:
        tasks: list[TaskNode] = [
            TaskNode(
                task_id="capability_check",
                goal="Verify that required information, tools and permissions are available",
                allowed_actions=("capability_check", "ask_user", "propose_tradeoff", "abort"),
                success_criteria={"required_artifact_types": ["capability_report"]},
                invalidates_on=("goal_changed",),
            )
        ]

        first_dependency = "capability_check"
        if goal.missing_information:
            tasks.append(
                TaskNode(
                    task_id="resolve_missing_information",
                    goal="Obtain travel information that only the user can provide",
                    depends_on=(first_dependency,),
                    allowed_actions=("ask_user",),
                    success_criteria={
                        "required_fact_keys": [
                            f"user_input.{item}" for item in goal.missing_information
                        ]
                    },
                    invalidates_on=("goal_changed",),
                )
            )
            first_dependency = "resolve_missing_information"

        tasks.extend(
            [
                TaskNode(
                    task_id="collect_weather",
                    goal="Collect date-specific weather facts",
                    depends_on=(first_dependency,),
                    allowed_actions=("get_weather",),
                    success_criteria={"min_successful_observations": 1},
                    invalidates_on=("destination_changed", "travel_dates_changed"),
                ),
                TaskNode(
                    task_id="search_candidates",
                    goal=(
                        "Iteratively search and decide when grounded candidates are sufficient "
                        "for planning"
                    ),
                    depends_on=(first_dependency,),
                    allowed_actions=(
                        "accept_candidates",
                        "search_pois",
                        "ask_user",
                        "propose_tradeoff",
                        "abort",
                    ),
                    success_criteria={
                        "required_fact_keys": ["candidate_poi_ids"],
                        "required_artifact_types": ["candidate_selection"],
                        "min_candidate_count": 1,
                        "decision_loop": True,
                    },
                    max_attempts=5,
                    invalidates_on=(
                        "destination_changed",
                        "travel_dates_changed",
                        "preferences_changed",
                    ),
                ),
                TaskNode(
                    task_id="collect_poi_details",
                    goal="Collect opening hours, prices and durations for selected POIs",
                    depends_on=("search_candidates",),
                    required_facts=("candidate_poi_ids",),
                    allowed_actions=("get_poi_detail",),
                    success_criteria={"required_artifact_types": ["poi_detail_set"]},
                    invalidates_on=(
                        "destination_changed",
                        "travel_dates_changed",
                        "candidate_set_changed",
                    ),
                ),
                TaskNode(
                    task_id="collect_route_matrix",
                    goal="Collect travel times among selected places",
                    depends_on=("search_candidates",),
                    required_facts=("candidate_poi_ids",),
                    allowed_actions=("get_route_matrix",),
                    success_criteria={"required_artifact_types": ["route_matrix"]},
                    invalidates_on=("candidate_set_changed", "transport_mode_changed"),
                ),
                TaskNode(
                    task_id="solve_itinerary",
                    goal="Generate a constraint-satisfying itinerary with the deterministic solver",
                    depends_on=(
                        "collect_weather",
                        "collect_poi_details",
                        "collect_route_matrix",
                    ),
                    allowed_actions=("solve_itinerary",),
                    success_criteria={"required_artifact_types": ["solver_result"]},
                    invalidates_on=("goal_changed", "planning_fact_changed"),
                ),
                TaskNode(
                    task_id="validate_itinerary",
                    goal="Programmatically compute all hard-constraint verification evidence",
                    depends_on=("solve_itinerary",),
                    allowed_actions=("validate_itinerary",),
                    success_criteria={
                        "required_artifact_types": ["validation_report"],
                    },
                    invalidates_on=("goal_changed", "solver_result_changed"),
                ),
                TaskNode(
                    task_id="review_itinerary",
                    goal=(
                        "Inspect verifier evidence and either accept the itinerary or choose a "
                        "bounded repair, clarification, tradeoff, or safe stop"
                    ),
                    depends_on=("validate_itinerary",),
                    allowed_actions=(
                        "accept_itinerary",
                        "retry_solve",
                        "search_pois",
                        "ask_user",
                        "propose_tradeoff",
                        "abort",
                    ),
                    success_criteria={
                        "required_artifact_types": ["verified_itinerary_acceptance"],
                    },
                    max_attempts=4,
                    invalidates_on=("goal_changed", "validation_result_changed"),
                ),
                TaskNode(
                    task_id="compose_draft",
                    goal="Compose the user-facing draft from verified artifacts",
                    depends_on=("review_itinerary",),
                    allowed_actions=("compose_draft",),
                    success_criteria={"required_artifact_types": ["itinerary_draft"]},
                    invalidates_on=("goal_changed", "validation_result_changed"),
                ),
                TaskNode(
                    task_id="await_confirmation",
                    goal="Present the verified draft and wait for user confirmation",
                    depends_on=("compose_draft",),
                    allowed_actions=("ask_user", "finish"),
                    success_criteria={"required_fact_keys": ["user_confirmation"]},
                    invalidates_on=("draft_changed",),
                ),
            ]
        )
        return TaskGraph(
            goal_version=goal.goal_version,
            plan_version=plan_version,
            tasks=tuple(tasks),
        )

    @staticmethod
    def ensure_mandatory_gates(graph: TaskGraph) -> None:
        task_ids = {task.task_id for task in graph.tasks}
        missing = MANDATORY_GATE_TASKS - task_ids
        if missing:
            raise ValueError(f"task graph removed mandatory gates: {sorted(missing)}")

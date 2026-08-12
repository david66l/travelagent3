"""Adapter from Agent Loop actions to existing deterministic travel tools."""

from __future__ import annotations

import json
from typing import Any

from agentic.loop import ActionOutcome, PolicyAction
from agentic.observations import ObservationEnvelope
from agentic.state import AgentLedgerState, ArtifactRecord, FactRecord, TaskNode
from tools.tool_executor import ToolExecutor


_ARTIFACT_TYPES = {
    "get_weather": "weather_snapshot",
    "get_poi_detail": "poi_detail_set",
    "check_reservation": "poi_detail_set",
    "get_route_matrix": "route_matrix",
    "solve_itinerary": "solver_result",
    "validate_itinerary": "validation_report",
}


class TravelActionExecutor:
    """Execute policy actions without granting the policy direct state writes."""

    def __init__(self, tool_executor: ToolExecutor | None = None) -> None:
        self.tools = tool_executor or ToolExecutor()

    async def execute(
        self,
        *,
        task: TaskNode,
        action: PolicyAction,
        ledger: AgentLedgerState,
    ) -> ActionOutcome:
        if action.action == "ask_user":
            return ActionOutcome(
                status="awaiting_user",
                artifacts=[
                    self._artifact(
                        ledger,
                        action,
                        "user_question",
                        {"question": action.arguments.get("question")},
                    )
                ],
            )
        if action.action == "abort":
            return ActionOutcome(
                status="failed",
                error_code="POLICY_ABORT",
                error_message=str(action.arguments.get("reason") or "policy aborted"),
            )
        if action.action == "capability_check":
            return ActionOutcome(
                artifacts=[
                    self._artifact(
                        ledger,
                        action,
                        "capability_report",
                        ledger.goal.capability.model_dump(mode="json"),
                    )
                ]
            )
        if action.action == "compose_draft":
            solver = self._latest_artifact(ledger, "solver_result")
            if solver is None:
                return ActionOutcome(
                    status="failed",
                    error_code="SOLVER_ARTIFACT_MISSING",
                    error_message="cannot compose a draft before solver output exists",
                )
            return ActionOutcome(
                artifacts=[
                    self._artifact(
                        ledger,
                        action,
                        "itinerary_draft",
                        solver.payload,
                        evidence_refs=[solver.artifact_id],
                    )
                ]
            )
        if action.action in {"finish", "propose_tradeoff"}:
            return ActionOutcome(
                status="awaiting_user",
                artifacts=[
                    self._artifact(
                        ledger,
                        action,
                        action.action,
                        action.arguments,
                    )
                ],
            )

        if action.action == "get_poi_detail":
            return await self._collect_poi_details(task, action, ledger)

        arguments = self._hydrate_arguments(ledger, action)
        call = {
            "id": action.action_id,
            "type": "function",
            "function": {
                "name": action.action,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
        records = await self.tools.execute(
            [call],
            guard_context={"allowed_tools": set(task.allowed_actions), "max_calls": 1},
        )
        record = records[0]
        observation = ObservationEnvelope(**record["observation"])
        if not observation.ok:
            error = observation.error
            return ActionOutcome(
                status="failed",
                observations=[observation],
                error_code=error.code if error else "TOOL_FAILED",
                error_message=error.message if error else "tool failed",
                retryable=error.retryable if error else False,
            )

        facts: list[FactRecord] = []
        artifacts: list[ArtifactRecord] = []
        if action.action == "search_pois":
            items = observation.data if isinstance(observation.data, list) else []
            poi_ids = [
                str(item.get("id") or item.get("name"))
                for item in items
                if isinstance(item, dict) and (item.get("id") or item.get("name"))
            ]
            facts.append(
                self._fact(
                    ledger,
                    action,
                    observation,
                    "candidate_poi_ids",
                    poi_ids,
                )
            )
            artifacts.append(
                self._artifact(
                    ledger,
                    action,
                    "poi_candidate_set",
                    {"pois": items},
                    evidence_refs=[action.action_id],
                )
            )
        elif artifact_type := _ARTIFACT_TYPES.get(action.action):
            payload = (
                observation.data
                if isinstance(observation.data, dict)
                else {"data": observation.data}
            )
            if action.action == "get_weather" and isinstance(observation.data, list):
                payload = {"days": observation.data}
            artifacts.append(
                self._artifact(
                    ledger,
                    action,
                    artifact_type,
                    payload,
                    evidence_refs=[action.action_id],
                )
            )
        return ActionOutcome(observations=[observation], facts=facts, artifacts=artifacts)

    async def _collect_poi_details(
        self,
        task: TaskNode,
        action: PolicyAction,
        ledger: AgentLedgerState,
    ) -> ActionOutcome:
        candidates = self._latest_artifact(ledger, "poi_candidate_set")
        items = candidates.payload.get("pois", []) if candidates else []
        names = [
            str(item.get("name")) for item in items if isinstance(item, dict) and item.get("name")
        ]
        if not names:
            return ActionOutcome(
                status="failed",
                error_code="CANDIDATE_POIS_MISSING",
                error_message="POI details require a verified candidate set",
            )
        city = ledger.goal.hard_constraints.get("destination")
        calls = [
            {
                "id": f"{action.action_id}:{index}",
                "type": "function",
                "function": {
                    "name": "get_poi_detail",
                    "arguments": json.dumps({"poi_name": name, "city": city}, ensure_ascii=False),
                },
            }
            for index, name in enumerate(names)
        ]
        records = await self.tools.execute(
            calls,
            guard_context={
                "allowed_tools": set(task.allowed_actions),
                "max_calls": len(calls),
            },
        )
        observations = [ObservationEnvelope(**record["observation"]) for record in records]
        usable = [item for item in observations if item.ok]
        if len(usable) != len(names):
            failed = next((item for item in observations if not item.ok), None)
            error = failed.error if failed else None
            return ActionOutcome(
                status="failed",
                observations=observations,
                error_code=error.code if error else "POI_DETAIL_SET_INCOMPLETE",
                error_message=error.message if error else "POI detail set is incomplete",
                retryable=error.retryable if error else True,
            )
        return ActionOutcome(
            observations=observations,
            artifacts=[
                self._artifact(
                    ledger,
                    action,
                    "poi_detail_set",
                    {"details": [item.data for item in usable], "expected_count": len(names)},
                    evidence_refs=[
                        item.tool_call_id for item in usable if item.tool_call_id is not None
                    ],
                )
            ],
        )

    def _hydrate_arguments(self, ledger: AgentLedgerState, action: PolicyAction) -> dict[str, Any]:
        """Inject trusted payloads so the policy selects rather than copies facts."""
        arguments = dict(action.arguments)
        destination = ledger.goal.hard_constraints.get("destination")
        if action.action in {"search_pois", "get_weather"} and destination:
            arguments["city"] = destination

        if action.action in {"get_route_matrix", "solve_itinerary"}:
            candidates = self._latest_artifact(ledger, "poi_candidate_set")
            if candidates is not None:
                arguments["pois"] = [
                    self._poi_input(item, index)
                    for index, item in enumerate(candidates.payload.get("pois") or [])
                    if isinstance(item, dict) and item.get("name")
                ]
            arguments["constraints"] = self._trusted_constraints(ledger)

        if action.action == "solve_itinerary":
            matrix = self._latest_artifact(ledger, "route_matrix")
            if matrix is not None:
                arguments["dist_matrix"] = matrix.payload.get("time_minutes")
                arguments["tc_matrix"] = matrix.payload.get("transport_cost")

        if action.action == "validate_itinerary":
            solver = self._latest_artifact(ledger, "solver_result")
            if solver is not None:
                arguments["itinerary"] = solver.payload.get("days") or []
            arguments["constraints"] = self._trusted_constraints(ledger)
            candidates = self._latest_artifact(ledger, "poi_candidate_set")
            arguments["facts"] = candidates.payload.get("pois", []) if candidates else []
        return arguments

    @staticmethod
    def _trusted_constraints(ledger: AgentLedgerState) -> dict[str, Any]:
        hard = ledger.goal.hard_constraints
        soft = ledger.goal.soft_preferences
        return {
            "travel_days": int(hard.get("travel_days") or 1),
            "total_budget": float(hard.get("budget_range") or 0),
            "max_transit_minutes": int(hard.get("max_transit_minutes") or 120),
            "must_visit": list(hard.get("must_visit") or []),
            "interests": list(soft.get("interests") or []),
        }

    @staticmethod
    def _poi_input(item: dict[str, Any], index: int) -> dict[str, Any]:
        location = item.get("location") or {}
        recommended = str(item.get("recommended_hours") or "")
        duration = 120
        try:
            duration = max(15, int(float(recommended) * 60))
        except ValueError:
            pass
        return {
            "id": str(item.get("id") or item.get("name") or f"poi-{index}"),
            "name": str(item.get("name") or f"POI-{index}"),
            "category": str(item.get("category") or "attraction"),
            "tags": list(item.get("tags") or []),
            "lat": float(item.get("lat") or location.get("lat") or 0),
            "lng": float(item.get("lng") or location.get("lng") or 0),
            "score": float(item.get("score") or 0.5),
            "ticket_price": float(item.get("ticket_price") or item.get("price") or 0),
            "duration_minutes": duration,
            "open_time": str(item.get("open_time") or "08:00"),
            "close_time": str(item.get("close_time") or "18:00"),
        }

    @staticmethod
    def _latest_artifact(ledger: AgentLedgerState, artifact_type: str) -> ArtifactRecord | None:
        matches = [
            artifact
            for artifact in ledger.artifacts.values()
            if artifact.artifact_type == artifact_type
            and artifact.goal_version == ledger.goal.goal_version
            and artifact.plan_version == ledger.task_graph.plan_version
        ]
        return matches[-1] if matches else None

    @staticmethod
    def _artifact(
        ledger: AgentLedgerState,
        action: PolicyAction,
        artifact_type: str,
        payload: dict[str, Any],
        *,
        evidence_refs: list[str] | None = None,
    ) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=f"{ledger.trajectory_id}:{action.action_id}:{artifact_type}",
            artifact_type=artifact_type,
            payload=payload,
            evidence_refs=evidence_refs or [action.action_id],
            goal_version=ledger.goal.goal_version,
            plan_version=ledger.task_graph.plan_version,
        )

    @staticmethod
    def _fact(
        ledger: AgentLedgerState,
        action: PolicyAction,
        observation: ObservationEnvelope,
        key: str,
        value: Any,
    ) -> FactRecord:
        return FactRecord(
            fact_id=f"{ledger.trajectory_id}:{action.action_id}:{key}",
            key=key,
            value=value,
            observation_ref=action.action_id,
            goal_version=ledger.goal.goal_version,
            plan_version=ledger.task_graph.plan_version,
            source=observation.source,
            confidence=observation.confidence,
        )

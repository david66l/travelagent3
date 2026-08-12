"""Hallucination detector for generated itineraries.

Cross-checks planned activities against retrieved POI candidates and tool
results (POI details, routes, reservations) to surface fabricated or
inconsistent facts before the itinerary is returned to the user.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from schemas import ValidationResult

logger = logging.getLogger(__name__)


class HallucinationDetectionAgent:
    """Detect hallucinations in itinerary activities.

    The agent is deliberately defensive: if the itinerary or the supporting
    tool results are missing, it returns ``passed=True`` so that the overall
    planning flow is not blocked by an absence of verification data.
    """

    _EXISTENCE_TOOLS = ("get_poi_detail", "find_restaurants", "find_hotels")
    _RESERVATION_KEYWORDS = ("预约", "reserve")

    @classmethod
    def detect(cls, state: dict) -> ValidationResult:
        """Run all hallucination checks and aggregate a ``ValidationResult``."""
        itinerary = state.get("itinerary") or []
        poi_candidates = state.get("poi_candidates") or []
        tool_results = state.get("tool_results") or []

        if not itinerary or not tool_results:
            # No evidence means "not evaluated", not a perfect verification
            # score. Keep the flow non-blocking while avoiding inflated quality
            # metrics that would later contaminate SFT/GRPO reward data.
            return ValidationResult(passed=True, total_score=0.0)

        poi_details = cls._extract_poi_details(tool_results)
        route_results = [r for r in tool_results if r.get("name") == "get_route"]
        reservation_results = [r for r in tool_results if r.get("name") == "check_reservation"]

        scores: dict[str, float] = {}
        issues: list[str] = []
        critical_failures: list[str] = []

        existence_score, existence_issues, existence_critical = cls.check_poi_existence(
            itinerary, poi_candidates, tool_results
        )
        scores["poi_existence"] = existence_score
        issues.extend(existence_issues)
        critical_failures.extend(existence_critical)

        hours_score, hours_issues = cls.check_opening_hours(itinerary, poi_details)
        scores["opening_hours"] = hours_score
        issues.extend(hours_issues)

        price_score, price_issues = cls.check_ticket_prices(itinerary, poi_details)
        scores["ticket_prices"] = price_score
        issues.extend(price_issues)

        route_score, route_issues = cls.check_route_commute(itinerary, route_results)
        scores["route_commute"] = route_score
        issues.extend(route_issues)

        reservation_score, reservation_issues = cls.check_reservation_annotations(
            itinerary, reservation_results
        )
        scores["reservation_annotations"] = reservation_score
        issues.extend(reservation_issues)

        total_score = sum(scores.values()) / len(scores) if scores else 1.0
        passed = total_score >= 0.8 and not critical_failures

        return ValidationResult(
            passed=passed,
            scores=scores,
            total_score=round(total_score, 2),
            critical_failures=critical_failures,
            improvement_suggestions=issues,
        )

    @classmethod
    def check_poi_existence(
        cls,
        itinerary: list[dict[str, Any]],
        poi_candidates: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> tuple[float, list[str], list[str]]:
        """Return (score, issues, critical_failures) for POI existence.

        An activity is considered to exist when its ``poi_name`` appears in
        ``poi_candidates`` or in the data returned by a relevant tool.
        Missing POIs are treated as critical failures.
        """
        candidate_names = {
            p.get("name") or p.get("spot_name", "")
            for p in poi_candidates or []
            if p.get("name") or p.get("spot_name")
        }
        tool_names: set[str] = set()

        for tr in tool_results or []:
            if tr.get("name") not in cls._EXISTENCE_TOOLS:
                continue
            data = (tr.get("result") or {}).get("data")
            if isinstance(data, dict):
                name = data.get("name")
                if name:
                    tool_names.add(name)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        name = item.get("name")
                        if name:
                            tool_names.add(name)

        existing_names = candidate_names | tool_names

        activities = [act for day in itinerary or [] for act in day.get("activities") or []]
        if not activities:
            return 1.0, [], []

        missing: list[str] = []
        for act in activities:
            poi_name = act.get("poi_name")
            if not poi_name:
                missing.append("未知POI")
                continue
            if poi_name not in existing_names:
                missing.append(poi_name)

        if not missing:
            return 1.0, [], []

        score = max(0.0, 1.0 - len(missing) / len(activities))
        issues = [f"POI不存在或无法验证：{name}" for name in missing]
        critical = [f"关键：POI '{name}' 未在候选结果或工具返回中找到" for name in missing]
        return score, issues, critical

    @classmethod
    def check_opening_hours(
        cls,
        itinerary: list[dict[str, Any]],
        poi_details: list[dict[str, Any]],
    ) -> tuple[float, list[str]]:
        """Return (score, issues) for opening-hour consistency."""
        details_by_name = {p.get("name"): p for p in poi_details or [] if p.get("name")}
        checked = 0
        issues: list[str] = []

        for day in itinerary or []:
            for act in day.get("activities") or []:
                start = act.get("start_time")
                end = act.get("end_time")
                if not start or not end:
                    continue

                poi_name = act.get("poi_name")
                detail = details_by_name.get(poi_name)
                if not detail:
                    continue

                open_time, close_time = cls._extract_hours(detail)
                if open_time is None or close_time is None:
                    continue

                try:
                    start_min = cls._time_to_minutes(start)
                    end_min = cls._time_to_minutes(end)
                    open_min = cls._time_to_minutes(open_time)
                    close_min = cls._time_to_minutes(close_time)
                except ValueError:
                    continue

                checked += 1
                if start_min < open_min or end_min > close_min:
                    issues.append(
                        f"{poi_name} 开放时间冲突：活动 {start}-{end} 不在"
                        f" {open_time}-{close_time} 内"
                    )

        if checked == 0:
            return 1.0, []
        return max(0.0, 1.0 - len(issues) / checked), issues

    @classmethod
    def check_ticket_prices(
        cls,
        itinerary: list[dict[str, Any]],
        poi_details: list[dict[str, Any]],
        tolerance: float = 0.3,
    ) -> tuple[float, list[str]]:
        """Return (score, issues) for ticket-price consistency."""
        details_by_name = {p.get("name"): p for p in poi_details or [] if p.get("name")}
        checked = 0
        issues: list[str] = []

        for day in itinerary or []:
            for act in day.get("activities") or []:
                activity_price = act.get("ticket_price")
                if activity_price is None:
                    continue

                poi_name = act.get("poi_name")
                detail = details_by_name.get(poi_name)
                if not detail:
                    continue

                detail_price = detail.get("ticket_price")
                if detail_price is None:
                    continue

                checked += 1
                if detail_price == 0:
                    if activity_price != 0:
                        issues.append(
                            f"{poi_name} 票价偏差：工具显示免费，活动标注 {activity_price}"
                        )
                    continue

                ratio = abs(activity_price - detail_price) / detail_price
                if ratio > tolerance:
                    issues.append(
                        f"{poi_name} 票价偏差：活动 {activity_price} vs"
                        f" 工具 {detail_price}（偏差 {ratio:.0%}）"
                    )

        if checked == 0:
            return 1.0, []
        return max(0.0, 1.0 - len(issues) / checked), issues

    @classmethod
    def check_route_commute(
        cls,
        itinerary: list[dict[str, Any]],
        route_results: list[dict[str, Any]],
        tolerance: float = 0.5,
    ) -> tuple[float, list[str]]:
        """Return (score, issues) for route-duration consistency."""
        route_by_dest: dict[str, list[float]] = {}
        for tr in route_results or []:
            data = (tr.get("result") or {}).get("data") or {}
            dest = data.get("destination")
            minutes = data.get("minutes")
            if dest is not None and minutes is not None:
                route_by_dest.setdefault(dest, []).append(float(minutes))

        checked = 0
        issues: list[str] = []

        for day in itinerary or []:
            for act in day.get("activities") or []:
                transit = act.get("transit_from_prev") or {}
                duration = transit.get("duration_min")
                if duration is None:
                    continue

                poi_name = act.get("poi_name")
                route_durations = route_by_dest.get(poi_name, [])
                if not route_durations:
                    continue

                checked += 1
                route_duration = sum(route_durations) / len(route_durations)
                if route_duration == 0:
                    if duration != 0:
                        issues.append(
                            f"{poi_name} 交通时间偏差：活动标注 {duration} 分钟，工具显示 0"
                        )
                    continue

                ratio = abs(duration - route_duration) / route_duration
                if ratio > tolerance:
                    issues.append(
                        f"{poi_name} 交通时间偏差：活动 {duration} 分钟 vs"
                        f" 工具 {route_duration:.0f} 分钟（偏差 {ratio:.0%}）"
                    )

        if checked == 0:
            return 1.0, []
        return max(0.0, 1.0 - len(issues) / checked), issues

    @classmethod
    def check_reservation_annotations(
        cls,
        itinerary: list[dict[str, Any]],
        reservation_results: list[dict[str, Any]],
    ) -> tuple[float, list[str]]:
        """Return (score, issues) for missing reservation reminders."""
        need_reserve: set[str] = set()
        for tr in reservation_results or []:
            data = (tr.get("result") or {}).get("data") or {}
            if data.get("need_reserve"):
                poi = data.get("poi")
                if poi:
                    need_reserve.add(poi)

        if not need_reserve:
            return 1.0, []

        checked = 0
        issues: list[str] = []

        for day in itinerary or []:
            for act in day.get("activities") or []:
                poi_name = act.get("poi_name")
                if poi_name not in need_reserve:
                    continue

                checked += 1
                tags = act.get("tags") or []
                note = act.get("note") or act.get("recommendation_reason") or ""
                text_blob = " ".join(str(t) for t in tags) + " " + str(note)
                text_blob = text_blob.lower()

                if not any(kw in text_blob for kw in cls._RESERVATION_KEYWORDS):
                    issues.append(f"{poi_name} 需要预约，但活动未标注预约提醒")

        if checked == 0:
            return 1.0, []
        return max(0.0, 1.0 - len(issues) / checked), issues

    @classmethod
    def _extract_poi_details(cls, tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collect POI detail dicts from ``get_poi_detail`` tool results."""
        details: list[dict[str, Any]] = []
        for tr in tool_results or []:
            if tr.get("name") != "get_poi_detail":
                continue
            data = (tr.get("result") or {}).get("data")
            if isinstance(data, dict) and data.get("name"):
                details.append(data)
        return details

    @staticmethod
    def _extract_hours(detail: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        """Extract open/close time strings from a POI detail dict."""
        open_time = detail.get("open_time")
        close_time = detail.get("close_time")
        if open_time and close_time:
            return open_time, close_time

        open_hours = detail.get("open_hours")
        if isinstance(open_hours, str) and "-" in open_hours:
            parts = open_hours.split("-")
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()

        return None, None

    @staticmethod
    def _time_to_minutes(time_str: str) -> int:
        """Convert ``HH:MM`` to minutes since midnight."""
        time_str = time_str.strip()
        if ":" not in time_str:
            raise ValueError(f"Unsupported time format: {time_str}")
        hours, minutes = time_str.split(":", 1)
        return int(hours) * 60 + int(minutes)

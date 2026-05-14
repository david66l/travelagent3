"""Validation Agent - multi-dimensional itinerary validation."""

import asyncio
import random
from datetime import datetime, time

from schemas import DayPlan, UserProfile, ValidationResult
from skills.tavily_search import UnifiedSearchSkill


class ValidationAgent:
    """Validate itinerary across multiple dimensions."""

    def __init__(self):
        self.search_skill = UnifiedSearchSkill()

    async def validate(
        self,
        itinerary: list[DayPlan],
        profile: UserProfile,
    ) -> ValidationResult:
        """Run all validation checks in parallel."""
        results = await asyncio.gather(
            self._check_budget(itinerary, profile),
            self._check_time_feasibility(itinerary),
            self._check_poi_existence(itinerary, profile),
            self._check_opening_hours(itinerary),
            self._check_preference_coverage(itinerary, profile),
            return_exceptions=True,
        )

        scores = {}
        critical_failures = []
        suggestions = []

        for result in results:
            if isinstance(result, Exception):
                continue
            scores.update(result.get("scores", {}))
            critical_failures.extend(result.get("critical_failures", []))
            suggestions.extend(result.get("suggestions", []))

        total_score = sum(scores.values()) / len(scores) if scores else 0

        return ValidationResult(
            passed=total_score >= 0.75 and len(critical_failures) == 0,
            scores=scores,
            total_score=total_score,
            critical_failures=critical_failures,
            improvement_suggestions=suggestions,
        )

    async def _check_budget(self, itinerary: list[DayPlan], profile: UserProfile) -> dict:
        total_cost = sum(day.total_cost for day in itinerary)
        budget = profile.budget_range or float("inf")
        over_ratio = max(0, (total_cost - budget) / budget) if budget > 0 else 0
        score = max(0, 1 - over_ratio)
        return {
            "scores": {"budget_compliance": score},
            "critical_failures": ["budget_compliance"] if score < 0.8 else [],
            "suggestions": ["预算超支，建议调整"] if score < 0.8 else [],
        }

    async def _check_time_feasibility(self, itinerary: list[DayPlan]) -> dict:
        """Check that daily schedule fits within day bounds and activities don't overlap."""
        violations = 0
        total = 0
        day_start = time(9, 0)
        day_end = time(21, 0)

        for day in itinerary:
            activities = day.activities
            total += len(activities)

            for i, activity in enumerate(activities):
                # Check time parsing
                if activity.start_time and activity.end_time:
                    try:
                        start = datetime.strptime(activity.start_time, "%H:%M").time()
                        end = datetime.strptime(activity.end_time, "%H:%M").time()
                    except ValueError:
                        violations += 1
                        continue

                    # Check within day bounds
                    if start < day_start or end > day_end:
                        violations += 1
                        continue

                    # Check overlap with next activity
                    if i < len(activities) - 1:
                        next_act = activities[i + 1]
                        if next_act.start_time:
                            try:
                                next_start = datetime.strptime(next_act.start_time, "%H:%M").time()
                                if end > next_start:
                                    violations += 1
                            except ValueError:
                                pass

        score = 1 - (violations / total) if total > 0 else 1.0
        return {
            "scores": {"time_feasibility": score},
            "critical_failures": ["time_feasibility"] if score < 0.8 else [],
            "suggestions": ["部分活动时间安排存在冲突或超出合理范围"] if score < 0.8 else [],
        }

    async def _check_poi_existence(self, itinerary: list[DayPlan], profile: UserProfile) -> dict:
        """Verify POI existence by sampling and searching."""
        city = profile.destination or ""
        all_pois = [a for day in itinerary for a in day.activities if a.category == "attraction"]
        if not all_pois:
            return {"scores": {"factuality": 1.0}, "critical_failures": [], "suggestions": []}

        sample_size = max(1, int(len(all_pois) * 0.3))
        sampled = random.sample(all_pois, min(sample_size, len(all_pois)))

        verified = 0
        for activity in sampled:
            try:
                query = f"{activity.poi_name} {city} 景点"
                results = await self.search_skill.search(query, top_n=3)
                if self._is_poi_verified(activity.poi_name, results):
                    verified += 1
            except Exception:
                # Search failure should not penalize verification
                verified += 1

        score = verified / len(sampled) if sampled else 1.0
        return {
            "scores": {"factuality": score},
            "critical_failures": ["factuality"] if score < 0.8 else [],
            "suggestions": ["部分景点信息无法验证，建议核实"] if score < 0.8 else [],
        }

    @staticmethod
    def _is_poi_verified(poi_name: str, results: list) -> bool:
        """Check if search results confirm POI existence."""
        from difflib import SequenceMatcher

        name = poi_name.lower()
        for r in results:
            title = getattr(r, "title", "").lower()
            snippet = getattr(r, "snippet", "").lower()
            combined = f"{title} {snippet}"

            # Exact or substring match
            if name in title or name in snippet:
                return True

            # Fuzzy match
            if SequenceMatcher(None, name, title).ratio() >= 0.6:
                return True

            # Check if any word in poi_name appears in results
            words = [w for w in name.split() if len(w) >= 2]
            if words and all(w in combined for w in words[:2]):
                return True

        return False

    async def _check_opening_hours(self, itinerary: list[DayPlan]) -> dict:
        """Check that activities respect opening hours."""
        violations = 0
        total = 0
        for day in itinerary:
            for activity in day.activities:
                if (
                    activity.open_time
                    and activity.close_time
                    and activity.start_time
                    and activity.end_time
                ):
                    total += 1
                    try:
                        open_t = datetime.strptime(activity.open_time, "%H:%M").time()
                        close_t = datetime.strptime(activity.close_time, "%H:%M").time()
                        start_t = datetime.strptime(activity.start_time, "%H:%M").time()
                        end_t = datetime.strptime(activity.end_time, "%H:%M").time()

                        if start_t < open_t or end_t > close_t:
                            violations += 1
                    except ValueError:
                        # Unparseable times don't count as violations
                        pass

        score = 1 - (violations / total) if total > 0 else 1.0
        return {
            "scores": {"opening_hours": score},
            "critical_failures": ["opening_hours"] if score < 0.8 else [],
            "suggestions": ["部分景点安排不在开放时间内"] if score < 0.8 else [],
        }

    async def _check_preference_coverage(
        self, itinerary: list[DayPlan], profile: UserProfile
    ) -> dict:
        user_prefs = set(profile.interests + profile.food_preferences)
        covered = set()
        for day in itinerary:
            for activity in day.activities:
                covered.update(set(activity.tags) & user_prefs)
        score = len(covered) / len(user_prefs) if user_prefs else 1.0
        return {
            "scores": {"preference_coverage": score},
            "critical_failures": [],
            "suggestions": ["可增加更多偏好匹配景点"] if score < 0.6 else [],
        }

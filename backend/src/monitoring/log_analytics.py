"""Log analytics engine for monitoring planning failures, user intents, and destination satisfaction."""

from datetime import timedelta
from typing import Any

from sqlalchemy import select

from core.clock import utc_now
from models import Itinerary, PlanningJob, UserModificationLog


class LogAnalyticsEngine:
    """Analyze operational logs to surface failure clusters, intent trends, and satisfaction rankings.

    The engine is defensive by design: any missing session or unexpected query error
    returns an empty result set rather than propagating exceptions.

    Note on schema adaptations:
    - ``PlanningJob`` has no native ``city`` column; city is extracted from the
      ``input_requirements`` JSON blob when available.
    - ``UserModificationLog`` has no native ``intent`` column; intent is extracted
      from the ``payload`` JSON blob, falling back to ``action_type``.
    - ``Itinerary`` has no native ``status`` column; every itinerary created within
      the window is treated as a positive outcome.
    """

    _ERROR_MESSAGE_MAX_LENGTH = 200
    _POSITIVE_ITINERARY_STATUSES = {"completed", "itinerary_final", "confirmed"}

    @classmethod
    def _normalize_error_message(cls, error_message: Any) -> str:
        """Strip and truncate an error message for stable grouping."""
        if not error_message:
            return "unknown"
        normalized = str(error_message).strip()
        if not normalized:
            return "unknown"
        return normalized[: cls._ERROR_MESSAGE_MAX_LENGTH]

    @classmethod
    def _extract_city(cls, job: PlanningJob) -> str:
        """Extract city from input_requirements JSON; fall back to 'unknown'."""
        requirements = getattr(job, "input_requirements", None) or {}
        if isinstance(requirements, dict):
            for key in ("city", "destination"):
                value = requirements.get(key)
                if value:
                    city = str(value).strip()
                    if city:
                        return city
        return "unknown"

    @classmethod
    def _extract_intent(cls, log: UserModificationLog) -> str:
        """Extract intent from payload JSON; fall back to action_type."""
        payload = getattr(log, "payload", None) or {}
        if isinstance(payload, dict):
            intent = payload.get("intent")
            if intent:
                intent_str = str(intent).strip()
                if intent_str:
                    return intent_str
        return getattr(log, "action_type", None) or "unknown"

    @classmethod
    async def cluster_planning_failures(cls, db_session, hours: int = 24) -> list[dict[str, Any]]:
        """Cluster recent planning job failures by normalized error message and city.

        Args:
            db_session: An async SQLAlchemy session.
            hours: Look-back window in hours.

        Returns:
            A list of dicts with ``error_message``, ``city``, and ``count``,
            sorted by count descending.
        """
        if db_session is None:
            return []

        try:
            cutoff = utc_now() - timedelta(hours=hours)
            stmt = (
                select(PlanningJob)
                .where(PlanningJob.status == "failed")
                .where(PlanningJob.created_at >= cutoff)
            )
            result = await db_session.execute(stmt)
            rows = result.scalars().all()

            clusters: dict[tuple[str, str], int] = {}
            for row in rows:
                error = cls._normalize_error_message(row.error_message)
                city = cls._extract_city(row)
                clusters[(error, city)] = clusters.get((error, city), 0) + 1

            return sorted(
                [
                    {"error_message": error, "city": city, "count": count}
                    for (error, city), count in clusters.items()
                ],
                key=lambda item: item["count"],
                reverse=True,
            )
        except Exception:
            return []

    @classmethod
    async def top_modification_intents(
        cls, db_session, hours: int = 24, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Rank the most common user modification intents in the look-back window.

        Args:
            db_session: An async SQLAlchemy session.
            hours: Look-back window in hours.
            limit: Maximum number of results to return.

        Returns:
            A list of dicts with ``intent`` and ``count``, sorted by count descending.
        """
        if db_session is None:
            return []

        try:
            cutoff = utc_now() - timedelta(hours=hours)
            stmt = select(UserModificationLog).where(UserModificationLog.created_at >= cutoff)
            result = await db_session.execute(stmt)
            rows = result.scalars().all()

            counts: dict[str, int] = {}
            for row in rows:
                intent = cls._extract_intent(row)
                counts[intent] = counts.get(intent, 0) + 1

            ranked = sorted(
                [{"intent": intent, "count": count} for intent, count in counts.items()],
                key=lambda item: item["count"],
                reverse=True,
            )
            return ranked[:limit]
        except Exception:
            return []

    @classmethod
    async def destination_satisfaction_ranking(
        cls, db_session, days: int = 30, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Rank destinations by satisfaction score over the look-back window.

        Because ``Itinerary`` does not expose a status column, every itinerary in
        the window is counted as a positive outcome.

        Args:
            db_session: An async SQLAlchemy session.
            days: Look-back window in days.
            limit: Maximum number of results to return.

        Returns:
            A list of dicts with ``destination``, ``positive_count``,
            ``total_count``, and ``score``, sorted by score descending.
        """
        if db_session is None:
            return []

        try:
            cutoff = utc_now() - timedelta(days=days)
            stmt = select(Itinerary).where(Itinerary.created_at >= cutoff)
            result = await db_session.execute(stmt)
            rows = result.scalars().all()

            totals: dict[str, int] = {}
            positives: dict[str, int] = {}
            for row in rows:
                destination = str(getattr(row, "destination", "") or "unknown").strip()
                totals[destination] = totals.get(destination, 0) + 1
                positives[destination] = positives.get(destination, 0) + 1

            ranking = []
            for destination, total in totals.items():
                positive = positives.get(destination, 0)
                score = round(positive / total, 4) if total else 0.0
                ranking.append(
                    {
                        "destination": destination,
                        "positive_count": positive,
                        "total_count": total,
                        "score": score,
                    }
                )

            return sorted(ranking, key=lambda item: item["score"], reverse=True)[:limit]
        except Exception:
            return []

    @classmethod
    def generate_iteration_suggestions(cls, failures: list[dict[str, Any]]) -> list[str]:
        """Generate human-readable iteration suggestions from the top failure clusters.

        Args:
            failures: Output from :meth:`cluster_planning_failures`.

        Returns:
            A list of suggestion strings for the top 3 failure clusters.
        """
        suggestions: list[str] = []
        for failure in failures[:3]:
            city = failure.get("city", "未知城市")
            error = failure.get("error_message", "未知错误")
            count = failure.get("count", 0)
            suggestions.append(
                f"{city} 的 {error} 失败次数较多（{count} 次），建议检查相关服务或输入参数。"
            )
        return suggestions

    @classmethod
    async def analyze(cls, db_session) -> dict[str, Any]:
        """Run the full analytics suite.

        Args:
            db_session: An async SQLAlchemy session.

        Returns:
            A dict with keys ``planning_failures``, ``modification_intents``,
            ``destination_ranking``, and ``iteration_suggestions``.
        """
        planning_failures = await cls.cluster_planning_failures(db_session)
        modification_intents = await cls.top_modification_intents(db_session)
        destination_ranking = await cls.destination_satisfaction_ranking(db_session)
        iteration_suggestions = cls.generate_iteration_suggestions(planning_failures)

        return {
            "planning_failures": planning_failures,
            "modification_intents": modification_intents,
            "destination_ranking": destination_ranking,
            "iteration_suggestions": iteration_suggestions,
        }

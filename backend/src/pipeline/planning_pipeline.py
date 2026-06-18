"""Planning pipeline — Phase 2A: Planning Core with embedded constraints.

Replaces monolithic LangGraph planner_node with deterministic algorithm draft
that streams to frontend immediately. Validation/Repair layers are no longer
invoked; the scheduler enforces constraints directly and only runs a lightweight
sanity check.
"""

import asyncio
import json
import logging
from contextlib import suppress
from datetime import datetime
from typing import Any, Optional

from models.planning_job import PlanningJob
from core.metrics import incr
from planner.core import build_strategy, build_schedule, detect_remote_pois
from planner.core.or_scheduler import solve_itinerary_or
from planner.core.daily_scheduler import _sanity_check
from planner.core.models import RuleViolation

logger = logging.getLogger(__name__)


class PlanningPipeline:
    """
    Planning pipeline — business job state managed here.
    Phase 2A: Planning Core (heuristic strategy + algorithm scheduler) + Rule Validator.
    """

    def __init__(self, worker: Any):
        self.worker = worker

    async def run(self, job: PlanningJob):
        """Execute Phase 2A pipeline: data → draft → validate → finalize."""
        from core.thought_logger import (
            set_current_session_id,
            set_current_user_id,
            set_current_user_role,
        )

        feedback = job.user_feedback or {}
        set_current_session_id(job.session_id)
        if job.user_id:
            set_current_user_id(job.user_id)
        set_current_user_role(feedback.get("user_role", "user"))

        try:
            await self._run_core(job)
        except asyncio.CancelledError:
            logger.info("Job %s cancelled", job.id)
            await self._release_job(job, "cancelled")
            raise

    async def _run_core(self, job: PlanningJob):
        """Core pipeline execution."""
        # Mark running
        if await self._check_cancelled(job):
            return
        if not await self._record_stage(job, "running"):
            logger.warning("Job %s lost ownership before running stage", job.id)
            return

        # ------------------------------------------------------------------ #
        # 1. Profile from conversation state (intent already applied in WS)
        # ------------------------------------------------------------------ #
        from core.conversation_turn import user_profile_from_job

        profile = user_profile_from_job(job)
        if not profile.destination:
            await self._release_job(job, "failed", "无法识别目的地")
            return

        if await self._check_cancelled(job):
            return

        from core.cache_keys import itinerary_draft_key
        from core.cache_policy import jitter_ttl
        from core.redis_client import redis_cache_client
        from core.settings import settings
        from planner.core.models import Strategy
        from schemas import DayPlan

        draft_key = itinerary_draft_key(profile)
        cached_draft: dict | None = None
        try:
            cached_draft = await redis_cache_client.get_json(draft_key)
        except Exception:
            pass

        cache_remote_warnings: list[RuleViolation] = []
        if cached_draft and isinstance(cached_draft.get("itinerary_draft"), list):
            logger.info("Itinerary draft cache hit for %s", profile.destination)
            strategy = Strategy(**cached_draft["strategy"])
            itinerary_draft = [DayPlan(**d) for d in cached_draft["itinerary_draft"]]
            fallback_used = bool(cached_draft.get("fallback_used", False))
            draft_payload = cached_draft
            pois: list = []
            if profile.destination:
                from agents.realtime_query import RealtimeQueryAgent

                try:
                    pois = await self._safe_wait_for(
                        RealtimeQueryAgent().query_pois(
                            profile.destination,
                            profile.interests + profile.food_preferences,
                        ),
                        timeout=3.0,
                    )
                except Exception:
                    pois = []

            # B1: cache hit must still run remote detection and filter cross-city POIs
            if pois:
                _, cross_city_names, _ = detect_remote_pois(pois)
                if cross_city_names:
                    for day in itinerary_draft:
                        original_count = len(day.activities)
                        day.activities = [
                            a for a in day.activities if a.poi_name not in cross_city_names
                        ]
                        if original_count > 0 and not day.activities:
                            cache_remote_warnings.append(
                                RuleViolation(
                                    rule="remote_poi_filtered",
                                    severity="soft",
                                    message=f"第{day.day_number}天仅含远程景点，已从缓存行程中移除",
                                    day_number=day.day_number,
                                )
                            )
        else:
            # ------------------------------------------------------------------ #
            # 2. Data collection (POI + weather) with timeout + fallback
            # ------------------------------------------------------------------ #
            from agents.realtime_query import RealtimeQueryAgent
            from schemas import WeatherDay

            query_agent = RealtimeQueryAgent()
            fallback_used = False

            # POI search with 3s timeout
            try:
                pois = await self._safe_wait_for(
                    query_agent.query_pois(
                        profile.destination,
                        profile.interests + profile.food_preferences,
                    ),
                    timeout=3.0,
                )
            except asyncio.TimeoutError:
                logger.warning("POI search timeout for %s, using fallback", profile.destination)
                from skills.city_data import CITY_DEFAULTS

                pois = list(CITY_DEFAULTS.get(profile.destination, []))
                fallback_used = True

            if not fallback_used:
                fallback_used = not pois or all(getattr(p, "is_fallback", True) for p in pois)

            # Weather query with 3s timeout
            weather: list[WeatherDay] = []
            try:
                start, end = _split_dates(profile.travel_dates or "")
                if start:
                    weather = await self._safe_wait_for(
                        query_agent.query_weather(profile.destination, start, end),
                        timeout=3.0,
                    )
            except asyncio.TimeoutError:
                logger.warning("Weather query timeout for %s", profile.destination)
            except Exception as exc:
                logger.warning("Weather query failed for %s: %s", profile.destination, exc)

            # ------------------------------------------------------------------ #
            # 3. Planning Core: OR-Tools CP-SAT (primary) + greedy (fallback)
            # ------------------------------------------------------------------ #
            strategy = build_strategy(pois, profile)

            # Try OR-Tools first
            try:
                itinerary_draft = solve_itinerary_or(
                    pois, profile, must_see=strategy.must_see,
                )
                logger.info("OR-Tools solver used for job %s", job.id)
            except Exception as exc:
                logger.warning(
                    "OR-Tools failed (%s), falling back to greedy scheduler", exc
                )
                itinerary_draft = build_schedule(strategy, pois, weather, profile)

            draft_payload = {
                "itinerary_draft": [day.model_dump() for day in itinerary_draft],
                "strategy": strategy.model_dump(),
                "fallback_used": fallback_used,
            }
            try:
                await redis_cache_client.set_json(
                    draft_key,
                    draft_payload,
                    ttl=jitter_ttl(settings.cache_ttl_itinerary),
                )
            except Exception:
                pass

        # Stream draft immediately (TTFI)
        if not await self._record_stage(job, "draft_ready", draft_payload):
            logger.warning("Job %s lost ownership before draft_ready", job.id)
            return

        if await self._check_cancelled(job):
            return

        # ------------------------------------------------------------------ #
        # 4. Sanity check (no validator/repair)
        # ------------------------------------------------------------------ #
        warnings = _sanity_check(itinerary_draft, profile)
        if cache_remote_warnings:
            warnings.extend([w.message for w in cache_remote_warnings])

        for warning in warnings:
            logger.warning("Job %s sanity warning: %s", job.id, warning)

        # Stream final itinerary
        final_payload = {
            "itinerary_final": [day.model_dump() for day in itinerary_draft],
            "warnings": warnings,
        }
        if not await self._record_stage(job, "itinerary_final", final_payload):
            logger.warning("Job %s lost ownership before itinerary_final", job.id)
            return

        if await self._check_cancelled(job):
            return

        # ------------------------------------------------------------------ #
        # 5. Writer (Phase 2C — LLM-enrich prose, per-activity validation)
        # ------------------------------------------------------------------ #
        from planner.core import enrich

        enriched_itinerary, proposal_text = await enrich(itinerary_draft, profile)

        writing_payload = {
            "proposal_text_preview": proposal_text,
            "itinerary_enriched": [day.model_dump() for day in enriched_itinerary],
        }
        if not await self._record_stage(job, "writing", writing_payload):
            logger.warning("Job %s lost ownership before writing", job.id)
            return

        # Stream proposal text as tokens to Redis for SSE subscribers.
        await self._stream_proposal_tokens(job, proposal_text)

        if await self._check_cancelled(job):
            return

        # ------------------------------------------------------------------ #
        # 6. Completed
        # ------------------------------------------------------------------ #
        await self._release_job(
            job,
            "completed",
            payload={
                "proposal_text": proposal_text,
                "itinerary_final": [day.model_dump() for day in enriched_itinerary],
                "strategy": strategy.model_dump(),
                "warnings": warnings,
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _safe_wait_for(aw, timeout: float):
        """Like asyncio.wait_for but ensures cancelled task cleanup.

        After a wait_for timeout, the cancelled inner task can leave
        asyncpg's connection pool in a stale state. By creating an
        explicit task and awaiting it after cancellation, we ensure
        all callbacks are processed before proceeding.
        """
        task = asyncio.create_task(aw)
        try:
            return await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
            # Yield to event loop so any remaining callbacks drain
            await asyncio.sleep(0)
            raise

    async def _check_cancelled(self, job: PlanningJob) -> bool:
        """Check cancellation; if cancelled, mark and stop."""
        cancelled = await self.worker.check_cancelled(job.id)
        if cancelled:
            await self.worker.mark_cancelled(job.id)
            return True
        return False

    async def _record_stage(
        self, job: PlanningJob, stage: str, payload: Optional[dict] = None
    ) -> bool:
        """Record stage completion."""
        return await self.worker.record_stage(job, stage, payload)

    async def _stream_proposal_tokens(
        self, job: PlanningJob, proposal_text: str
    ) -> None:
        """Publish proposal text chunks to Redis so SSE clients can type them out."""
        from api.chat_runtime import publish_token

        if not proposal_text:
            return
        # Stream in small chunks to create a smooth typing effect.
        chunk_size = 8
        for i in range(0, len(proposal_text), chunk_size):
            chunk = proposal_text[i : i + chunk_size]
            await publish_token(job.id, chunk)
            # Tiny delay keeps the stream visible without blocking the worker.
            await asyncio.sleep(0.01)

    async def _release_job(
        self,
        job: PlanningJob,
        status: str,
        error: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> bool:
        """Release job with final status."""
        from repositories.planning_job import PlanningJobRepository
        from core.database import async_session_maker

        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)
            if payload:
                ok = await repo.update_stage(
                    job.id,
                    status,
                    payload,
                    worker_id=self.worker.worker_id,
                )
                if not ok:
                    await db.rollback()
                    logger.warning(
                        "Job %s lost ownership before %s payload write",
                        job.id,
                        status,
                    )
                    return False

            ok = await repo.release(job.id, self.worker.worker_id, status, error)
            if not ok:
                await db.rollback()
                logger.warning(
                    "Job %s release skipped because worker %s no longer owns it",
                    job.id,
                    self.worker.worker_id,
                )
                return False

            await repo.add_event(
                job_id=job.id,
                stage=status,
                event_type=status if status in ("failed", "cancelled") else "completed",
                payload=payload,
                error=error,
            )
            await db.commit()

        if status == "completed":
            incr("planning_jobs_completed_total")

        # Notify via Redis
        from core.redis_client import redis_client

        elapsed = (datetime.utcnow() - job.created_at).total_seconds()
        try:
            await redis_client._client.publish(
                f"job:status:{job.id}",
                json.dumps(
                    {
                        "type": "stage",
                        "stage": status,
                        "elapsed": round(elapsed, 1),
                        "job_id": job.id,
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception as exc:
            logger.warning(
                "Redis publish failed for job %s after %s (job state committed): %s",
                job.id,
                status,
                exc,
            )
        return True


# --------------------------------------------------------------------------- #
# Helpers (moved from graph/nodes.py)
# --------------------------------------------------------------------------- #


def _split_dates(dates: str) -> tuple[str, str]:
    """Split date range string into start and end dates."""
    if not dates:
        return "", ""

    for sep in [" to ", " ~ ", " - ", " 到 ", "至", "—"]:
        if sep in dates:
            parts = dates.split(sep, 1)
            return parts[0].strip(), parts[1].strip()

    # Single date
    return dates.strip(), dates.strip()

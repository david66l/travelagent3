"""Celery application configuration for background tasks."""

from celery import Celery

from core.settings import settings

celery_app = Celery(
    "travel_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "worker.memory_tasks",
        "worker.planning_tasks",
        "worker.cache_tasks",
        "worker.dlq_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_default_queue=settings.celery_task_default_queue,
    worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
    result_expires=3600,
    broker_transport_options={
        # Redis has no broker-side ack; a message is re-delivered if it is not
        # acked within visibility_timeout. Set it well above the longest task /
        # countdown (now ~30s) so long tasks are never executed twice.
        "visibility_timeout": settings.celery_visibility_timeout_seconds,
    },
    task_routes={
        "worker.planning_tasks.execute_planning_job": {
            "queue": settings.celery_planning_queue,
        },
        "worker.memory_tasks.archive_session": {
            "queue": settings.celery_memory_queue,
        },
        "worker.memory_tasks.schedule_delayed_archive": {
            "queue": settings.celery_memory_queue,
        },
        "worker.memory_tasks.archive_active_sessions": {
            "queue": settings.celery_memory_queue,
        },
        "worker.memory_tasks.cleanup_expired_memories": {
            "queue": settings.celery_memory_queue,
        },
        "worker.memory_tasks.compact_stale_sessions": {
            "queue": settings.celery_memory_queue,
        },
        "worker.memory_tasks.sweep_due_archives": {
            "queue": settings.celery_memory_queue,
        },
    },
    beat_schedule={
        "archive-active-sessions": {
            "task": "worker.memory_tasks.archive_active_sessions",
            "schedule": 300.0,  # every 5 minutes (PRD)
        },
        "cleanup-expired-memories": {
            "task": "worker.memory_tasks.cleanup_expired_memories",
            "schedule": 86400.0,  # daily
        },
        "compact-stale-sessions": {
            "task": "worker.memory_tasks.compact_stale_sessions",
            "schedule": 3600.0,  # every hour
        },
        "sweep-due-archives": {
            "task": "worker.memory_tasks.sweep_due_archives",
            "schedule": 60.0,  # reap sessions past the disconnect grace period
        },
        "warm-top-cities-cache": {
            "task": "worker.cache_tasks.warm_top_cities_cache",
            "schedule": 86400.0,  # daily
        },
        "inspect-planning-dead-letters": {
            "task": "worker.dlq_tasks.inspect_dead_letters",
            "schedule": 86400.0,  # daily
        },
        "archive-stale-dead-letters": {
            "task": "worker.dlq_tasks.archive_stale_dead_letters",
            "schedule": 86400.0,  # daily
        },
    },
)

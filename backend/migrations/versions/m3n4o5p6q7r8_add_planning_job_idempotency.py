"""add per-user planning job idempotency keys

Revision ID: m3n4o5p6q7r8
Revises: l2m3n4o5p6q7
Create Date: 2026-08-30
"""

from alembic import op
import sqlalchemy as sa


revision = "m3n4o5p6q7r8"
down_revision = "l2m3n4o5p6q7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "planning_jobs",
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
    )
    op.create_unique_constraint(
        "uq_planning_jobs_user_idempotency_key",
        "planning_jobs",
        ["user_uuid", "idempotency_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_planning_jobs_user_idempotency_key",
        "planning_jobs",
        type_="unique",
    )
    op.drop_column("planning_jobs", "idempotency_key")

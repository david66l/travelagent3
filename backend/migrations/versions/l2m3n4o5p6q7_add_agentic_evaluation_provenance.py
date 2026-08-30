"""add provenance to paired Agentic evaluation records

Revision ID: l2m3n4o5p6q7
Revises: k1l2m3n4o5p6
Create Date: 2026-08-15
"""

from alembic import op
import sqlalchemy as sa


revision = "l2m3n4o5p6q7"
down_revision = "k1l2m3n4o5p6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agentic_evaluation_records",
        sa.Column(
            "evaluation_source",
            sa.String(length=32),
            server_default="legacy_unknown",
            nullable=False,
        ),
    )
    op.add_column(
        "agentic_evaluation_records",
        sa.Column(
            "deployment_id",
            sa.String(length=64),
            server_default="legacy",
            nullable=False,
        ),
    )
    op.add_column(
        "agentic_evaluation_records",
        sa.Column("batch_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "agentic_evaluation_records",
        sa.Column("source_case_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "agentic_evaluation_records",
        sa.Column(
            "release_gate_eligible",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_agentic_eval_provenance",
        "agentic_evaluation_records",
        ["evaluation_source", "deployment_id", "batch_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_agentic_eval_provenance", table_name="agentic_evaluation_records")
    op.drop_column("agentic_evaluation_records", "release_gate_eligible")
    op.drop_column("agentic_evaluation_records", "source_case_id")
    op.drop_column("agentic_evaluation_records", "batch_id")
    op.drop_column("agentic_evaluation_records", "deployment_id")
    op.drop_column("agentic_evaluation_records", "evaluation_source")

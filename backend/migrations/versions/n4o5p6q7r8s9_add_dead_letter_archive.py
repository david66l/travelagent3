"""add durable dead-letter archive

Revision ID: n4o5p6q7r8s9
Revises: m3n4o5p6q7r8
Create Date: 2026-08-30

Some long-lived development databases created this table through metadata
bootstrap before it was represented in Alembic.  The existence checks keep
this migration safe for those databases while making a fresh install complete.
"""

from alembic import op
import sqlalchemy as sa


revision = "n4o5p6q7r8s9"
down_revision = "m3n4o5p6q7r8"
branch_labels = None
depends_on = None


TABLE_NAME = "dead_letter_archive"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(TABLE_NAME):
        op.create_table(
            TABLE_NAME,
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("task_id", sa.String(length=64), nullable=False),
            sa.Column("task_name", sa.String(length=128), nullable=False),
            sa.Column("job_id", sa.String(length=36), nullable=True),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("exception_type", sa.String(length=256), nullable=True),
            sa.Column("traceback", sa.Text(), nullable=True),
            sa.Column("failed_at", sa.DateTime(), nullable=True),
            sa.Column("archived_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    index_names = {index["name"] for index in sa.inspect(bind).get_indexes(TABLE_NAME)}
    for name, column in (
        ("ix_dead_letter_archive_task_id", "task_id"),
        ("ix_dead_letter_archive_job_id", "job_id"),
        ("ix_dead_letter_archive_archived_at", "archived_at"),
    ):
        if name not in index_names:
            op.create_index(name, TABLE_NAME, [column], unique=False)


def downgrade() -> None:
    op.drop_table(TABLE_NAME)

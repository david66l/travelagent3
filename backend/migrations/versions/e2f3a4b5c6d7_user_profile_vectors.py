"""user_profile_vectors

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
Create Date: 2026-06-18

用户画像向量表 — 长期偏好 embedding + 结构化画像信息。
与 user_profiles 表并行，不修改已有表。
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "e2f3a4b5c6d7"
down_revision: Union[str, Sequence[str], None] = "d1e2f3a4b5c6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_profile_vectors",
        sa.Column("user_id", sa.UUID(), primary_key=True),
        sa.Column("preference_embedding", sa.Text(), nullable=True),  # JSON string of float[1024]
        sa.Column("profile_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("visited_cities", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("favorite_spots", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("avoid_spots", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("liked_foods", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("avoided_foods", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("avg_daily_budget", sa.Numeric(10, 2), nullable=True),
        sa.Column("trip_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_profile_vectors_updated", "user_profile_vectors", ["updated_at"])


def downgrade() -> None:
    op.drop_table("user_profile_vectors")

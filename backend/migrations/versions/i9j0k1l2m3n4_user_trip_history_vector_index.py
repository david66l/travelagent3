"""user_trip_history_vector_index

Revision ID: i9j0k1l2m3n4
Revises: 4114fc8f89a7
Create Date: 2026-06-18

为 user_trip_history.trip_vector 添加 HNSW 索引，支撑基于历史行程向量的相似召回。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "i9j0k1l2m3n4"
down_revision: Union[str, Sequence[str], None] = "4114fc8f89a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_user_trip_history_trip_vector_hnsw
        ON user_trip_history USING hnsw (trip_vector vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_user_trip_history_trip_vector_hnsw")

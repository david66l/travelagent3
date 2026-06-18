"""enable_pgvector

Revision ID: f3a4b5c6d7e8
Revises: e2f3a4b5c6d7
Create Date: 2026-06-18

启用 pgvector 扩展 + 迁移 embedding 列到 vector(1024)。
pgvector 已从源码编译安装（PG16 兼容）。
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "f3a4b5c6d7e8"
down_revision: Union[str, Sequence[str], None] = "e2f3a4b5c6d7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # knowledge_tips: TEXT → vector(1024)
    op.execute(
        "ALTER TABLE knowledge_tips "
        "ALTER COLUMN embedding TYPE vector(1024) USING embedding::vector"
    )

    # user_profile_vectors: TEXT → vector(1024)
    op.execute(
        "ALTER TABLE user_profile_vectors "
        "ALTER COLUMN preference_embedding TYPE vector(1024) USING preference_embedding::vector"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE user_profile_vectors ALTER COLUMN preference_embedding TYPE text")
    op.execute("ALTER TABLE knowledge_tips ALTER COLUMN embedding TYPE text")
    op.execute("DROP EXTENSION IF EXISTS vector")

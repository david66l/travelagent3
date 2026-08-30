"""profile_transport_accommodation

Revision ID: a76ff0cc2276
Revises: g4h5i6j7k8l9
Create Date: 2026-06-18

补全 user_profile_vectors 缺失列.
"""

from alembic import op
import sqlalchemy as sa

revision = "a76ff0cc2276"
down_revision = "g4h5i6j7k8l9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_profile_vectors",
        sa.Column("preferred_transport", sa.String(50), nullable=False, server_default=""),
    )
    op.add_column(
        "user_profile_vectors",
        sa.Column("preferred_accommodation", sa.String(50), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("user_profile_vectors", "preferred_accommodation")
    op.drop_column("user_profile_vectors", "preferred_transport")

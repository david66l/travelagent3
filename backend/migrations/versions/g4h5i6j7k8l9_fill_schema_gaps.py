"""fill_schema_gaps

Revision ID: g4h5i6j7k8l9
Revises: f3a4b5c6d7e8
Create Date: 2026-06-18

补全蓝图缺失:
  - hotels 表
  - attractions.description / peak_hours / best_season 列
  - attractions.walk_intensity → 已有，skip
"""

from alembic import op
import sqlalchemy as sa

revision: str = "g4h5i6j7k8l9"
down_revision = "f3a4b5c6d7e8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # hotels 表
    op.create_table(
        "hotels",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("city", sa.String(50), nullable=False),
        sa.Column("district", sa.String(100), nullable=False, server_default=""),
        sa.Column("price_range", sa.String(20), nullable=False, server_default="mid"),
        sa.Column("has_elevator", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("has_breakfast", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("has_parking", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("child_friendly", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("lat", sa.Float(), nullable=False, server_default="0"),
        sa.Column("lng", sa.Float(), nullable=False, server_default="0"),
        sa.Column("rating", sa.Float(), nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="amap"),
        sa.Column("source_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_hotels_city", "hotels", ["city"])

    # attractions 补充列
    op.add_column("attractions", sa.Column("description", sa.Text(), nullable=True))
    op.add_column("attractions", sa.Column("peak_hours", sa.String(50), nullable=True))
    op.add_column("attractions", sa.Column("best_season", sa.String(50), nullable=True))


def downgrade() -> None:
    op.drop_column("attractions", "best_season")
    op.drop_column("attractions", "peak_hours")
    op.drop_column("attractions", "description")
    op.drop_table("hotels")

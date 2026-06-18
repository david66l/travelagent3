"""prd_schema_completion

Revision ID: h8i9j0k1l2m3
Revises: a76ff0cc2276
Create Date: 2026-06-18

PRD 缺失补充:
  - transport_hub 交通枢纽表
  - city_info 城市基础信息表
  - restaurants.queue_time_min / cancel_policy
  - hotels.cancel_policy / distance_to_center_km
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision = "h8i9j0k1l2m3"
down_revision = "a76ff0cc2276"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 城市基础信息
    op.create_table(
        "city_info",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("city", sa.String(50), nullable=False, unique=True),
        sa.Column("climate", sa.String(100), nullable=False, server_default=""),
        sa.Column("best_season", sa.String(50), nullable=False, server_default=""),
        sa.Column("district_count", sa.Integer(), server_default="0"),
        sa.Column("main_districts", sa.ARRAY(sa.Text()), server_default="{}"),
        sa.Column("transport_hubs", sa.JSON(), server_default="{}"),
        sa.Column("peak_months", sa.ARRAY(sa.Text()), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # 交通枢纽
    op.create_table(
        "transport_hubs",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("city", sa.String(50), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("hub_type", sa.String(20), nullable=False),  # airport/railway/bus/subway
        sa.Column("lat", sa.Float(), nullable=False, server_default="0"),
        sa.Column("lng", sa.Float(), nullable=False, server_default="0"),
        sa.Column("lines", sa.ARRAY(sa.Text()), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_transport_hubs_city", "transport_hubs", ["city"])

    # 餐厅补充列
    op.add_column("restaurants", sa.Column("queue_time_min", sa.Integer(), nullable=True))
    op.add_column("restaurants", sa.Column("cancel_policy", sa.String(50), nullable=True))

    # 酒店补充列
    op.add_column("hotels", sa.Column("cancel_policy", sa.String(100), nullable=True))
    op.add_column("hotels", sa.Column("distance_to_center_km", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("hotels", "distance_to_center_km")
    op.drop_column("hotels", "cancel_policy")
    op.drop_column("restaurants", "cancel_policy")
    op.drop_column("restaurants", "queue_time_min")
    op.drop_table("transport_hubs")
    op.drop_table("city_info")

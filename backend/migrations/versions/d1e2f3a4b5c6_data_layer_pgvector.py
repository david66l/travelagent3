"""data_layer_pgvector

Revision ID: d1e2f3a4b5c6
Revises: 6ccb368c4300
Create Date: 2026-06-18

新增数据层四表 + pgvector 扩展：
  - attractions:    景点结构化数据
  - restaurants:     餐饮结构化数据
  - knowledge_tips:  攻略向量知识库 (pgvector)
  - data_audit_log:  数据质量审计日志
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
# from pgvector.sqlalchemy import Vector  # TODO: enable after pgvector installed

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, Sequence[str], None] = "6ccb368c4300"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # pgvector 扩展需先 brew install pgvector，暂时跳过
    # op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── 景点表 ──
    op.create_table(
        "attractions",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("city", sa.String(50), nullable=False),
        sa.Column("category", sa.String(50), nullable=False, server_default="attraction"),
        # 动态字段（API 日更）
        sa.Column("ticket_price", sa.Numeric(10, 2), nullable=True),
        sa.Column("open_time", sa.Time(), nullable=True),
        sa.Column("close_time", sa.Time(), nullable=True),
        # 静态字段
        sa.Column("lat", sa.Float(), nullable=False, server_default="0"),
        sa.Column("lng", sa.Float(), nullable=False, server_default="0"),
        sa.Column("address", sa.String(500), nullable=False, server_default=""),
        sa.Column("duration_minutes", sa.Integer(), nullable=False, server_default="120"),
        sa.Column("need_reservation", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("wheelchair_accessible", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("walk_intensity", sa.Integer(), nullable=False, server_default="3"),
        # 标签
        sa.Column("tags", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("suitable_for", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        # 数据治理
        sa.Column("source", sa.String(20), nullable=False, server_default="amap"),
        sa.Column("source_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("review_status", sa.String(20), nullable=False, server_default="verified"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_attractions_city", "attractions", ["city"])
    op.create_index("ix_attractions_category", "attractions", ["category"])
    op.create_index("ix_attractions_name_city", "attractions", ["name", "city"], unique=True)

    # ── 餐饮表 ──
    op.create_table(
        "restaurants",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("city", sa.String(50), nullable=False),
        sa.Column("cuisine", sa.String(50), nullable=True),
        sa.Column("avg_price", sa.Numeric(10, 2), nullable=True),
        sa.Column("open_time", sa.Time(), nullable=True),
        sa.Column("close_time", sa.Time(), nullable=True),
        sa.Column("lat", sa.Float(), nullable=False, server_default="0"),
        sa.Column("lng", sa.Float(), nullable=False, server_default="0"),
        sa.Column("address", sa.String(500), nullable=False, server_default=""),
        sa.Column("tags", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("signature_dishes", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("rating", sa.Float(), nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="amap"),
        sa.Column("source_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_restaurants_city", "restaurants", ["city"])
    op.create_index("ix_restaurants_name_city", "restaurants", ["name", "city"], unique=True)

    # ── 攻略向量表 ──
    op.create_table(
        "knowledge_tips",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("city", sa.String(50), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(20), nullable=False, server_default="guide"),
        sa.Column("walk_intensity", sa.Integer(), nullable=True),
        sa.Column("suitable_for", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column(
            "embedding", sa.Text(), nullable=True
        ),  # TODO: Vector(1024) after pgvector installed
        sa.Column("source", sa.String(20), nullable=False, server_default="manual"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_knowledge_tips_city", "knowledge_tips", ["city"])
    # pgvector IVFFlat 索引（等数据量 > 1000 后再建更有效）
    # op.execute(
    #     "CREATE INDEX ix_knowledge_tips_embedding ON knowledge_tips "
    #     "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    # )

    # ── 数据审计日志 ──
    op.create_table(
        "data_audit_log",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("table_name", sa.String(50), nullable=False),
        sa.Column("record_id", sa.UUID(), nullable=True),
        sa.Column("field", sa.String(50), nullable=False),
        sa.Column("reported_value", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_data_audit_log_status", "data_audit_log", ["status"])


def downgrade() -> None:
    op.drop_table("data_audit_log")
    op.drop_table("knowledge_tips")
    op.drop_table("restaurants")
    op.drop_table("attractions")
    op.execute("DROP EXTENSION IF EXISTS vector")

"""fix attraction rating and portable full-text search

Revision ID: j0k1l2m3n4o5
Revises: i9j0k1l2m3n4
Create Date: 2026-08-11

The retrieval repository orders attractions by ``rating`` but the original
table migration never created that column. It also referenced a non-standard
PostgreSQL text-search configuration named ``chinese``. Use PostgreSQL's
built-in ``simple`` configuration, which tokenises CJK text without requiring
an undeclared extension, and keep ``search_vector`` in sync on writes.
"""

from alembic import op
import sqlalchemy as sa


revision = "j0k1l2m3n4o5"
down_revision = "i9j0k1l2m3n4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("attractions", sa.Column("rating", sa.Float(), nullable=True))
    op.execute(
        """
        UPDATE attractions
        SET search_vector = to_tsvector(
            'simple',
            coalesce(name, '') || ' ' ||
            coalesce(description, '') || ' ' ||
            coalesce(array_to_string(tags, ' '), '')
        )
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION attractions_search_vector_update()
        RETURNS trigger AS $$
        BEGIN
            NEW.search_vector := to_tsvector(
                'simple',
                coalesce(NEW.name, '') || ' ' ||
                coalesce(NEW.description, '') || ' ' ||
                coalesce(array_to_string(NEW.tags, ' '), '')
            );
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_attractions_search_vector
        BEFORE INSERT OR UPDATE OF name, description, tags ON attractions
        FOR EACH ROW EXECUTE FUNCTION attractions_search_vector_update()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_attractions_search_vector ON attractions")
    op.execute("DROP FUNCTION IF EXISTS attractions_search_vector_update()")
    op.drop_column("attractions", "rating")

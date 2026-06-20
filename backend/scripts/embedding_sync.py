"""Batch sync embeddings for attractions.

Usage:
    cd backend
    uv run python scripts/embedding_sync.py [--city CITY] [--batch-size N]

Reads attractions with empty description_vector, encodes description via
bge-large-zh-v1.5, and writes vectors back to PostgreSQL.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from sqlalchemy import text

from core.database import async_session_maker
from data.embedding import get_embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def sync_attraction_embeddings(
    *,
    city: str | None = None,
    batch_size: int = 64,
) -> dict[str, int]:
    """Sync description embeddings for attractions in batches."""
    embedder = await get_embedder()

    where_clause = "WHERE description_vector IS NULL AND description IS NOT NULL AND status != 'deprecated'"
    params: dict[str, object] = {}
    if city:
        where_clause += " AND city = :city"
        params["city"] = city

    updated = 0
    skipped = 0

    async with async_session_maker() as db:
        while True:
            result = await db.execute(
                text(
                    f"""
                    SELECT id, name, city, description
                    FROM attractions
                    {where_clause}
                    ORDER BY id
                    LIMIT :limit
                    """
                ),
                {**params, "limit": batch_size},
            )
            rows = result.mappings().all()
            if not rows:
                break

            ids = []
            embeddings = []
            for row in rows:
                description = (row["description"] or "").strip()
                if not description:
                    skipped += 1
                    continue
                ids.append(str(row["id"]))
                embeddings.append(embedder.encode_single(description))

            if ids:
                for aid, emb in zip(ids, embeddings):
                    await db.execute(
                        text(
                            "UPDATE attractions SET description_vector = :embedding::vector WHERE id = :id"
                        ),
                        {"embedding": emb, "id": aid},
                    )
                await db.commit()
                updated += len(ids)
                logger.info("Synced %d attraction embeddings (skipped %d)", updated, skipped)

            if len(rows) < batch_size:
                break

    return {"updated": updated, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync attraction embeddings to pgvector")
    parser.add_argument("--city", type=str, help="Only sync attractions for this city")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding")
    args = parser.parse_args()

    stats = asyncio.run(sync_attraction_embeddings(city=args.city, batch_size=args.batch_size))
    logger.info("Done. updated=%(updated)d skipped=%(skipped)d", stats)


if __name__ == "__main__":
    main()

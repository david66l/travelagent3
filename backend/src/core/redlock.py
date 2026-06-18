"""Redlock-style distributed lock across one or more Redis masters."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import redis.asyncio as aioredis

from core.settings import settings

logger = logging.getLogger(__name__)

_RELEASE_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class Redlock:
    """Minimal Redlock quorum lock (works with a single Redis in dev)."""

    def __init__(self, urls: Optional[list[str]] = None):
        self._urls = urls or settings.redis_redlock_url_list
        self._clients: list[aioredis.Redis] = []

    async def connect(self) -> None:
        if self._clients:
            return
        for url in self._urls:
            client = await aioredis.from_url(url, decode_responses=True)
            self._clients.append(client)

    async def disconnect(self) -> None:
        for client in self._clients:
            await client.close()
        self._clients.clear()

    async def acquire(
        self,
        resource: str,
        ttl_seconds: int,
        *,
        blocking: bool = True,
        blocking_timeout: float = 2.0,
    ) -> Optional[str]:
        if not self._clients:
            await self.connect()
        token = uuid.uuid4().hex
        quorum = len(self._clients) // 2 + 1
        deadline = asyncio.get_event_loop().time() + blocking_timeout

        while True:
            acquired = 0
            for client in self._clients:
                try:
                    ok = await client.set(resource, token, nx=True, ex=ttl_seconds)
                    if ok:
                        acquired += 1
                except Exception:
                    logger.debug("Redlock acquire failed on one node for %s", resource)

            if acquired >= quorum:
                return token

            if not blocking:
                return None
            if asyncio.get_event_loop().time() >= deadline:
                return None
            await asyncio.sleep(0.05)

    async def release(self, resource: str, token: str) -> None:
        if not self._clients:
            return
        for client in self._clients:
            try:
                await client.eval(_RELEASE_LUA, 1, resource, token)
            except Exception:
                logger.debug("Redlock release failed on one node for %s", resource)

    async def extend(self, resource: str, token: str, ttl_seconds: int) -> bool:
        if not self._clients:
            return False
        extended = 0
        quorum = len(self._clients) // 2 + 1
        for client in self._clients:
            try:
                raw = await client.get(resource)
                if raw == token:
                    await client.expire(resource, ttl_seconds)
                    extended += 1
            except Exception:
                pass
        return extended >= quorum


redlock = Redlock()

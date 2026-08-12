"""
BGE Embedding Service — bge-large-zh-v1.5 向量化。

首次加载会从 HuggingFace 下载模型（~1.3GB），之后缓存在本地。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_embedder: Optional["BGEEmbedder"] = None
_embedder_failed: bool = False
_embedder_lock: Optional[asyncio.Lock] = None


class BGEEmbedder:
    """BGE 中文 Embedding 模型封装。

    选型: BAAI/bge-large-zh-v1.5
    维度: 1024
    协议: MIT
    """

    MODEL_NAME = "BAAI/bge-large-zh-v1.5"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading BGE model: %s", self.MODEL_NAME)
        try:
            # Load strictly from the local HF cache. The default path triggers an
            # online metadata check on every cold load, which hangs in httpx
            # backoff (~45s) and then fails with "client has been closed",
            # disabling vector search and wasting minutes on every plan.
            self._model = SentenceTransformer(self.MODEL_NAME, local_files_only=True)
        except Exception:
            logger.warning("BGE model not found in local cache; downloading once from HuggingFace")
            self._model = SentenceTransformer(self.MODEL_NAME)
        self._dim = 1024

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        """编码文本列表为向量。"""
        if isinstance(texts, str):
            texts = [texts]
        # normalize_embeddings=True for cosine similarity
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        """编码单条文本。"""
        return self.encode([text])[0]

    async def aencode_single(self, text: str) -> list[float]:
        """异步编码单条文本——把同步 CPU 推理放到线程，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.encode_single, text)


async def get_embedder() -> BGEEmbedder:
    """获取全局 Embedder 实例（懒加载，单例）。

    A failed load is cached so callers fail fast instead of re-attempting the
    expensive model load (and any network backoff) on every request.

    The 1.3GB BGE model load takes ~10s; it runs in a thread so the event loop
    stays responsive, and an asyncio.Lock prevents concurrent requests from each
    kicking off their own (duplicate) load on a cold start. Prefer calling
    ``warmup_embedder`` at startup so the first user request hits a warm cache.
    """
    global _embedder, _embedder_failed, _embedder_lock
    from core.settings import settings

    if settings.embedding_provider == "disabled":
        raise RuntimeError("Vector embeddings are disabled; using structured/lexical RAG")
    if _embedder is not None:
        return _embedder
    if _embedder_failed:
        raise RuntimeError("Embedder previously failed to load; vector search disabled")

    if _embedder_lock is None:
        _embedder_lock = asyncio.Lock()
    async with _embedder_lock:
        # Re-check inside the lock: another coroutine may have loaded it while we waited.
        if _embedder is not None:
            return _embedder
        if _embedder_failed:
            raise RuntimeError("Embedder previously failed to load; vector search disabled")
        try:
            _embedder = await asyncio.to_thread(BGEEmbedder)
        except Exception:
            _embedder_failed = True
            logger.exception("Failed to load embedder; vector search will be skipped")
            raise
    return _embedder


async def warmup_embedder() -> None:
    """Preload the BGE model at startup so the first plan request isn't blocked.

    Best-effort: a failure here is logged and swallowed (vector search will fall
    back), so a missing model never blocks application startup.
    """
    from core.settings import settings

    if settings.embedding_provider == "disabled":
        logger.info("Vector embeddings disabled; structured/lexical RAG is active")
        return
    try:
        await get_embedder()
        logger.info("BGE embedder warmed up and ready")
    except Exception:
        logger.warning("BGE embedder warmup failed; vector search will use fallback")

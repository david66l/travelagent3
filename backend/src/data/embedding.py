"""
BGE Embedding Service — bge-large-zh-v1.5 向量化。

首次加载会从 HuggingFace 下载模型（~1.3GB），之后缓存在本地。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_embedder: Optional["BGEEmbedder"] = None


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


async def get_embedder() -> BGEEmbedder:
    """获取全局 Embedder 实例（懒加载）。"""
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
    return _embedder

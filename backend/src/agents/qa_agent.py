"""Q&A Agent - answer user questions using RAG + web search."""

from core.llm_client import llm
from schemas import ScoredPOI
from skills.web_search import WebSearchSkill


class QAAgent:
    """Answer user questions about destinations, attractions, food, etc."""

    def __init__(self):
        self.search_skill = WebSearchSkill()

    async def answer(self, question: str, city: str | None = None) -> str:
        """Answer user question using RAG + web search."""
        # Search for relevant information
        query = f"{city} {question}" if city else question
        search_results = await self.search_skill.search(query, top_n=5)

        # Build context from search results
        context = "\n".join(
            f"[{i+1}] {r.title}: {r.snippet}"
            for i, r in enumerate(search_results[:3])
        )

        prompt = f"""你是旅行顾问。基于以下搜索结果回答用户问题。

搜索结果：
{context}

用户问题：{question}

要求：
1. 回答要准确、实用
2. 如果搜索结果不够，坦诚告知
3. 推荐具体、可操作
4. 语气友好，像当地朋友

直接输出回答："""

        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.strip()

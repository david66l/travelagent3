"""Optional LLM strategy enhancement — non-blocking, timeout-guarded.

If the LLM call exceeds 3s, the enhancement is skipped and the heuristic
strategy is used as-is.
"""

import asyncio
from typing import Optional

from schemas import ScoredPOI, UserProfile
from planner.core.models import Strategy


async def enhance_strategy(
    heuristic: Strategy,
    pois: list[ScoredPOI],
    profile: UserProfile,
    timeout: float = 3.0,
) -> Optional[Strategy]:
    """Attempt to enhance heuristic strategy with LLM.

    Returns enhanced Strategy if completed within timeout, else None.
    """
    try:
        return await asyncio.wait_for(
            _llm_enhance(heuristic, pois, profile),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return None


async def _llm_enhance(
    heuristic: Strategy,
    pois: list[ScoredPOI],
    profile: UserProfile,
) -> Strategy:
    """LLM call to refine day themes and suggest improvements.

    Currently a placeholder that returns the heuristic strategy unchanged.
    In production, this would call llm.json_chat() with a small prompt.
    """
    # TODO: Implement actual LLM enhancement when needed.
    # For Phase 2A, we ship with heuristic-only to guarantee TTFI.
    await asyncio.sleep(0)  # yield control
    return heuristic

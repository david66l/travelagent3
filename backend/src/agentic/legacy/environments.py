"""Archived pre-react TRL environments and factory branches.

The controller_first / policy_driven execution modes were retired when the
production runtime collapsed to the single react mode; this module keeps the
removed wiring runnable for ablation reproduction.  ``TRLPolicyDrivenEnvironment``
itself stays in ``agentic.trl_environment`` because the react environments
reuse its action methods as donors.
"""

from __future__ import annotations

from typing import Any, Callable

from agentic.trl_environment import (
    TRLClarificationEnvironment,
    TRLPolicyDrivenEnvironment,
    TRLTradeoffEnvironment,
    _TRLTravelEnvironmentBase,
)


class TRLSearchEnvironment(_TRLTravelEnvironmentBase):
    """Controller-first baseline exposing only the delegated search decision."""

    def __init__(self, *, audit_enabled: bool = True) -> None:
        super().__init__(audit_enabled=audit_enabled, execution_mode="controller_first")

    def search_pois(
        self,
        keywords: list[str] | None = None,
    ) -> str:
        """Search POIs using grounded preferences.

        Args:
            keywords: Grounded preference keywords.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("search_pois", {"keywords": keywords or []})


def build_legacy_environment_factories(
    execution_mode: str,
) -> dict[str, Callable[..., _TRLTravelEnvironmentBase]]:
    """Archived factory branches for the retired execution modes."""
    if execution_mode == "policy_driven":
        return {
            "search": _ArchivedPolicyDrivenEnvironment,
            "search_current": _ArchivedPolicyDrivenEnvironment,
            "search_transport": _ArchivedPolicyDrivenEnvironment,
            "clarification": _ArchivedPolicyDrivenEnvironment,
            "tradeoff": _ArchivedPolicyDrivenEnvironment,
        }
    if execution_mode == "controller_first":
        return {
            "search": TRLSearchEnvironment,
            "search_current": TRLSearchEnvironment,
            "search_transport": TRLSearchEnvironment,
            "clarification": TRLClarificationEnvironment,
            "tradeoff": TRLTradeoffEnvironment,
        }
    raise ValueError(f"not a legacy execution mode: {execution_mode}")


class _ArchivedPolicyDrivenEnvironment(TRLPolicyDrivenEnvironment):
    """policy_driven default restored for archived-mode reproduction."""

    def __init__(self, *, audit_enabled: bool = True, **kwargs: Any) -> None:
        kwargs.setdefault("execution_mode", "policy_driven")
        super().__init__(audit_enabled=audit_enabled, **kwargs)


# Archived module-level default of the retired policy_driven era.
TRLTravelEnvironment = _ArchivedPolicyDrivenEnvironment
TRL_ENVIRONMENT_FACTORIES: dict[str, Callable[..., _TRLTravelEnvironmentBase]] = (
    build_legacy_environment_factories("policy_driven")
)

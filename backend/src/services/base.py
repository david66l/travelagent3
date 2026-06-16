"""Base class for all application services."""

from abc import ABC


class BaseService(ABC):
    """Minimal service base for application services.

    Services are stateless wrappers around repositories and infrastructure
    clients.  Keeping the constructor lightweight makes them easy to build in
    dependency providers and easy to mock in unit tests.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

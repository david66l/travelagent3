"""Application-level exceptions.

All custom exceptions inherit from AppException so that the global error
handler can map them to a consistent HTTP response shape.
"""

from typing import Any, Optional


class AppException(Exception):
    """Base application exception.

    Attributes:
        status_code: HTTP status code to return.
        code: Machine-readable error code.
        message: Human-readable error message.
        details: Optional additional context for debugging.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Any] = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


class NotFoundException(AppException):
    """Resource not found."""

    def __init__(self, resource: str = "Resource", identifier: Any = None):
        detail = f"{resource} not found"
        if identifier is not None:
            detail = f"{resource} {identifier} not found"
        super().__init__(404, "NOT_FOUND", detail, {"resource": resource, "id": identifier})


class ValidationException(AppException):
    """Input validation failed."""

    def __init__(self, message: str = "Validation failed", details: Optional[Any] = None):
        super().__init__(422, "VALIDATION_ERROR", message, details)


class ConflictException(AppException):
    """Resource conflict."""

    def __init__(self, message: str = "Conflict", details: Optional[Any] = None):
        super().__init__(409, "CONFLICT", message, details)


class UnauthorizedException(AppException):
    """Authentication required or invalid credentials."""

    def __init__(self, message: str = "Unauthorized", code: str = "UNAUTHORIZED"):
        super().__init__(401, code, message)


class ForbiddenException(AppException):
    """Permission denied."""

    def __init__(self, message: str = "Forbidden", code: str = "FORBIDDEN"):
        super().__init__(403, code, message)


class RateLimitException(AppException):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(
            429,
            "RATE_LIMIT_EXCEEDED",
            message,
            {"retry_after": retry_after},
        )

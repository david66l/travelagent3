"""Global exception handler registration."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.exceptions import AppException
from core.responses import error_response

logger = logging.getLogger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers on the FastAPI app."""

    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException):
        """Handle application exceptions."""
        response = error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )
        if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            retry_after = None
            if isinstance(exc.details, dict):
                retry_after = exc.details.get("retry_after")
            if retry_after is not None:
                response.headers["Retry-After"] = str(retry_after)
        return response

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(request: Request, exc: RequestValidationError):
        """Handle FastAPI/Pydantic validation errors."""
        return error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details=exc.errors(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions."""
        return error_response(
            status_code=exc.status_code,
            code="HTTP_ERROR",
            message=str(exc.detail),
        )

    @app.exception_handler(SQLAlchemyError)
    async def handle_db_exception(request: Request, exc: SQLAlchemyError):
        """Handle database errors without leaking internals."""
        logger.exception("Database error during request %s", request.url.path)
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_ERROR",
            message="An internal error occurred",
        )

    @app.exception_handler(Exception)
    async def handle_generic_exception(request: Request, exc: Exception):
        """Catch-all handler for unexpected errors."""
        logger.exception("Unhandled error during request %s", request.url.path)
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_ERROR",
            message="An unexpected error occurred",
        )

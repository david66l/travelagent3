"""Tests for global exception handlers."""

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from core.exceptions import (
    AppException,
    ConflictException,
    NotFoundException,
    RateLimitException,
    UnauthorizedException,
    ValidationException,
)
from middleware.error_handler import setup_exception_handlers


@pytest.fixture
def app():
    app = FastAPI()
    setup_exception_handlers(app)
    return app


def _request():
    scope = {"type": "http", "method": "GET", "path": "/test", "headers": []}
    return Request(scope)


class TestAppExceptionHandler:
    @pytest.fixture(autouse=True)
    def setup_handler(self, app):
        self.handler = app.exception_handlers[AppException]

    async def test_not_found_exception(self):
        exc = NotFoundException("Conversation", "abc-123")
        response = await self.handler(_request(), exc)
        assert response.status_code == 404
        assert b"NOT_FOUND" in response.body

    async def test_validation_exception(self):
        exc = ValidationException("invalid")
        response = await self.handler(_request(), exc)
        assert response.status_code == 422
        assert b"VALIDATION_ERROR" in response.body

    async def test_conflict_exception(self):
        exc = ConflictException("duplicate")
        response = await self.handler(_request(), exc)
        assert response.status_code == 409
        assert b"CONFLICT" in response.body

    async def test_unauthorized_exception(self):
        exc = UnauthorizedException()
        response = await self.handler(_request(), exc)
        assert response.status_code == 401
        assert b"UNAUTHORIZED" in response.body

    async def test_rate_limit_exception(self):
        exc = RateLimitException(retry_after=60)
        response = await self.handler(_request(), exc)
        assert response.status_code == 429
        assert b"RATE_LIMIT_EXCEEDED" in response.body


class TestValidationErrorHandler:
    async def test_request_validation_error(self, app):
        handler = app.exception_handlers[RequestValidationError]
        exc = RequestValidationError(
            errors=[{"loc": ["body", "x"], "msg": "err", "type": "value_error"}]
        )
        response = await handler(_request(), exc)
        assert response.status_code == 422
        assert b"VALIDATION_ERROR" in response.body


class TestHTTPExceptionHandler:
    async def test_starlette_http_exception(self, app):
        handler = app.exception_handlers[StarletteHTTPException]
        exc = StarletteHTTPException(status_code=418, detail="I'm a teapot")
        response = await handler(_request(), exc)
        assert response.status_code == 418
        assert b"HTTP_ERROR" in response.body
        assert b"teapot" in response.body


class TestDatabaseErrorHandler:
    async def test_sqlalchemy_error(self, app):
        handler = app.exception_handlers[SQLAlchemyError]
        exc = SQLAlchemyError("connection refused")
        response = await handler(_request(), exc)
        assert response.status_code == 500
        assert b"INTERNAL_ERROR" in response.body


class TestCatchAllHandler:
    async def test_generic_exception(self, app):
        handler = app.exception_handlers[Exception]
        exc = RuntimeError("unexpected")
        response = await handler(_request(), exc)
        assert response.status_code == 500
        assert b"INTERNAL_ERROR" in response.body

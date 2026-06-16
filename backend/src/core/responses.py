"""Unified API response helpers."""

from typing import Any, Generic, Optional, TypeVar

from fastapi import status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

DataT = TypeVar("DataT")


class ApiResponse(BaseModel, Generic[DataT]):
    """Standard envelope for all API responses.

    Every successful response follows this shape so clients can rely on a
    consistent structure regardless of endpoint.
    """

    success: bool = True
    code: str = "OK"
    message: str = "Success"
    data: Optional[DataT] = None
    meta: Optional[dict] = None

    model_config = ConfigDict(from_attributes=True)


class ErrorDetail(BaseModel):
    """Error detail payload."""

    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    success: bool = False
    error: ErrorDetail


def success_response(
    data: Any = None,
    *,
    message: str = "Success",
    code: str = "OK",
    meta: Optional[dict] = None,
    status_code: int = status.HTTP_200_OK,
) -> JSONResponse:
    """Build a successful JSON response."""
    body = ApiResponse(success=True, code=code, message=message, data=data, meta=meta)
    return JSONResponse(
        content=body.model_dump(exclude_none=True, mode="json"),
        status_code=status_code,
    )


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
) -> JSONResponse:
    """Build an error JSON response."""
    body = ErrorResponse(
        success=False,
        error=ErrorDetail(code=code, message=message, details=details),
    )
    return JSONResponse(
        content=body.model_dump(exclude_none=True, mode="json"),
        status_code=status_code,
    )

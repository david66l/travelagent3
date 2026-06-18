package middleware

import (
	"net/http"

	"github.com/labstack/echo/v4"
)

// ErrorCode represents a stable error code returned to clients.
type ErrorCode string

const (
	CodeAuthMissing         ErrorCode = "AUTH_MISSING"
	CodeTokenExpired        ErrorCode = "TOKEN_EXPIRED"
	CodeTokenInvalid        ErrorCode = "TOKEN_INVALID"
	CodeTokenRevoked        ErrorCode = "TOKEN_REVOKED"
	CodeDeviceMismatch      ErrorCode = "DEVICE_MISMATCH"
	CodeForbidden           ErrorCode = "FORBIDDEN"
	CodeRateLimited         ErrorCode = "RATE_LIMITED"
	CodeSSELimit            ErrorCode = "SSE_LIMIT"
	CodeCircuitOpen         ErrorCode = "CIRCUIT_OPEN"
	CodeUpstreamUnavailable ErrorCode = "UPSTREAM_UNAVAILABLE"
)

// ErrorResponse is the standardized JSON error body.
type ErrorResponse struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// HTTPError is the gateway-specific error type carrying status and code.
type HTTPError struct {
	Status  int
	Code    ErrorCode
	Message string
}

// Error implements the error interface.
func (e *HTTPError) Error() string {
	return e.Message
}

// NewHTTPError constructs an HTTPError.
func NewHTTPError(status int, code ErrorCode, message string) *HTTPError {
	return &HTTPError{Status: status, Code: code, Message: message}
}

// ErrorHandler returns a custom Echo error handler that emits PRD-compliant bodies.
func ErrorHandler() echo.HTTPErrorHandler {
	return func(err error, c echo.Context) {
		if c.Response().Committed {
			return
		}

		var he *HTTPError
		if errorsAs(err, &he) {
			_ = c.JSON(he.Status, ErrorResponse{Code: string(he.Code), Message: he.Message})
			return
		}

		var echoErr *echo.HTTPError
		if errorsAs(err, &echoErr) {
			code := "INTERNAL_ERROR"
			msg := "服务器内部错误"
			if m, ok := echoErr.Message.(string); ok {
				msg = m
			}
			_ = c.JSON(echoErr.Code, ErrorResponse{Code: code, Message: msg})
			return
		}

		_ = c.JSON(http.StatusInternalServerError, ErrorResponse{
			Code:    "INTERNAL_ERROR",
			Message: "服务器内部错误",
		})
	}
}

// errorsAs is a small helper that avoids importing errors for a single call.
func errorsAs(err error, target interface{}) bool {
	if err == nil {
		return false
	}
	// Use the standard library's errors.As via type assertion for HTTPError.
	if he, ok := err.(*HTTPError); ok {
		if t, ok := target.(**HTTPError); ok {
			*t = he
			return true
		}
	}
	if ee, ok := err.(*echo.HTTPError); ok {
		if t, ok := target.(**echo.HTTPError); ok {
			*t = ee
			return true
		}
	}
	return false
}

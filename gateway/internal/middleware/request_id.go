package middleware

import (
	"context"

	"github.com/google/uuid"
	"github.com/labstack/echo/v4"
)

const requestIDHeader = "X-Request-ID"

// RequestID attaches a unique request id to every request and propagates it via context.
func RequestID() echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			requestID := c.Request().Header.Get(requestIDHeader)
			if requestID == "" {
				requestID = uuid.Must(uuid.NewV7()).String()
			}
			c.Request().Header.Set(requestIDHeader, requestID)
			c.Response().Header().Set(requestIDHeader, requestID)

			ctx := WithRequestID(c.Request().Context(), requestID)
			c.SetRequest(c.Request().WithContext(ctx))

			return next(c)
		}
	}
}

// contextKey avoids collisions with other context keys.
type requestIDKey struct{}

// WithRequestID stores the request id in context.
func WithRequestID(ctx context.Context, requestID string) context.Context {
	return context.WithValue(ctx, requestIDKey{}, requestID)
}

// RequestIDFrom extracts the request id from context.
func RequestIDFrom(ctx context.Context) string {
	v := ctx.Value(requestIDKey{})
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

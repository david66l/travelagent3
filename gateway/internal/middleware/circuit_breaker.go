package middleware

import (
	"net/http"

	"github.com/labstack/echo/v4"

	"github.com/travelagent/gateway/internal/breaker"
)

// CircuitBreaker protects upstream calls with a circuit breaker.
// It records success/failure based on the proxied response status.
func CircuitBreaker(b breaker.Breaker) echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			if !b.Allow() {
				return NewHTTPError(http.StatusServiceUnavailable, CodeCircuitOpen, "服务暂时不可用，请稍后重试")
			}

			err := next(c)
			status := c.Response().Status
			if err != nil {
				// Delegate to Echo error handler; record failure when appropriate.
				if isUpstreamFailure(status) {
					b.RecordFailure()
				}
				return err
			}

			if isUpstreamFailure(status) {
				b.RecordFailure()
			} else {
				b.RecordSuccess()
			}
			return nil
		}
	}
}

func isUpstreamFailure(status int) bool {
	return status >= http.StatusInternalServerError || status == http.StatusServiceUnavailable
}

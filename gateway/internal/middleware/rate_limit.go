package middleware

import (
	"context"
	"net/http"
	"strings"

	"github.com/labstack/echo/v4"

	"github.com/travelagent/gateway/internal/auth"
	"github.com/travelagent/gateway/internal/config"
	"github.com/travelagent/gateway/internal/limit"
)

// IPRateLimit enforces per-IP sliding-window rate limits on every request.
func IPRateLimit(svc limit.Service, cfg config.Config) echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			ctx := c.Request().Context()
			clientIP := c.RealIP()
			if allowed, err := svc.Allow(ctx, limit.IPKey(clientIP), 60, cfg.RateLimitIP); err != nil {
				c.Logger().Errorf("ip rate limit redis error: %v", err)
			} else if !allowed {
				return NewHTTPError(http.StatusTooManyRequests, CodeRateLimited, "请求过于频繁，请稍后再试")
			}
			return next(c)
		}
	}
}

// RateLimit enforces per-user and per-guest sliding-window rate limits
// and limits concurrent SSE connections per user. It must run after Auth
// so that the caller identity is present in the request context.
func RateLimit(svc limit.Service, cfg config.Config) echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			ctx := c.Request().Context()
			path := c.Request().URL.Path

			id := auth.IdentityFrom(ctx)
			if id == nil {
				return next(c)
			}

			if id.Type == "guest" || id.Role == "guest" {
				key := limit.GuestKey(id.UserID)
				if allowed, err := svc.Allow(ctx, key, 60, cfg.RateLimitGuest); err != nil {
					c.Logger().Errorf("guest rate limit redis error: %v", err)
				} else if !allowed {
					return NewHTTPError(http.StatusTooManyRequests, CodeRateLimited, "请求过于频繁，请稍后再试")
				}
			} else {
				key := limit.UserKey(id.UserID)
				if allowed, err := svc.Allow(ctx, key, 60, cfg.RateLimitUser); err != nil {
					c.Logger().Errorf("user rate limit redis error: %v", err)
				} else if !allowed {
					return NewHTTPError(http.StatusTooManyRequests, CodeRateLimited, "请求过于频繁，请稍后再试")
				}
			}

			if isSSEPath(path) {
				ok, err := svc.AcquireSSE(ctx, id.UserID, cfg.MaxConcurrentSSE)
				if err != nil {
					c.Logger().Errorf("sse acquire redis error: %v", err)
				} else if !ok {
					return NewHTTPError(http.StatusTooManyRequests, CodeSSELimit, "并发连接数已达上限")
				}
				defer func() {
					if err := svc.ReleaseSSE(context.Background(), id.UserID); err != nil {
						c.Logger().Errorf("sse release redis error: %v", err)
					}
				}()
			}

			return next(c)
		}
	}
}

func isSSEPath(path string) bool {
	return strings.Contains(path, "/chat/stream") || (strings.Contains(path, "/itineraries/") && strings.HasSuffix(path, "/stream"))
}

package middleware

import (
	"net/http"
	"strings"

	"github.com/labstack/echo/v4"
	"github.com/redis/go-redis/v9"

	"github.com/travelagent/gateway/internal/auth"
)

// Auth extracts and validates the JWT from the Authorization header.
// Public paths (health, metrics, auth endpoints) are skipped.
func Auth(svc auth.Service, rdb redis.Cmdable, publicPaths ...string) echo.MiddlewareFunc {
	isPublic := func(path string) bool {
		if path == "/health" || path == "/metrics" {
			return true
		}
		if strings.HasPrefix(path, "/api/v1/auth/") {
			return true
		}
		for _, p := range publicPaths {
			if p == path {
				return true
			}
		}
		return false
	}

	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			if isPublic(c.Request().URL.Path) {
				return next(c)
			}

			authHeader := c.Request().Header.Get("Authorization")
			if authHeader == "" || !strings.HasPrefix(strings.ToLower(authHeader), "bearer ") {
				return NewHTTPError(http.StatusUnauthorized, CodeAuthMissing, "请先登录")
			}

			token := strings.TrimSpace(authHeader[7:])
			if token == "" {
				return NewHTTPError(http.StatusUnauthorized, CodeAuthMissing, "请先登录")
			}

			ctx := c.Request().Context()

			// Check token blacklist.
			blacklisted, err := rdb.Exists(ctx, svc.BlacklistKey(token)).Result()
			if err == nil && blacklisted > 0 {
				return NewHTTPError(http.StatusUnauthorized, CodeTokenRevoked, "登录状态已失效")
			}

			claims, err := svc.Parse(token)
			if err != nil {
				switch err {
				case auth.ErrExpiredToken:
					return NewHTTPError(http.StatusUnauthorized, CodeTokenExpired, "登录已过期，请重新登录")
				case auth.ErrMissingToken, auth.ErrInvalidToken:
					return NewHTTPError(http.StatusUnauthorized, CodeTokenInvalid, "登录状态异常")
				default:
					return NewHTTPError(http.StatusUnauthorized, CodeTokenInvalid, "登录状态异常")
				}
			}

			// Check user ban set.
			if claims.Sub != "" {
				banned, err := rdb.Exists(ctx, svc.UserBanKey(claims.Sub)).Result()
				if err == nil && banned > 0 {
					return NewHTTPError(http.StatusUnauthorized, CodeTokenRevoked, "登录状态已失效")
				}
			}

			// Guest device fingerprint validation.
			if claims.Type == "guest" || claims.Role == "guest" {
				expected := claims.DeviceFingerprint
				actual := c.Request().Header.Get("X-Device-Fingerprint")
				if expected != "" && expected != actual {
					return NewHTTPError(http.StatusForbidden, CodeDeviceMismatch, "请在原设备继续使用或重新获取游客身份")
				}
			}

			identity := auth.Identity{
				UserID:            claims.Sub,
				Role:              claims.Role,
				Type:              claims.Type,
				DeviceFingerprint: claims.DeviceFingerprint,
			}
			ctx = auth.WithIdentity(ctx, identity)
			c.SetRequest(c.Request().WithContext(ctx))

			// Propagate upstream headers.
			c.Request().Header.Set("X-User-ID", claims.Sub)
			c.Request().Header.Set("X-User-Role", claims.Role)

			return next(c)
		}
	}
}

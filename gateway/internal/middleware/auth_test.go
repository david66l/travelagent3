package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/golang-jwt/jwt/v5"
	"github.com/labstack/echo/v4"
	"github.com/redis/go-redis/v9"

	"github.com/travelagent/gateway/internal/auth"
)

// testClaims mirrors the internal jwtClaims layout used by auth.Service.
type testClaims struct {
	Sub               string `json:"sub"`
	Role              string `json:"role"`
	Type              string `json:"type"`
	DeviceFingerprint string `json:"device_fingerprint"`
	jwt.RegisteredClaims
}

func setupAuthTest(t *testing.T) (*miniredis.Miniredis, *redis.Client, echo.Context, *httptest.ResponseRecorder, *echo.Echo) {
	t.Helper()
	s := miniredis.RunT(t)
	opt, _ := redis.ParseURL("redis://" + s.Addr())
	rdb := redis.NewClient(opt)

	e := echo.New()
	e.HTTPErrorHandler = ErrorHandler()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/chat/stream", nil)
	rec := httptest.NewRecorder()
	c := e.NewContext(req, rec)
	return s, rdb, c, rec, e
}

func signToken(claims auth.Claims, exp time.Time) string {
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, testClaims{
		Sub:               claims.Sub,
		Role:              claims.Role,
		Type:              claims.Type,
		DeviceFingerprint: claims.DeviceFingerprint,
		RegisteredClaims:  jwt.RegisteredClaims{ExpiresAt: jwt.NewNumericDate(exp)},
	})
	signed, _ := token.SignedString([]byte("test-secret"))
	return signed
}

func runHandler(t *testing.T, e *echo.Echo, h echo.HandlerFunc, c echo.Context) {
	t.Helper()
	err := h(c)
	if err != nil {
		e.HTTPErrorHandler(err, c)
	}
}

func TestAuthMissingToken(t *testing.T) {
	_, rdb, c, rec, e := setupAuthTest(t)
	defer rdb.Close()

	svc := auth.NewValidator("test-secret", "HS256")
	h := Auth(svc, rdb)(func(c echo.Context) error { return c.String(http.StatusOK, "ok") })

	runHandler(t, e, h, c)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
	if !contains(rec.Body.String(), "AUTH_MISSING") {
		t.Fatalf("body = %s, want AUTH_MISSING", rec.Body.String())
	}
}

func TestAuthValidUser(t *testing.T) {
	_, rdb, c, rec, e := setupAuthTest(t)
	defer rdb.Close()

	token := signToken(auth.Claims{Sub: "user-1", Role: "user", Type: "access"}, time.Now().Add(time.Hour))
	c.Request().Header.Set("Authorization", "Bearer "+token)

	svc := auth.NewValidator("test-secret", "HS256")
	var captured *auth.Identity
	h := Auth(svc, rdb)(func(c echo.Context) error {
		id := auth.IdentityFrom(c.Request().Context())
		captured = id
		return c.String(http.StatusOK, "ok")
	})

	runHandler(t, e, h, c)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	if captured == nil || captured.UserID != "user-1" {
		t.Fatalf("identity not propagated: %v", captured)
	}
}

func TestAuthExpiredToken(t *testing.T) {
	_, rdb, c, rec, e := setupAuthTest(t)
	defer rdb.Close()

	token := signToken(auth.Claims{Sub: "user-1", Role: "user", Type: "access"}, time.Now().Add(-time.Hour))
	c.Request().Header.Set("Authorization", "Bearer "+token)

	svc := auth.NewValidator("test-secret", "HS256")
	h := Auth(svc, rdb)(func(c echo.Context) error { return c.String(http.StatusOK, "ok") })

	runHandler(t, e, h, c)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
	if !contains(rec.Body.String(), "TOKEN_EXPIRED") {
		t.Fatalf("body = %s, want TOKEN_EXPIRED", rec.Body.String())
	}
}

func TestAuthBlacklistedToken(t *testing.T) {
	s, rdb, c, rec, e := setupAuthTest(t)
	defer rdb.Close()

	token := signToken(auth.Claims{Sub: "user-1", Role: "user", Type: "access"}, time.Now().Add(time.Hour))
	svc := auth.NewValidator("test-secret", "HS256")
	s.Set(svc.BlacklistKey(token), "1")

	c.Request().Header.Set("Authorization", "Bearer "+token)
	h := Auth(svc, rdb)(func(c echo.Context) error { return c.String(http.StatusOK, "ok") })

	runHandler(t, e, h, c)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
	if !contains(rec.Body.String(), "TOKEN_REVOKED") {
		t.Fatalf("body = %s, want TOKEN_REVOKED", rec.Body.String())
	}
}

func TestAuthDeviceMismatch(t *testing.T) {
	_, rdb, c, rec, e := setupAuthTest(t)
	defer rdb.Close()

	token := signToken(auth.Claims{Sub: "guest-1", Role: "guest", Type: "guest", DeviceFingerprint: "fp-expected"}, time.Now().Add(time.Hour))
	c.Request().Header.Set("Authorization", "Bearer "+token)
	c.Request().Header.Set("X-Device-Fingerprint", "fp-wrong")

	svc := auth.NewValidator("test-secret", "HS256")
	h := Auth(svc, rdb)(func(c echo.Context) error { return c.String(http.StatusOK, "ok") })

	runHandler(t, e, h, c)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusForbidden)
	}
	if !contains(rec.Body.String(), "DEVICE_MISMATCH") {
		t.Fatalf("body = %s, want DEVICE_MISMATCH", rec.Body.String())
	}
}

func TestAuthPublicPathSkipped(t *testing.T) {
	_, rdb, c, rec, e := setupAuthTest(t)
	defer rdb.Close()

	svc := auth.NewValidator("test-secret", "HS256")
	c.SetRequest(httptest.NewRequest(http.MethodPost, "/api/v1/auth/guest", nil))
	h := Auth(svc, rdb)(func(c echo.Context) error { return c.String(http.StatusOK, "ok") })

	runHandler(t, e, h, c)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
}

func contains(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

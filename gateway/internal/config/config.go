package config

import (
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds all gateway configuration with sensible defaults.
type Config struct {
	// Server
	Port         string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration

	// Security
	JWTSecret                string
	JWTAlgorithm             string
	AccessTokenExpireMinutes int
	RefreshTokenExpireDays   int

	// Redis
	RedisURL string

	// Upstream
	BackendURL  string
	FrontendURL string

	// Browser origins allowed to call the API. Credentials are enabled, so a
	// wildcard is never valid here and every real deployment origin must be
	// listed explicitly.
	CORSAllowOrigins []string

	// Browser request headers the API accepts. Any header a client sends on a
	// cross-origin request must appear here or the preflight rejects the call.
	CORSAllowHeaders []string

	// Rate limiting (requests per minute)
	RateLimitIP    int
	RateLimitUser  int
	RateLimitGuest int

	// SSE limiting
	MaxConcurrentSSE int

	// Circuit breaker
	BreakerFailThreshold int     // minimum failures in window to open
	BreakerWindowSec     int     // observation window in seconds
	BreakerOpenSec       int     // how long breaker stays open
	BreakerFailRate      float64 // failure rate threshold (0.0-1.0)
}

// Load reads configuration from environment variables and fills defaults.
func Load() Config {
	return Config{
		Port:                     env("GATEWAY_PORT", "8080"),
		ReadTimeout:              envDuration("GATEWAY_READ_TIMEOUT", 10*time.Second),
		// A planning SSE response commonly lasts 30-90 seconds. Go's server-level
		// WriteTimeout covers the entire streamed response, so a finite default
		// truncates a healthy stream mid-plan. Per-request upstream timeouts and
		// SSE keepalives still protect the service.
		WriteTimeout:             envDuration("GATEWAY_WRITE_TIMEOUT", 0),
		JWTSecret:                env("JWT_SECRET", "dev-secret-change-me"),
		JWTAlgorithm:             env("JWT_ALGORITHM", "HS256"),
		AccessTokenExpireMinutes: envInt("ACCESS_TOKEN_EXPIRE_MINUTES", 30),
		RefreshTokenExpireDays:   envInt("REFRESH_TOKEN_EXPIRE_DAYS", 7),
		RedisURL:                 env("REDIS_URL", "redis://localhost:6379/0"),
		BackendURL:               env("BACKEND_URL", "http://localhost:8000"),
		FrontendURL:              env("FRONTEND_URL", "http://localhost:3000"),
		CORSAllowOrigins: envList("GATEWAY_CORS_ORIGINS", []string{
			"http://localhost:3000",
			"http://127.0.0.1:3000",
		}),
		CORSAllowHeaders: envList("GATEWAY_CORS_ALLOW_HEADERS", []string{
			"Authorization",
			"Content-Type",
			// The chat client sends this to make message submission idempotent.
			// Omitting it made every cross-origin POST fail its CORS preflight.
			"Idempotency-Key",
			"X-Device-Fingerprint",
			"X-Request-ID",
		}),
		RateLimitIP:              envInt("RATE_LIMIT_IP_PER_MINUTE", 60),
		RateLimitUser:            envInt("RATE_LIMIT_USER_PER_MINUTE", 60),
		RateLimitGuest:           envInt("RATE_LIMIT_GUEST_PER_MINUTE", 30),
		MaxConcurrentSSE:         envInt("RATE_LIMIT_MAX_CONCURRENT_SSE", 3),
		BreakerFailThreshold:     envInt("GATEWAY_BREAKER_FAIL_THRESHOLD", 20),
		BreakerWindowSec:         envInt("GATEWAY_BREAKER_WINDOW_SEC", 10),
		BreakerOpenSec:           envInt("GATEWAY_BREAKER_OPEN_SEC", 30),
		BreakerFailRate:          envFloat("GATEWAY_BREAKER_FAIL_RATE", 0.5),
	}
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envDuration(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

// envList parses a comma-separated environment variable into a slice, dropping
// empty entries. An unset or blank value keeps the default list.
func envList(key string, def []string) []string {
	raw := os.Getenv(key)
	if strings.TrimSpace(raw) == "" {
		return def
	}
	items := make([]string, 0, len(def))
	for _, part := range strings.Split(raw, ",") {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			items = append(items, trimmed)
		}
	}
	if len(items) == 0 {
		return def
	}
	return items
}

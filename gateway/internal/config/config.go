package config

import (
	"os"
	"strconv"
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
		WriteTimeout:             envDuration("GATEWAY_WRITE_TIMEOUT", 30*time.Second),
		JWTSecret:                env("JWT_SECRET", "dev-secret-change-me"),
		JWTAlgorithm:             env("JWT_ALGORITHM", "HS256"),
		AccessTokenExpireMinutes: envInt("ACCESS_TOKEN_EXPIRE_MINUTES", 30),
		RefreshTokenExpireDays:   envInt("REFRESH_TOKEN_EXPIRE_DAYS", 7),
		RedisURL:                 env("REDIS_URL", "redis://localhost:6379/0"),
		BackendURL:               env("BACKEND_URL", "http://localhost:8000"),
		FrontendURL:              env("FRONTEND_URL", "http://localhost:3000"),
		RateLimitIP:              envInt("RATE_LIMIT_IP_PER_MINUTE", 60),
		RateLimitUser:            envInt("RATE_LIMIT_USER_PER_MINUTE", 30),
		RateLimitGuest:           envInt("RATE_LIMIT_GUEST_PER_MINUTE", 10),
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

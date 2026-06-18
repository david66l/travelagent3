package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"

	"github.com/travelagent/gateway/internal/auth"
	"github.com/travelagent/gateway/internal/breaker"
	"github.com/travelagent/gateway/internal/config"
	gwHandler "github.com/travelagent/gateway/internal/handler"
	"github.com/travelagent/gateway/internal/limit"
	gwMiddleware "github.com/travelagent/gateway/internal/middleware"
	"github.com/travelagent/gateway/internal/proxy"
)

var (
	reqTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{Name: "gateway_requests_total", Help: "Gateway HTTP requests"},
		[]string{"path", "status"},
	)
	reqDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{Name: "gateway_request_duration_seconds", Help: "Gateway latency", Buckets: prometheus.DefBuckets},
		[]string{"path"},
	)
)

func init() {
	prometheus.MustRegister(reqTotal, reqDuration)
}

func main() {
	cfg := config.Load()

	// Redis client.
	opt, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		log.Fatalf("invalid REDIS_URL: %v", err)
	}
	rdb := redis.NewClient(opt)
	defer func() {
		_ = rdb.Close()
	}()

	// Domain services.
	authSvc := auth.NewValidator(cfg.JWTSecret, cfg.JWTAlgorithm)
	limitSvc := limit.NewService(rdb)
	cb := breaker.New(breaker.Options{
		FailThreshold: cfg.BreakerFailThreshold,
		WindowSec:     cfg.BreakerWindowSec,
		OpenSec:       cfg.BreakerOpenSec,
		FailRate:      cfg.BreakerFailRate,
	})

	// Upstream handlers.
	backendHandler, err := proxy.NewBackend(cfg.BackendURL)
	if err != nil {
		log.Fatalf("invalid BACKEND_URL: %v", err)
	}
	frontendHandler, err := proxy.NewFrontend(cfg.FrontendURL)
	if err != nil {
		log.Fatalf("invalid FRONTEND_URL: %v", err)
	}

	// Echo setup.
	e := echo.New()
	e.HideBanner = true
	e.HidePort = true
	e.HTTPErrorHandler = gwMiddleware.ErrorHandler()

	// Built-in middleware.
	e.Use(middleware.Recover())
	e.Use(middleware.LoggerWithConfig(middleware.LoggerConfig{
		Format: "{time_rfc3339} method=${method}, uri=${uri}, status=${status}, latency=${latency_human}, request_id=${header:X-Request-ID}\n",
	}))
	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowOrigins:     []string{"http://localhost:3000", "http://127.0.0.1:3000"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Authorization", "Content-Type", "X-Device-Fingerprint", "X-Request-ID"},
		AllowCredentials: true,
	}))
	e.Use(gwMiddleware.RequestID())
	e.Use(prometheusMiddleware())

	// Global IP rate limit applies to every request.
	e.Use(gwMiddleware.IPRateLimit(limitSvc, cfg))

	// Public routes (no auth, no breaker).
	e.GET("/health", gwHandler.Health)
	e.GET("/api/v1/health", gwHandler.Health)
	e.GET("/metrics", echo.WrapHandler(promhttp.Handler()))

	// Public auth endpoints: IP rate limit only.
	publicAPI := e.Group("/api/v1/auth")
	publicAPI.Any("/*", backendHandler)

	// Protected API endpoints: auth -> user/guest rate limit -> circuit breaker.
	protectedAPI := e.Group("/api/v1")
	protectedAPI.Use(gwMiddleware.Auth(authSvc, rdb))
	protectedAPI.Use(gwMiddleware.RateLimit(limitSvc, cfg))
	protectedAPI.Use(gwMiddleware.CircuitBreaker(cb))
	protectedAPI.Any("/*", backendHandler)

	// Frontend catch-all.
	e.GET("/*", frontendHandler)

	// Graceful shutdown.
	srv := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      e,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
	}

	go func() {
		log.Printf("Go gateway listening on :%s -> %s", cfg.Port, cfg.BackendURL)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := e.Shutdown(ctx); err != nil {
		log.Fatalf("shutdown error: %v", err)
	}
}

// prometheusMiddleware records request metrics using path and status labels.
func prometheusMiddleware() echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			start := time.Now()
			pathLabel := c.Request().URL.Path

			err := next(c)

			status := c.Response().Status
			if status == 0 {
				status = http.StatusOK
			}
			// Bucket wildcard IDs to avoid metric explosion.
			pathLabel = normalizePath(pathLabel)

			reqDuration.WithLabelValues(pathLabel).Observe(time.Since(start).Seconds())
			reqTotal.WithLabelValues(pathLabel, http.StatusText(status)).Inc()
			return err
		}
	}
}

func normalizePath(path string) string {
	// Replace UUID-like and numeric path segments with placeholders.
	parts := strings.Split(path, "/")
	for i, p := range parts {
		if isUUIDLike(p) || isNumeric(p) {
			parts[i] = ":id"
		}
	}
	return strings.Join(parts, "/")
}

func isUUIDLike(s string) bool {
	if len(s) != 36 {
		return false
	}
	for i, r := range s {
		switch {
		case r >= '0' && r <= '9', r >= 'a' && r <= 'f', r >= 'A' && r <= 'F':
		case (i == 8 || i == 13 || i == 18 || i == 23) && r == '-':
		default:
			return false
		}
	}
	return true
}

func isNumeric(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

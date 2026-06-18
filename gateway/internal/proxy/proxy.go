package proxy

import (
	"net/http"
	"net/http/httputil"
	"net/url"

	"github.com/labstack/echo/v4"
)

// Target describes an upstream target.
type Target struct {
	URL  *url.URL
	Name string
}

// NewBackend creates a reverse proxy handler for the backend service.
func NewBackend(backendURL string) (echo.HandlerFunc, error) {
	u, err := url.Parse(backendURL)
	if err != nil {
		return nil, err
	}
	proxy := httputil.NewSingleHostReverseProxy(u)

	// Preserve the original host header so the upstream sees its expected host
	// when running behind the gateway.
	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		req.Host = req.URL.Host
	}

	return func(c echo.Context) error {
		proxy.ServeHTTP(c.Response(), c.Request())
		return nil
	}, nil
}

// NewFrontend creates a reverse proxy handler for the frontend service.
func NewFrontend(frontendURL string) (echo.HandlerFunc, error) {
	u, err := url.Parse(frontendURL)
	if err != nil {
		return nil, err
	}
	proxy := httputil.NewSingleHostReverseProxy(u)
	return func(c echo.Context) error {
		proxy.ServeHTTP(c.Response(), c.Request())
		return nil
	}, nil
}

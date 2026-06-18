package handler

import (
	"net/http"

	"github.com/labstack/echo/v4"
)

// Health returns the gateway health status.
func Health(c echo.Context) error {
	return c.JSON(http.StatusOK, map[string]string{
		"status":  "ok",
		"service": "gateway",
	})
}

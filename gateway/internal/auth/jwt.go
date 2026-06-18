package auth

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// validator implements Service using golang-jwt/jwt/v5.
type validator struct {
	secret    []byte
	algorithm string
}

// NewValidator creates a new JWT validation service.
func NewValidator(secret, algorithm string) Service {
	if algorithm == "" {
		algorithm = "HS256"
	}
	return &validator{
		secret:    []byte(secret),
		algorithm: algorithm,
	}
}

// Parse validates a raw JWT string and returns its claims.
func (v *validator) Parse(token string) (*Claims, error) {
	if strings.TrimSpace(token) == "" {
		return nil, ErrMissingToken
	}

	parsed, err := jwt.ParseWithClaims(token, &jwtClaims{}, func(t *jwt.Token) (interface{}, error) {
		if t.Method.Alg() != v.algorithm {
			return nil, fmt.Errorf("unexpected signing method: %s", t.Method.Alg())
		}
		return v.secret, nil
	})
	if err != nil {
		if errors.Is(err, jwt.ErrTokenExpired) {
			return nil, ErrExpiredToken
		}
		return nil, ErrInvalidToken
	}

	jc, ok := parsed.Claims.(*jwtClaims)
	if !ok || !parsed.Valid || jc.Subject == "" || jc.Type == "" {
		return nil, ErrInvalidToken
	}

	return &Claims{
		Sub:               jc.Subject,
		Role:              jc.Role,
		Type:              jc.Type,
		DeviceFingerprint: jc.DeviceFingerprint,
	}, nil
}

// BlacklistKey returns the Redis key for a revoked token hash.
func (v *validator) BlacklistKey(token string) string {
	return fmt.Sprintf("jwt_blacklist:%s", tokenHash(token))
}

// UserBanKey returns the Redis key used to mark a banned user.
func (v *validator) UserBanKey(userID string) string {
	return fmt.Sprintf("jwt_banned_user:%s", userID)
}

// tokenHash computes SHA256 of the token for stable Redis keys.
func tokenHash(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}

// RemainingTTL returns the remaining TTL in seconds for an expiration time.
func RemainingTTL(exp time.Time) int {
	ttl := int(time.Until(exp).Seconds())
	if ttl < 0 {
		return 0
	}
	return ttl
}

// jwtClaims wraps gateway Claims with jwt.RegisteredClaims.
// NOTE: Do NOT add a custom Sub field — jwt.RegisteredClaims.Subject
// already maps to json:"sub". Adding another would cause a conflict
// and the custom field would remain empty after parsing.
type jwtClaims struct {
	Role              string `json:"role"`
	Type              string `json:"type"`
	DeviceFingerprint string `json:"device_fingerprint"`
	jwt.RegisteredClaims
}

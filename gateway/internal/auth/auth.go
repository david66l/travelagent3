package auth

import (
	"context"
	"errors"
)

// Common auth errors returned by the gateway.
var (
	ErrMissingToken   = errors.New("missing token")
	ErrInvalidToken   = errors.New("invalid token")
	ErrExpiredToken   = errors.New("token expired")
	ErrRevokedToken   = errors.New("token revoked")
	ErrBannedUser     = errors.New("user banned")
	ErrDeviceMismatch = errors.New("device mismatch")
)

// Claims represents the JWT claims extracted from a token.
type Claims struct {
	Sub               string `json:"sub"`
	Role              string `json:"role"`
	Type              string `json:"type"`
	DeviceFingerprint string `json:"device_fingerprint"`
}

// Service validates tokens and provides blacklist helpers.
type Service interface {
	// Parse validates a raw JWT string and returns its claims.
	Parse(token string) (*Claims, error)

	// BlacklistKey returns the Redis key used to store a revoked token hash.
	BlacklistKey(token string) string

	// UserBanKey returns the Redis key used to mark a banned user.
	UserBanKey(userID string) string
}

// Identity holds the authenticated caller identity attached to request context.
type Identity struct {
	UserID            string
	Role              string
	Type              string
	DeviceFingerprint string
}

// contextKey is an unexported type to avoid context key collisions.
type contextKey struct{}

var identityKey = contextKey{}

// WithIdentity stores the identity in the context.
func WithIdentity(ctx context.Context, id Identity) context.Context {
	return context.WithValue(ctx, identityKey, id)
}

// IdentityFrom extracts the identity from the context.
// Returns nil if no identity has been stored.
func IdentityFrom(ctx context.Context) *Identity {
	v := ctx.Value(identityKey)
	if v == nil {
		return nil
	}
	id, ok := v.(Identity)
	if !ok {
		return nil
	}
	return &id
}

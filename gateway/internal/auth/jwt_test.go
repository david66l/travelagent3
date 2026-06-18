package auth

import (
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

func TestValidatorParseGuest(t *testing.T) {
	v := NewValidator("test-secret", "HS256")
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwtClaims{
		Role:              "guest",
		Type:              "guest",
		DeviceFingerprint: "fp-abc",
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   "guest-1",
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
		},
	})
	signed, err := token.SignedString([]byte("test-secret"))
	if err != nil {
		t.Fatal(err)
	}

	claims, err := v.Parse(signed)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if claims.Sub != "guest-1" {
		t.Errorf("sub = %q, want %q", claims.Sub, "guest-1")
	}
	if claims.Role != "guest" {
		t.Errorf("role = %q, want %q", claims.Role, "guest")
	}
	if claims.DeviceFingerprint != "fp-abc" {
		t.Errorf("device fingerprint = %q, want %q", claims.DeviceFingerprint, "fp-abc")
	}
}

func TestValidatorExpired(t *testing.T) {
	v := NewValidator("test-secret", "HS256")
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwtClaims{
		Type: "access",
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   "u1",
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(-time.Minute)),
		},
	})
	signed, _ := token.SignedString([]byte("test-secret"))
	_, err := v.Parse(signed)
	if err != ErrExpiredToken {
		t.Fatalf("expected ErrExpiredToken, got %v", err)
	}
}

func TestValidatorInvalidSignature(t *testing.T) {
	v := NewValidator("test-secret", "HS256")
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwtClaims{
		Type: "access",
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   "u1",
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour)),
		},
	})
	signed, _ := token.SignedString([]byte("wrong-secret"))
	_, err := v.Parse(signed)
	if err != ErrInvalidToken {
		t.Fatalf("expected ErrInvalidToken, got %v", err)
	}
}

func TestValidatorMissingToken(t *testing.T) {
	v := NewValidator("test-secret", "HS256")
	_, err := v.Parse("   ")
	if err != ErrMissingToken {
		t.Fatalf("expected ErrMissingToken, got %v", err)
	}
}

func TestBlacklistKey(t *testing.T) {
	v := NewValidator("test-secret", "HS256")
	key := v.BlacklistKey("abc")
	want := "jwt_blacklist:" + tokenHash("abc")
	if key != want {
		t.Errorf("blacklist key = %q, want %q", key, want)
	}
}

func TestRemainingTTL(t *testing.T) {
	if got := RemainingTTL(time.Now().Add(time.Hour)); got <= 3590 || got > 3600 {
		t.Errorf("remaining ttl = %d, want around 3600", got)
	}
	if got := RemainingTTL(time.Now().Add(-time.Minute)); got != 0 {
		t.Errorf("remaining ttl = %d, want 0", got)
	}
}

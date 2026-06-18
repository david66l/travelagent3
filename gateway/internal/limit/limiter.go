package limit

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// RedisCmdable matches the subset of redis.Cmdable used by this package.
// It allows tests to inject miniredis or a real client uniformly.
type RedisCmdable interface {
	Eval(ctx context.Context, script string, keys []string, args ...interface{}) *redis.Cmd
	Incr(ctx context.Context, key string) *redis.IntCmd
	Decr(ctx context.Context, key string) *redis.IntCmd
	Expire(ctx context.Context, key string, ttl time.Duration) *redis.BoolCmd
}

// Service provides rate limiting operations.
type Service interface {
	// Allow returns true if the request identified by key is within the limit.
	Allow(ctx context.Context, key string, windowSec, limit int) (bool, error)

	// AcquireSSE increments the user's active SSE connection count and returns true
	// if it is within max concurrent connections.
	AcquireSSE(ctx context.Context, userID string, max int) (bool, error)

	// ReleaseSSE decrements the user's active SSE connection count.
	ReleaseSSE(ctx context.Context, userID string) error
}

// redisService implements Service using Redis.
type redisService struct {
	rdb RedisCmdable
}

// NewService creates a rate limiter backed by Redis.
func NewService(rdb RedisCmdable) Service {
	return &redisService{rdb: rdb}
}

// Allow uses a Redis sorted-set sliding window.
func (s *redisService) Allow(ctx context.Context, key string, windowSec, limit int) (bool, error) {
	now := time.Now()
	score := float64(now.Unix())
	member := now.Format(time.RFC3339Nano)
	lua := `
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local score = tonumber(ARGV[3])
local member = ARGV[4]
local min_score = score - window
redis.call("ZREMRANGEBYSCORE", key, 0, min_score)
local count = redis.call("ZCARD", key)
if count < limit then
  redis.call("ZADD", key, score, member)
  redis.call("EXPIRE", key, window)
  return 1
end
return 0
`
	res, err := s.rdb.Eval(ctx, lua, []string{key}, windowSec, limit, score, member).Int()
	if err != nil {
		return false, err
	}
	return res == 1, nil
}

// AcquireSSE increments the active SSE connection counter.
func (s *redisService) AcquireSSE(ctx context.Context, userID string, max int) (bool, error) {
	key := sseKey(userID)
	count, err := s.rdb.Incr(ctx, key).Result()
	if err != nil {
		return false, err
	}
	if count == 1 {
		_ = s.rdb.Expire(ctx, key, 3600*time.Second).Err()
	}
	if count > int64(max) {
		_ = s.rdb.Decr(ctx, key).Err()
		return false, nil
	}
	return true, nil
}

// ReleaseSSE decrements the active SSE connection counter.
func (s *redisService) ReleaseSSE(ctx context.Context, userID string) error {
	return s.rdb.Decr(ctx, sseKey(userID)).Err()
}

// Key builders.
func IPKey(ip string) string         { return fmt.Sprintf("rate_limit:ip:%s", ip) }
func UserKey(userID string) string   { return fmt.Sprintf("rate_limit:user:%s", userID) }
func GuestKey(guestID string) string { return fmt.Sprintf("rate_limit:guest:%s", guestID) }
func sseKey(userID string) string    { return fmt.Sprintf("gw:sse:%s", userID) }

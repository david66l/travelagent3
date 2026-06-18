package limit

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func setupRedis(t *testing.T) (*redis.Client, func()) {
	t.Helper()
	s := miniredis.RunT(t)
	opt, err := redis.ParseURL("redis://" + s.Addr())
	if err != nil {
		t.Fatal(err)
	}
	client := redis.NewClient(opt)
	return client, func() { client.Close(); s.Close() }
}

func TestAllowWithinLimit(t *testing.T) {
	rdb, cleanup := setupRedis(t)
	defer cleanup()

	svc := NewService(rdb)
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		ok, err := svc.Allow(ctx, "rl:test", 60, 5)
		if err != nil {
			t.Fatalf("allow error: %v", err)
		}
		if !ok {
			t.Fatalf("request %d should be allowed", i+1)
		}
	}
	ok, err := svc.Allow(ctx, "rl:test", 60, 5)
	if err != nil {
		t.Fatalf("allow error: %v", err)
	}
	if ok {
		t.Fatal("6th request should be denied")
	}
}

func TestAllowSlidingWindowExpires(t *testing.T) {
	rdb, cleanup := setupRedis(t)
	defer cleanup()

	svc := NewService(rdb)
	ctx := context.Background()
	ok, _ := svc.Allow(ctx, "rl:slide", 1, 1)
	if !ok {
		t.Fatal("first request should be allowed")
	}
	ok, _ = svc.Allow(ctx, "rl:slide", 1, 1)
	if ok {
		t.Fatal("second request should be denied within window")
	}
	time.Sleep(1100 * time.Millisecond)
	ok, _ = svc.Allow(ctx, "rl:slide", 1, 1)
	if !ok {
		t.Fatal("request after window expiry should be allowed")
	}
}

func TestAcquireAndReleaseSSE(t *testing.T) {
	rdb, cleanup := setupRedis(t)
	defer cleanup()

	svc := NewService(rdb)
	ctx := context.Background()
	userID := "user-sse-1"

	for i := 0; i < 3; i++ {
		ok, err := svc.AcquireSSE(ctx, userID, 3)
		if err != nil {
			t.Fatalf("acquire error: %v", err)
		}
		if !ok {
			t.Fatalf("connection %d should be acquired", i+1)
		}
	}
	ok, _ := svc.AcquireSSE(ctx, userID, 3)
	if ok {
		t.Fatal("4th concurrent SSE connection should be denied")
	}

	if err := svc.ReleaseSSE(ctx, userID); err != nil {
		t.Fatalf("release error: %v", err)
	}
	ok, _ = svc.AcquireSSE(ctx, userID, 3)
	if !ok {
		t.Fatal("after release, connection should be acquirable again")
	}
}

func TestKeyBuilders(t *testing.T) {
	if got := IPKey("1.2.3.4"); got != "rate_limit:ip:1.2.3.4" {
		t.Errorf("IPKey = %q", got)
	}
	if got := UserKey("u1"); got != "rate_limit:user:u1" {
		t.Errorf("UserKey = %q", got)
	}
	if got := GuestKey("g1"); got != "rate_limit:guest:g1" {
		t.Errorf("GuestKey = %q", got)
	}
}

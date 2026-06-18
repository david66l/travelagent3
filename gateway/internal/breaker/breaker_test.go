package breaker

import (
	"testing"
	"time"
)

func TestBreakerStaysClosedOnLowFailures(t *testing.T) {
	b := New(Options{FailThreshold: 20, WindowSec: 10, OpenSec: 1, FailRate: 0.5})
	for i := 0; i < 10; i++ {
		b.RecordFailure()
	}
	if !b.Allow() {
		t.Fatal("breaker should remain closed with only 10 failures")
	}
}

func TestBreakerOpensOnHighFailureRate(t *testing.T) {
	b := New(Options{FailThreshold: 20, WindowSec: 10, OpenSec: 1, FailRate: 0.5})
	for i := 0; i < 30; i++ {
		b.RecordFailure()
	}
	if b.Allow() {
		t.Fatal("breaker should open after 30 failures")
	}
	if b.State() != StateOpen {
		t.Fatalf("state = %v, want Open", b.State())
	}
}

func TestBreakerDoesNotOpenWhenFailureRateLow(t *testing.T) {
	b := New(Options{FailThreshold: 20, WindowSec: 10, OpenSec: 1, FailRate: 0.5})
	for i := 0; i < 30; i++ {
		b.RecordSuccess()
	}
	for i := 0; i < 20; i++ {
		b.RecordFailure()
	}
	// failure rate = 20/50 = 0.4 (< 0.5), should stay closed
	if !b.Allow() {
		t.Fatal("breaker should stay closed when failure rate is below threshold")
	}
}

func TestBreakerHalfOpenThenCloses(t *testing.T) {
	b := New(Options{FailThreshold: 20, WindowSec: 10, OpenSec: 1, FailRate: 0.5})
	for i := 0; i < 30; i++ {
		b.RecordFailure()
	}
	if b.State() != StateOpen {
		t.Fatal("breaker should be open")
	}
	time.Sleep(1100 * time.Millisecond)
	if !b.Allow() {
		t.Fatal("breaker should be half-open after open duration")
	}
	b.RecordSuccess()
	if b.State() != StateClosed {
		t.Fatalf("state = %v, want Closed after success in half-open", b.State())
	}
}

func TestBreakerHalfOpenThenOpensAgain(t *testing.T) {
	b := New(Options{FailThreshold: 20, WindowSec: 10, OpenSec: 1, FailRate: 0.5})
	for i := 0; i < 30; i++ {
		b.RecordFailure()
	}
	time.Sleep(1100 * time.Millisecond)
	if !b.Allow() {
		t.Fatal("breaker should be half-open")
	}
	b.RecordFailure()
	if b.State() != StateOpen {
		t.Fatalf("state = %v, want Open after failure in half-open", b.State())
	}
}

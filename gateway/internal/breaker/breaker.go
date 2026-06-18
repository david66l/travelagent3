package breaker

import (
	"sync"
	"time"
)

// State represents the circuit breaker state.
type State int

const (
	StateClosed State = iota
	StateOpen
	StateHalfOpen
)

// Breaker provides circuit breaker protection for upstream calls.
type Breaker interface {
	// Allow returns true when the breaker is closed or half-open.
	Allow() bool

	// RecordSuccess records a successful upstream call.
	RecordSuccess()

	// RecordFailure records a failed upstream call.
	RecordFailure()

	// State returns the current breaker state.
	State() State
}

// Options configures the circuit breaker.
type Options struct {
	// FailThreshold is the minimum number of failures within the window required
	// before the breaker can open.
	FailThreshold int

	// WindowSec is the observation window in seconds.
	WindowSec int

	// OpenSec is how long the breaker stays open before transitioning to half-open.
	OpenSec int

	// FailRate is the failure rate threshold (0.0-1.0). The breaker opens only
	// when failures >= FailThreshold AND failure rate >= FailRate.
	FailRate float64
}

// countingBreaker implements Breaker with a sliding failure window.
type countingBreaker struct {
	mu            sync.RWMutex
	failures      []time.Time
	successes     []time.Time
	state         State
	openUntil     time.Time
	window        time.Duration
	openDuration  time.Duration
	failThreshold int
	failRate      float64
}

// New creates a circuit breaker with the supplied options.
func New(opts Options) Breaker {
	if opts.FailThreshold <= 0 {
		opts.FailThreshold = 20
	}
	if opts.WindowSec <= 0 {
		opts.WindowSec = 10
	}
	if opts.OpenSec <= 0 {
		opts.OpenSec = 30
	}
	if opts.FailRate <= 0 {
		opts.FailRate = 0.5
	}
	return &countingBreaker{
		window:        time.Duration(opts.WindowSec) * time.Second,
		openDuration:  time.Duration(opts.OpenSec) * time.Second,
		failThreshold: opts.FailThreshold,
		failRate:      opts.FailRate,
	}
}

// Allow reports whether a request may proceed.
func (b *countingBreaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	now := time.Now()
	b.trim(now)

	switch b.state {
	case StateOpen:
		if now.After(b.openUntil) {
			b.state = StateHalfOpen
			return true
		}
		return false
	case StateHalfOpen:
		return true
	default: // StateClosed
		return true
	}
}

// RecordSuccess records a successful call.
func (b *countingBreaker) RecordSuccess() {
	b.mu.Lock()
	defer b.mu.Unlock()

	now := time.Now()
	b.trim(now)
	b.successes = append(b.successes, now)

	if b.state == StateHalfOpen {
		b.state = StateClosed
		b.failures = b.failures[:0]
		b.successes = b.successes[:0]
	}
}

// RecordFailure records a failed call and may open the breaker.
func (b *countingBreaker) RecordFailure() {
	b.mu.Lock()
	defer b.mu.Unlock()

	now := time.Now()
	b.trim(now)
	b.failures = append(b.failures, now)

	if b.state == StateHalfOpen {
		b.open(now)
		return
	}

	total := len(b.failures) + len(b.successes)
	if total == 0 {
		return
	}
	failureRate := float64(len(b.failures)) / float64(total)
	if len(b.failures) >= b.failThreshold && failureRate >= b.failRate {
		b.open(now)
	}
}

// State returns the current breaker state.
func (b *countingBreaker) State() State {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.state
}

func (b *countingBreaker) open(now time.Time) {
	b.state = StateOpen
	b.openUntil = now.Add(b.openDuration)
	b.failures = b.failures[:0]
	b.successes = b.successes[:0]
}

// trim removes entries outside the observation window.
func (b *countingBreaker) trim(now time.Time) {
	cutoff := now.Add(-b.window)
	b.failures = trimSlice(b.failures, cutoff)
	b.successes = trimSlice(b.successes, cutoff)
}

func trimSlice(times []time.Time, cutoff time.Time) []time.Time {
	i := 0
	for i < len(times) && times[i].Before(cutoff) {
		i++
	}
	if i == 0 {
		return times
	}
	return append(times[:0], times[i:]...)
}

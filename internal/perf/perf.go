package perf

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// Profile collects timing spans for a single run.
// Safe for concurrent use.
type Profile struct {
	mu    sync.Mutex
	spans []Span
}

// Span records the duration of one operation.
type Span struct {
	Name     string  `json:"name"`
	Parent   string  `json:"parent,omitempty"`
	Duration float64 `json:"duration_ms"`
}

// New returns a new Profile.
func New() *Profile {
	return &Profile{}
}

// Start begins timing a named span. Call the returned function to end it.
// Parent is optional context (e.g. the section name).
func (p *Profile) Start(name, parent string) func() {
	start := time.Now()
	return func() {
		d := time.Since(start)
		p.mu.Lock()
		p.spans = append(p.spans, Span{
			Name:     name,
			Parent:   parent,
			Duration: float64(d.Microseconds()) / 1000.0,
		})
		p.mu.Unlock()
	}
}

// Spans returns a copy of all collected spans.
func (p *Profile) Spans() []Span {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]Span, len(p.spans))
	copy(out, p.spans)
	return out
}

// WriteFile writes the profile as JSON to the given path.
func (p *Profile) WriteFile(path string) error {
	data, err := json.MarshalIndent(p.Spans(), "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling profile: %w", err)
	}
	return os.WriteFile(path, append(data, '\n'), 0644)
}

// Print writes a human-readable summary to stderr.
func (p *Profile) Print() {
	spans := p.Spans()
	if len(spans) == 0 {
		return
	}

	fmt.Fprintln(os.Stderr, "\n--- performance profile ---")

	// Group by parent
	type group struct {
		parent string
		spans  []Span
	}
	order := []string{}
	groups := map[string]*group{}

	for _, s := range spans {
		key := s.Parent
		if key == "" {
			key = "(top)"
		}
		g, ok := groups[key]
		if !ok {
			g = &group{parent: key}
			groups[key] = g
			order = append(order, key)
		}
		g.spans = append(g.spans, s)
	}

	for _, key := range order {
		g := groups[key]
		if key != "(top)" {
			fmt.Fprintf(os.Stderr, "  [%s]\n", g.parent)
		}
		for _, s := range g.spans {
			fmt.Fprintf(os.Stderr, "    %-40s %8.1f ms\n", s.Name, s.Duration)
		}
	}
	fmt.Fprintln(os.Stderr, "---")
}

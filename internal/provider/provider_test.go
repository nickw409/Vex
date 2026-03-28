package provider

import (
	"fmt"
	"testing"

	"github.com/nickw409/vex/internal/config"
)

func TestNewClaudeCLI(t *testing.T) {
	cfg := &config.Config{Provider: "claude-cli", Model: "sonnet"}
	p, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}

	cli, ok := p.(*ClaudeCLI)
	if !ok {
		t.Fatal("expected *ClaudeCLI")
	}
	if cli.Model != "sonnet" {
		t.Errorf("expected model sonnet, got %s", cli.Model)
	}
}

func TestNewUnknownProvider(t *testing.T) {
	cfg := &config.Config{Provider: "unknown"}
	_, err := New(cfg)
	if err == nil {
		t.Error("expected error for unknown provider")
	}
}

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		err  string
		want bool
	}{
		{"claude cli failed: rate_limit_error", true},
		{"claude cli failed: Rate limit reached", true},
		{"claude cli failed: 429 {\"type\":\"error\"}", true},
		{"claude cli failed: 529 overloaded", true},
		{"claude cli failed: API is temporarily overloaded", true},
		{"parsing check response: invalid json", false},
		{"pass 1 check failed: claude cli not found", false},
		{"", false},
	}
	for _, tt := range tests {
		got := IsRetryable(fmt.Errorf("%s", tt.err))
		if got != tt.want {
			t.Errorf("IsRetryable(%q) = %v, want %v", tt.err, got, tt.want)
		}
	}

	if IsRetryable(nil) {
		t.Error("IsRetryable(nil) should be false")
	}
}

package provider

import (
	"context"
	"fmt"
	"strings"

	"github.com/nickw409/vex/internal/config"
)

// IsRetryable reports whether err is a transient error (rate limit or
// overload) that may succeed if retried after a delay.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "rate_limit") ||
		strings.Contains(msg, "rate limit") ||
		strings.Contains(msg, "overloaded") ||
		strings.Contains(msg, "429") ||
		strings.Contains(msg, "529")
}

type CompletionRequest struct {
	SystemPrompt string
	UserPrompt   string
	MaxTokens    int
}

type CompletionResponse struct {
	Content string
	Usage   TokenUsage
}

type TokenUsage struct {
	InputTokens  int
	OutputTokens int
	CostUSD      float64
	DurationMS   int
}

type Provider interface {
	Complete(ctx context.Context, req CompletionRequest) (CompletionResponse, error)
}

func New(cfg *config.Config) (Provider, error) {
	return NewWithModel(cfg, cfg.Model)
}

// NewWithModel creates a provider using the given model override instead of
// the config's default model. Used to create a cheaper provider for pass 1.
func NewWithModel(cfg *config.Config, model string) (Provider, error) {
	switch cfg.Provider {
	case "claude-cli":
		return &ClaudeCLI{Model: model}, nil
	default:
		return nil, fmt.Errorf("unknown provider: %s", cfg.Provider)
	}
}

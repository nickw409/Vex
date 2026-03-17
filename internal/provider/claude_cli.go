package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"

	"github.com/nickw409/vex/internal/log"
)

type ClaudeCLI struct {
	Model string
}

type claudeJSONOutput struct {
	Result       string     `json:"result"`
	TotalCostUSD float64    `json:"total_cost_usd"`
	DurationMS   int        `json:"duration_ms"`
	Usage        claudeUsage `json:"usage"`
}

type claudeUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

func (c *ClaudeCLI) Complete(ctx context.Context, req CompletionRequest) (CompletionResponse, error) {
	cmd := c.buildCmd(ctx, req)
	cmd.Stdin = strings.NewReader(req.UserPrompt)

	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return CompletionResponse{}, fmt.Errorf("claude cli failed: %s", string(exitErr.Stderr))
		}
		if errors.Is(err, exec.ErrNotFound) {
			return CompletionResponse{}, fmt.Errorf("claude cli not found; install from https://docs.anthropic.com/en/docs/claude-code")
		}
		return CompletionResponse{}, fmt.Errorf("running claude cli: %w", err)
	}

	return parseResponse(out)
}

func (c *ClaudeCLI) buildCmd(ctx context.Context, req CompletionRequest) *exec.Cmd {
	args := []string{
		"--print",
		"--output-format", "json",
		"--model", c.Model,
		"--system-prompt", req.SystemPrompt,
	}

	return exec.CommandContext(ctx, "claude", args...)
}

func parseResponse(data []byte) (CompletionResponse, error) {
	var out claudeJSONOutput
	if err := json.Unmarshal(data, &out); err != nil {
		return CompletionResponse{}, fmt.Errorf("parsing claude response: %w", err)
	}

	usage := TokenUsage{
		InputTokens:  out.Usage.InputTokens + out.Usage.CacheCreationInputTokens + out.Usage.CacheReadInputTokens,
		OutputTokens: out.Usage.OutputTokens,
		CostUSD:      out.TotalCostUSD,
		DurationMS:   out.DurationMS,
	}

	log.Info("tokens: %d in / %d out | cost: $%.4f | time: %.1fs",
		usage.InputTokens, usage.OutputTokens, usage.CostUSD, float64(usage.DurationMS)/1000)

	return CompletionResponse{
		Content: out.Result,
		Usage:   usage,
	}, nil
}

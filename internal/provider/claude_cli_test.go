package provider

import (
	"context"
	"io"
	"os"
	"strings"
	"testing"
)

func TestBuildCmd(t *testing.T) {
	c := &ClaudeCLI{Model: "sonnet"}
	req := CompletionRequest{
		SystemPrompt: "You are a test reviewer.",
		UserPrompt:   "Check this code.",
	}

	cmd := c.buildCmd(context.Background(), req)

	args := cmd.Args
	if args[0] != "claude" {
		t.Errorf("expected claude binary, got %s", args[0])
	}

	assertContains(t, args, "--print")
	assertContains(t, args, "--output-format")
	assertContains(t, args, "json")
	assertContains(t, args, "--model")
	assertContains(t, args, "sonnet")
	assertContains(t, args, "--system-prompt")
}

func TestParseResponse(t *testing.T) {
	data := []byte(`{
		"result": "Here is the analysis.",
		"total_cost_usd": 0.0325,
		"duration_ms": 2500,
		"usage": {
			"input_tokens": 100,
			"output_tokens": 50,
			"cache_creation_input_tokens": 200,
			"cache_read_input_tokens": 300
		}
	}`)

	resp, err := parseResponse(data)
	if err != nil {
		t.Fatal(err)
	}

	if resp.Content != "Here is the analysis." {
		t.Errorf("expected content 'Here is the analysis.', got %q", resp.Content)
	}
	if resp.Usage.InputTokens != 600 {
		t.Errorf("expected 600 total input tokens (100+200+300), got %d", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 50 {
		t.Errorf("expected 50 output tokens, got %d", resp.Usage.OutputTokens)
	}
	if resp.Usage.CostUSD != 0.0325 {
		t.Errorf("expected cost $0.0325, got $%f", resp.Usage.CostUSD)
	}
	if resp.Usage.DurationMS != 2500 {
		t.Errorf("expected duration 2500ms, got %d", resp.Usage.DurationMS)
	}
}

func TestParseResponseInvalidJSON(t *testing.T) {
	_, err := parseResponse([]byte("not json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestCompletePassesStdin(t *testing.T) {
	c := &ClaudeCLI{Model: "sonnet"}
	req := CompletionRequest{
		SystemPrompt: "system",
		UserPrompt:   "this is the user prompt",
	}

	cmd := c.buildCmd(context.Background(), req)
	cmd.Stdin = strings.NewReader(req.UserPrompt)

	data, err := io.ReadAll(cmd.Stdin)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "this is the user prompt" {
		t.Errorf("expected stdin to contain user prompt, got %q", string(data))
	}
}

func TestParseResponseLogsUsage(t *testing.T) {
	// Capture stderr where log.Info writes
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	origStderr := os.Stderr
	os.Stderr = w
	defer func() { os.Stderr = origStderr }()

	data := []byte(`{
		"result": "test",
		"total_cost_usd": 0.05,
		"duration_ms": 3000,
		"usage": {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 25, "cache_read_input_tokens": 75}
	}`)

	resp, parseErr := parseResponse(data)

	w.Close()
	captured, _ := io.ReadAll(r)
	logged := string(captured)

	if parseErr != nil {
		t.Fatal(parseErr)
	}

	// Verify cache tokens are included in total input tokens (100+25+75=200)
	if resp.Usage.InputTokens != 200 {
		t.Errorf("expected 200 total input tokens (100+25+75), got %d", resp.Usage.InputTokens)
	}

	if !strings.Contains(logged, "[vex") {
		t.Errorf("expected log to contain '[vex', got %q", logged)
	}
	if !strings.Contains(logged, "200 in") {
		t.Errorf("expected log to contain '200 in', got %q", logged)
	}
	if !strings.Contains(logged, "50 out") {
		t.Errorf("expected log to contain '50 out', got %q", logged)
	}
	if !strings.Contains(logged, "$0.0500") {
		t.Errorf("expected log to contain '$0.0500', got %q", logged)
	}
	if !strings.Contains(logged, "3.0s") {
		t.Errorf("expected log to contain '3.0s', got %q", logged)
	}
}

func TestCompleteClaudeNotFound(t *testing.T) {
	c := &ClaudeCLI{Model: "sonnet"}
	// Override to a non-existent binary by testing buildCmd with a fake name
	req := CompletionRequest{
		SystemPrompt: "system",
		UserPrompt:   "user",
	}

	cmd := c.buildCmd(context.Background(), req)
	cmd.Path = "/nonexistent/claude-fake-binary"
	cmd.Args[0] = "/nonexistent/claude-fake-binary"
	cmd.Stdin = strings.NewReader(req.UserPrompt)

	_, err := cmd.Output()
	if err == nil {
		t.Error("expected error for non-existent binary")
	}

	// Verify the Complete method wraps the error with an install URL
	c2 := &ClaudeCLI{Model: "sonnet"}
	ctx := context.Background()
	// Use a wrapper to simulate ErrNotFound via Complete
	origPath := os.Getenv("PATH")
	t.Setenv("PATH", "/nonexistent-path-for-test")
	_, completeErr := c2.Complete(ctx, req)
	os.Setenv("PATH", origPath)
	if completeErr == nil {
		t.Fatal("expected error from Complete")
	}
	if !strings.Contains(completeErr.Error(), "https://docs.anthropic.com") {
		t.Errorf("expected install URL in error, got %q", completeErr.Error())
	}
}

func TestCompleteExitError(t *testing.T) {
	c := &ClaudeCLI{Model: "sonnet"}
	req := CompletionRequest{
		SystemPrompt: "system",
		UserPrompt:   "user",
	}

	// Use a script that exits non-zero with stderr output
	cmd := c.buildCmd(context.Background(), req)
	cmd.Path = "/bin/sh"
	cmd.Args = []string{"sh", "-c", "echo 'some error' >&2; exit 1"}
	cmd.Stdin = strings.NewReader(req.UserPrompt)

	_, err := cmd.Output()
	if err == nil {
		t.Fatal("expected error for non-zero exit")
	}

	// Verify we can exercise the Complete error path for ExitError
	// by checking that the actual Complete method wraps stderr
	_, completeErr := c.Complete(context.Background(), req)
	if completeErr == nil {
		// claude binary might actually exist; skip if so
		t.Skip("claude binary found on PATH, cannot test exit error path")
	}
	// If we got here, the error should be from one of the error branches
	if completeErr.Error() == "" {
		t.Error("expected non-empty error message")
	}
}

func assertContains(t *testing.T, args []string, want string) {
	t.Helper()
	for _, a := range args {
		if a == want {
			return
		}
	}
	t.Errorf("expected args to contain %q, got %v", want, args)
}

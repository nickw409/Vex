package spec

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/nickw409/vex/internal/provider"
)

type mockProvider struct {
	response string
	err      error
}

func (m *mockProvider) Complete(ctx context.Context, req provider.CompletionRequest) (provider.CompletionResponse, error) {
	if m.err != nil {
		return provider.CompletionResponse{}, m.err
	}
	return provider.CompletionResponse{Content: m.response}, nil
}

type mockProviderFunc struct {
	fn func(ctx context.Context, req provider.CompletionRequest) (provider.CompletionResponse, error)
}

func (m *mockProviderFunc) Complete(ctx context.Context, req provider.CompletionRequest) (provider.CompletionResponse, error) {
	return m.fn(ctx, req)
}

func TestBuildSectionValidatePrompt(t *testing.T) {
	sec := Section{
		Name:        "Auth",
		Description: "Authentication module",
		Shared:      []string{"error-handling"},
		Behaviors: []Behavior{
			{Name: "login", Description: "POST /login returns JWT"},
		},
		Subsections: []Subsection{
			{
				Name: "Token Refresh",
				Behaviors: []Behavior{
					{Name: "refresh", Description: "POST /refresh returns new token"},
				},
			},
		},
	}
	shared := []Behavior{
		{Name: "error-handling", Description: "Structured errors"},
	}

	prompt := buildSectionValidatePrompt("MyApp", "Test application", &sec, shared)

	for _, want := range []string{
		"MyApp",
		"Test application",
		"error-handling",
		"Auth",
		"login",
		"Token Refresh",
		"refresh",
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt should contain %q", want)
		}
	}
}

func TestParseValidationResponse(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		complete bool
		count    int
	}{
		{
			name:     "complete",
			input:    `{"complete": true, "suggestions": []}`,
			complete: true,
			count:    0,
		},
		{
			name:     "incomplete",
			input:    `{"complete": false, "suggestions": [{"section": "Auth", "behavior_name": "revocation", "description": "Token revocation", "relation": "new"}]}`,
			complete: false,
			count:    1,
		},
		{
			name:     "with markdown fences",
			input:    "```json\n{\"complete\": true, \"suggestions\": []}\n```",
			complete: true,
			count:    0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parseValidationResponse(tt.input)
			if err != nil {
				t.Fatal(err)
			}
			if result.Complete != tt.complete {
				t.Errorf("expected complete=%v, got %v", tt.complete, result.Complete)
			}
			if len(result.Suggestions) != tt.count {
				t.Errorf("expected %d suggestions, got %d", tt.count, len(result.Suggestions))
			}
		})
	}
}

func TestParseValidationResponseFields(t *testing.T) {
	input := `{"complete": false, "suggestions": [{"section": "Auth", "behavior_name": "revocation", "description": "Token revocation flow", "relation": "new"}]}`
	result, err := parseValidationResponse(input)
	if err != nil {
		t.Fatal(err)
	}
	s := result.Suggestions[0]
	if s.Section != "Auth" {
		t.Errorf("expected section 'Auth', got %q", s.Section)
	}
	if s.BehaviorName != "revocation" {
		t.Errorf("expected behavior_name 'revocation', got %q", s.BehaviorName)
	}
	if s.Relation != "new" {
		t.Errorf("expected relation 'new', got %q", s.Relation)
	}
}

func TestParseValidationResponseInvalid(t *testing.T) {
	_, err := parseValidationResponse("not json at all")
	if err == nil {
		t.Error("expected error for invalid response")
	}
}

func TestValidateProjectEndToEnd(t *testing.T) {
	mock := &mockProvider{
		response: `{"complete": true, "suggestions": []}`,
	}

	ps := &ProjectSpec{
		Project: "MyApp",
		Sections: []Section{
			{
				Name:        "Auth",
				Description: "Auth module",
				Behaviors: []Behavior{
					{Name: "login", Description: "Login endpoint"},
				},
			},
		},
	}

	result, err := ValidateProject(context.Background(), mock, ps)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Complete {
		t.Error("expected complete=true")
	}
	if len(result.Suggestions) != 0 {
		t.Errorf("expected 0 suggestions, got %d", len(result.Suggestions))
	}
}

func TestValidateProjectWithSuggestions(t *testing.T) {
	mock := &mockProvider{
		response: `{
			"complete": false,
			"suggestions": [
				{
					"section": "Auth",
					"behavior_name": "logout",
					"description": "What happens when user logs out",
					"relation": "new"
				},
				{
					"section": "Auth",
					"behavior_name": "login",
					"description": "Missing test for expired credentials",
					"relation": "extends login"
				}
			]
		}`,
	}

	ps := &ProjectSpec{
		Project: "MyApp",
		Sections: []Section{
			{
				Name:        "Auth",
				Description: "Auth module",
				Behaviors: []Behavior{
					{Name: "login", Description: "Login endpoint"},
				},
			},
		},
	}

	result, err := ValidateProject(context.Background(), mock, ps)
	if err != nil {
		t.Fatal(err)
	}
	if result.Complete {
		t.Error("expected complete=false")
	}
	if len(result.Suggestions) != 2 {
		t.Fatalf("expected 2 suggestions, got %d", len(result.Suggestions))
	}

	s0 := result.Suggestions[0]
	if s0.Section != "Auth" {
		t.Errorf("suggestion 0: expected section Auth, got %s", s0.Section)
	}
	if s0.BehaviorName != "logout" {
		t.Errorf("suggestion 0: expected behavior_name logout, got %s", s0.BehaviorName)
	}
	if s0.Relation != "new" {
		t.Errorf("suggestion 0: expected relation new, got %s", s0.Relation)
	}

	s1 := result.Suggestions[1]
	if s1.Relation != "extends login" {
		t.Errorf("suggestion 1: expected relation 'extends login', got %s", s1.Relation)
	}
}

func TestValidateProjectProviderError(t *testing.T) {
	mock := &mockProvider{
		err: fmt.Errorf("provider unavailable"),
	}

	ps := &ProjectSpec{
		Project: "MyApp",
		Sections: []Section{
			{
				Name:        "Auth",
				Description: "Auth module",
				Behaviors: []Behavior{
					{Name: "login", Description: "Login endpoint"},
				},
			},
		},
	}

	_, err := ValidateProject(context.Background(), mock, ps)
	if err == nil {
		t.Error("expected error when provider returns error")
	}
	if !strings.Contains(err.Error(), "provider unavailable") {
		t.Errorf("expected error to contain 'provider unavailable', got %q", err.Error())
	}
}

func TestValidateProjectConcurrentSections(t *testing.T) {
	var mu sync.Mutex
	var calls int
	mock := &mockProviderFunc{
		fn: func(ctx context.Context, req provider.CompletionRequest) (provider.CompletionResponse, error) {
			mu.Lock()
			calls++
			mu.Unlock()

			if strings.Contains(req.UserPrompt, "Storage") {
				return provider.CompletionResponse{
					Content: `{"complete": false, "suggestions": [{"section": "Storage", "behavior_name": "delete", "description": "Delete files", "relation": "new"}]}`,
				}, nil
			}
			return provider.CompletionResponse{
				Content: `{"complete": true, "suggestions": []}`,
			}, nil
		},
	}

	ps := &ProjectSpec{
		Project: "MyApp",
		Sections: []Section{
			{
				Name:        "Auth",
				Description: "Auth module",
				Behaviors:   []Behavior{{Name: "login", Description: "Login"}},
			},
			{
				Name:        "Storage",
				Description: "File storage",
				Behaviors:   []Behavior{{Name: "upload", Description: "Upload files"}},
			},
			{
				Name:        "API",
				Description: "REST API",
				Behaviors:   []Behavior{{Name: "list", Description: "List items"}},
			},
		},
	}

	result, err := ValidateProject(context.Background(), mock, ps, 3)
	if err != nil {
		t.Fatal(err)
	}

	if result.Complete {
		t.Error("expected complete=false (Storage has suggestions)")
	}
	if len(result.Suggestions) != 1 {
		t.Fatalf("expected 1 suggestion, got %d", len(result.Suggestions))
	}
	if result.Suggestions[0].Section != "Storage" {
		t.Errorf("expected suggestion for Storage, got %q", result.Suggestions[0].Section)
	}

	mu.Lock()
	if calls != 3 {
		t.Errorf("expected 3 LLM calls (one per section), got %d", calls)
	}
	mu.Unlock()
}

func TestValidateSystemPromptFormulaTolerance(t *testing.T) {
	if !strings.Contains(validateSystemPrompt, "Mathematical formulas and equations ARE valid behaviors") {
		t.Error("validateSystemPrompt should state that formulas are valid behaviors")
	}
}

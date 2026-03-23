package spec

import (
	"context"
	"fmt"
	"strings"
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

func TestBuildProjectValidatePrompt(t *testing.T) {
	ps := &ProjectSpec{
		Project:     "MyApp",
		Description: "Test application",
		Shared: []Behavior{
			{Name: "error-handling", Description: "Structured errors"},
		},
		Sections: []Section{
			{
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
			},
		},
	}

	prompt := buildProjectValidatePrompt(ps)

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

func TestValidateSystemPromptFormulaTolerance(t *testing.T) {
	if !strings.Contains(validateSystemPrompt, "Mathematical formulas and equations ARE valid behaviors") {
		t.Error("validateSystemPrompt should state that formulas are valid behaviors")
	}
}

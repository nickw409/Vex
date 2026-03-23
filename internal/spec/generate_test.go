package spec

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/nickw409/vex/internal/provider"
)

type mockGenProvider struct {
	response string
	err      error
}

func (m *mockGenProvider) Complete(ctx context.Context, req provider.CompletionRequest) (provider.CompletionResponse, error) {
	if m.err != nil {
		return provider.CompletionResponse{}, m.err
	}
	return provider.CompletionResponse{Content: m.response}, nil
}

func TestParseGenerateResponse(t *testing.T) {
	content := `- name: Auth
  path: internal/auth
  description: |
    JWT authentication module.
  behaviors:
    - name: login
      description: |
        POST /login accepts credentials and returns JWT.
    - name: logout
      description: |
        POST /logout invalidates the session.
`

	sections, err := parseGenerateResponse(content)
	if err != nil {
		t.Fatal(err)
	}

	if len(sections) != 1 {
		t.Fatalf("expected 1 section, got %d", len(sections))
	}
	if sections[0].Name != "Auth" {
		t.Errorf("expected name 'Auth', got %q", sections[0].Name)
	}
	if len(sections[0].Behaviors) != 2 {
		t.Errorf("expected 2 behaviors, got %d", len(sections[0].Behaviors))
	}
}

func TestParseGenerateResponseWithFences(t *testing.T) {
	content := "```yaml\n- name: Core\n  path: src/core\n  description: |\n    Core module.\n  behaviors:\n    - name: process\n      description: |\n        Processes data.\n```"

	sections, err := parseGenerateResponse(content)
	if err != nil {
		t.Fatal(err)
	}
	if sections[0].Name != "Core" {
		t.Errorf("expected name 'Core', got %q", sections[0].Name)
	}
}

func TestParseGenerateResponseInvalid(t *testing.T) {
	_, err := parseGenerateResponse("not yaml at all {{{")
	if err == nil {
		t.Error("expected error for invalid YAML")
	}
}

func TestParseGenerateResponseMissingName(t *testing.T) {
	content := `- path: src/core
  description: Core
  behaviors:
    - name: process
      description: Does things
`
	_, err := parseGenerateResponse(content)
	if err == nil {
		t.Error("expected error for missing section name")
	}
}

func TestParseGenerateResponseMultipleSections(t *testing.T) {
	content := `- name: Auth
  path: internal/auth
  description: Auth module
  behaviors:
    - name: login
      description: Login endpoint
- name: API
  path: internal/api
  description: API layer
  behaviors:
    - name: routing
      description: Request routing
`

	sections, err := parseGenerateResponse(content)
	if err != nil {
		t.Fatal(err)
	}
	if len(sections) != 2 {
		t.Errorf("expected 2 sections, got %d", len(sections))
	}
}

func TestParseGenerateResponseWithSubsections(t *testing.T) {
	content := `- name: App Server
  path: app
  description: Main server
  behaviors:
    - name: websocket
      description: WebSocket handling
  subsections:
    - name: Auth Handlers
      file: app/handlers/auth.go
      behaviors:
        - name: login-handler
          description: Login request handler
`

	sections, err := parseGenerateResponse(content)
	if err != nil {
		t.Fatal(err)
	}
	if len(sections[0].Subsections) != 1 {
		t.Errorf("expected 1 subsection, got %d", len(sections[0].Subsections))
	}
	if sections[0].Subsections[0].Name != "Auth Handlers" {
		t.Errorf("expected subsection 'Auth Handlers', got %q", sections[0].Subsections[0].Name)
	}
}

func TestParseExtendResponse(t *testing.T) {
	content := `behaviors:
  - name: token-revocation
    description: |
      POST /revoke invalidates a token.
subsections:
  - name: Admin Auth
    path: internal/auth/admin/
    behaviors:
      - name: admin-login
        description: |
          Admin login with elevated privileges.
`

	result, err := parseExtendResponse(content)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Behaviors) != 1 {
		t.Errorf("expected 1 behavior, got %d", len(result.Behaviors))
	}
	if result.Behaviors[0].Name != "token-revocation" {
		t.Errorf("expected 'token-revocation', got %q", result.Behaviors[0].Name)
	}
	if len(result.Subsections) != 1 {
		t.Errorf("expected 1 subsection, got %d", len(result.Subsections))
	}
}

func TestParseExtendResponseBehaviorsOnly(t *testing.T) {
	content := `behaviors:
  - name: logout
    description: Invalidates session
`

	result, err := parseExtendResponse(content)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Behaviors) != 1 {
		t.Errorf("expected 1 behavior, got %d", len(result.Behaviors))
	}
	if len(result.Subsections) != 0 {
		t.Errorf("expected 0 subsections, got %d", len(result.Subsections))
	}
}

func TestParseExtendResponseInvalid(t *testing.T) {
	_, err := parseExtendResponse("not yaml {{{")
	if err == nil {
		t.Error("expected error for invalid YAML")
	}
}

func TestBuildExtendPrompt(t *testing.T) {
	section := &Section{
		Name:        "Auth",
		Description: "Authentication module",
		Path:        PathList{"internal/auth"},
		Behaviors: []Behavior{
			{Name: "login", Description: "Login endpoint"},
		},
		Subsections: []Subsection{
			{
				Name: "Token",
				Behaviors: []Behavior{
					{Name: "refresh", Description: "Token refresh"},
				},
			},
		},
	}

	prompt := buildExtendPrompt(section, "Add logout and token revocation")

	for _, want := range []string{"Auth", "login", "Token", "refresh", "Add logout"} {
		if !containsString(prompt, want) {
			t.Errorf("prompt should contain %q", want)
		}
	}
}

func containsString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestGenerateEndToEnd(t *testing.T) {
	mock := &mockGenProvider{
		response: `- name: Auth
  path: internal/auth
  description: |
    Authentication module.
  behaviors:
    - name: login
      description: |
        POST /login returns JWT.
`,
	}

	sections, err := Generate(context.Background(), mock, "Build an auth module with login")
	if err != nil {
		t.Fatal(err)
	}
	if len(sections) != 1 {
		t.Fatalf("expected 1 section, got %d", len(sections))
	}
	if sections[0].Name != "Auth" {
		t.Errorf("expected section name 'Auth', got %q", sections[0].Name)
	}
	if len(sections[0].Behaviors) != 1 {
		t.Errorf("expected 1 behavior, got %d", len(sections[0].Behaviors))
	}
}

func TestGenerateProviderError(t *testing.T) {
	mock := &mockGenProvider{
		err: fmt.Errorf("connection refused"),
	}

	_, err := Generate(context.Background(), mock, "Build an auth module")
	if err == nil {
		t.Error("expected error when provider returns error")
	}
	if !strings.Contains(err.Error(), "connection refused") {
		t.Errorf("expected error to contain 'connection refused', got %q", err.Error())
	}
}

func TestGenerateExtendEndToEnd(t *testing.T) {
	mock := &mockGenProvider{
		response: `behaviors:
  - name: logout
    description: |
      POST /logout invalidates the session.
`,
	}

	section := &Section{
		Name:        "Auth",
		Description: "Authentication module",
		Path:        PathList{"internal/auth"},
		Behaviors: []Behavior{
			{Name: "login", Description: "Login endpoint"},
		},
	}

	result, err := GenerateExtend(context.Background(), mock, section, "Add logout")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Behaviors) != 1 {
		t.Fatalf("expected 1 behavior, got %d", len(result.Behaviors))
	}
	if result.Behaviors[0].Name != "logout" {
		t.Errorf("expected behavior name 'logout', got %q", result.Behaviors[0].Name)
	}
}

func TestGenerateExtendProviderError(t *testing.T) {
	mock := &mockGenProvider{
		err: fmt.Errorf("rate limited"),
	}

	section := &Section{
		Name:        "Auth",
		Description: "Authentication module",
	}

	_, err := GenerateExtend(context.Background(), mock, section, "Add logout")
	if err == nil {
		t.Error("expected error when provider returns error")
	}
	if !strings.Contains(err.Error(), "rate limited") {
		t.Errorf("expected error to contain 'rate limited', got %q", err.Error())
	}
}

func TestParseGenerateResponseBehaviorMissingDescription(t *testing.T) {
	content := `- name: Auth
  path: internal/auth
  description: Auth module
  behaviors:
    - name: login
`
	_, err := parseGenerateResponse(content)
	if err == nil {
		t.Error("expected error for behavior missing description")
	}
}

func TestParseGenerateResponseSubsectionBehaviorMissingName(t *testing.T) {
	content := `- name: Auth
  path: internal/auth
  description: Auth module
  behaviors:
    - name: login
      description: Login endpoint
  subsections:
    - name: Token
      path: internal/auth/token
      behaviors:
        - description: Refresh token without a name
`
	_, err := parseGenerateResponse(content)
	if err == nil {
		t.Error("expected error for subsection behavior missing name")
	}
}

func TestGenerateSystemPromptFormulaSupport(t *testing.T) {
	if !strings.Contains(generateSystemPrompt, "mathematical formula") {
		t.Error("generateSystemPrompt should accept mathematical formulas as valid behaviors")
	}
	if !strings.Contains(generateSystemPrompt, "formula") {
		t.Error("generateSystemPrompt should mention formulas in behavior IS section")
	}
}

func TestExtendSystemPromptFormulaSupport(t *testing.T) {
	if !strings.Contains(generateExtendSystemPrompt, "mathematical formula") {
		t.Error("generateExtendSystemPrompt should accept mathematical formulas as valid behaviors")
	}
}

package check

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/nwiley/vex/internal/provider"
	"github.com/nwiley/vex/internal/spec"
)

type mockProvider struct {
	response      string
	err           error
	mu            sync.Mutex
	calls         int
	maxConcurrent int
	concurrent    int
}

func (m *mockProvider) Complete(ctx context.Context, req provider.CompletionRequest) (provider.CompletionResponse, error) {
	m.mu.Lock()
	m.calls++
	m.concurrent++
	if m.concurrent > m.maxConcurrent {
		m.maxConcurrent = m.concurrent
	}
	m.mu.Unlock()

	// Small sleep to test concurrency
	time.Sleep(10 * time.Millisecond)

	m.mu.Lock()
	m.concurrent--
	m.mu.Unlock()

	if m.err != nil {
		return provider.CompletionResponse{}, m.err
	}
	return provider.CompletionResponse{Content: m.response}, nil
}

func makeInput(name string) SectionInput {
	return SectionInput{
		Section: &spec.Section{Name: name, Description: name + " section"},
		Behaviors: []spec.Behavior{
			{Name: name + "-b1", Description: "behavior one"},
		},
		SourceFiles: map[string]string{name + ".go": "package x"},
		TestFiles:   map[string]string{name + "_test.go": "package x"},
	}
}

var validResponse = `{"gaps": [{"behavior": "b1", "detail": "missing", "suggestion": "add test"}], "covered": [{"behavior": "b2", "detail": "ok", "test_file": "a_test.go", "test_name": "TestA"}]}`

func TestRunProjectBasic(t *testing.T) {
	mp := &mockProvider{response: validResponse}
	inputs := []SectionInput{makeInput("sec1"), makeInput("sec2")}
	ps := &spec.ProjectSpec{}

	rpt, err := RunProject(context.Background(), mp, ps, inputs, 2)
	if err != nil {
		t.Fatal(err)
	}

	// Should have called provider once per section
	mp.mu.Lock()
	calls := mp.calls
	mp.mu.Unlock()
	if calls != 2 {
		t.Errorf("expected 2 provider calls, got %d", calls)
	}

	// Results from both sections should be merged
	if len(rpt.Gaps) != 2 {
		t.Errorf("expected 2 gaps (1 per section), got %d", len(rpt.Gaps))
	}
	if len(rpt.Covered) != 2 {
		t.Errorf("expected 2 covered (1 per section), got %d", len(rpt.Covered))
	}

	// Summary should reflect total behaviors from all inputs
	if rpt.BehaviorsChecked != 2 {
		t.Errorf("expected BehaviorsChecked=2, got %d", rpt.BehaviorsChecked)
	}
}

func TestRunProjectMaxConcurrencyDefaultsTo4(t *testing.T) {
	mp := &mockProvider{response: `{"gaps": [], "covered": []}`}
	// Create 6 inputs; concurrency 0 should default to 4 and still complete
	var inputs []SectionInput
	for i := 0; i < 6; i++ {
		inputs = append(inputs, makeInput(fmt.Sprintf("s%d", i)))
	}

	rpt, err := RunProject(context.Background(), mp, &spec.ProjectSpec{}, inputs, 0)
	if err != nil {
		t.Fatal(err)
	}

	mp.mu.Lock()
	calls := mp.calls
	mp.mu.Unlock()
	if calls != 6 {
		t.Errorf("expected 6 calls, got %d", calls)
	}
	if rpt.BehaviorsChecked != 6 {
		t.Errorf("expected 6 behaviors checked, got %d", rpt.BehaviorsChecked)
	}
}

func TestRunProjectBoundedConcurrency(t *testing.T) {
	mp := &mockProvider{response: `{"gaps": [], "covered": []}`}
	var inputs []SectionInput
	for i := 0; i < 8; i++ {
		inputs = append(inputs, makeInput(fmt.Sprintf("s%d", i)))
	}

	_, err := RunProject(context.Background(), mp, &spec.ProjectSpec{}, inputs, 2)
	if err != nil {
		t.Fatal(err)
	}

	mp.mu.Lock()
	maxC := mp.maxConcurrent
	mp.mu.Unlock()
	if maxC > 2 {
		t.Errorf("expected max concurrency <= 2, got %d", maxC)
	}
}

func TestRunProjectPartialError(t *testing.T) {
	// First call succeeds, second fails. Use a provider that fails on the second call.
	failProvider := &sectionFailProvider{
		failSection: "fail",
		response:    validResponse,
	}

	inputs := []SectionInput{
		makeInput("ok"),
		makeInput("fail"),
	}

	rpt, err := RunProject(context.Background(), failProvider, &spec.ProjectSpec{}, inputs, 4)
	if err == nil {
		t.Fatal("expected error when a section fails")
	}

	if !strings.Contains(err.Error(), "1 section(s)") {
		t.Errorf("error should mention 1 failing section, got: %s", err.Error())
	}

	// Partial results from the successful section should still be present
	if len(rpt.Gaps) != 1 {
		t.Errorf("expected 1 gap from successful section, got %d", len(rpt.Gaps))
	}
	if len(rpt.Covered) != 1 {
		t.Errorf("expected 1 covered from successful section, got %d", len(rpt.Covered))
	}
}

// sectionFailProvider fails for sections whose name matches failSection.
type sectionFailProvider struct {
	failSection string
	response    string
}

func (p *sectionFailProvider) Complete(ctx context.Context, req provider.CompletionRequest) (provider.CompletionResponse, error) {
	if strings.Contains(req.UserPrompt, p.failSection) {
		return provider.CompletionResponse{}, fmt.Errorf("provider error for section")
	}
	return provider.CompletionResponse{Content: p.response}, nil
}

func TestBuildSectionPrompt(t *testing.T) {
	input := &SectionInput{
		Section: &spec.Section{
			Name:        "Auth",
			Description: "Authentication module",
		},
		Behaviors: []spec.Behavior{
			{Name: "login", Description: "POST /login returns JWT"},
		},
		SourceFiles: map[string]string{
			"auth.go": "package auth\nfunc Login() {}",
		},
		TestFiles: map[string]string{
			"auth_test.go": "package auth\nfunc TestLogin(t *testing.T) {}",
		},
	}

	prompt, err := buildSectionPrompt(input)
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{"Auth", "login", "POST /login", "auth.go", "auth_test.go", "Authentication module"} {
		if !containsStr(prompt, want) {
			t.Errorf("prompt should contain %q", want)
		}
	}
}

func TestBuildSectionPromptTooLarge(t *testing.T) {
	large := make([]byte, maxContentSize)
	for i := range large {
		large[i] = 'x'
	}

	input := &SectionInput{
		Section: &spec.Section{Name: "Big"},
		Behaviors: []spec.Behavior{
			{Name: "b", Description: "d"},
		},
		SourceFiles: map[string]string{"big.go": string(large)},
		TestFiles:   map[string]string{},
	}

	_, err := buildSectionPrompt(input)
	if err == nil {
		t.Error("expected error for oversized content")
	}
	if err != nil && !strings.Contains(err.Error(), "--diff") {
		t.Errorf("error message should contain '--diff', got: %s", err.Error())
	}
}

func TestParseSectionResponse(t *testing.T) {
	content := `{
  "gaps": [
    {"behavior": "login", "detail": "No expiry test", "suggestion": "Add TestLoginExpiry"}
  ],
  "covered": [
    {"behavior": "login", "detail": "Valid creds", "test_file": "auth_test.go", "test_name": "TestLogin"}
  ]
}`

	gaps, covered, err := parseSectionResponse(content)
	if err != nil {
		t.Fatal(err)
	}

	if len(gaps) != 1 {
		t.Errorf("expected 1 gap, got %d", len(gaps))
	}
	if len(covered) != 1 {
		t.Errorf("expected 1 covered, got %d", len(covered))
	}
}

func TestParseSectionResponseInvalid(t *testing.T) {
	_, _, err := parseSectionResponse("not json")
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseSectionResponseEmpty(t *testing.T) {
	gaps, covered, err := parseSectionResponse(`{"gaps": [], "covered": []}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(gaps) != 0 {
		t.Error("expected no gaps")
	}
	if len(covered) != 0 {
		t.Error("expected no covered")
	}
}

func TestParseSectionResponseNullArrays(t *testing.T) {
	gaps, covered, err := parseSectionResponse(`{"gaps": null, "covered": null}`)
	if err != nil {
		t.Fatal(err)
	}
	if gaps == nil {
		t.Error("gaps should not be nil")
	}
	if covered == nil {
		t.Error("covered should not be nil")
	}
}

func TestParseSectionResponseMarkdownFenced(t *testing.T) {
	content := "```json\n{\"gaps\": [], \"covered\": [{\"behavior\": \"login\", \"detail\": \"tested\", \"test_file\": \"a.go\", \"test_name\": \"TestA\"}]}\n```"

	_, covered, err := parseSectionResponse(content)
	if err != nil {
		t.Fatal(err)
	}
	if len(covered) != 1 {
		t.Errorf("expected 1 covered, got %d", len(covered))
	}
}

func TestParseSectionResponsePreambleBeforeJSON(t *testing.T) {
	content := "Here is my analysis:\n" + `{"gaps": [], "covered": [{"behavior": "login", "detail": "tested", "test_file": "a.go", "test_name": "TestA"}]}`

	_, covered, err := parseSectionResponse(content)
	if err != nil {
		t.Fatalf("should handle preamble text before JSON: %v", err)
	}
	if len(covered) != 1 {
		t.Errorf("expected 1 covered, got %d", len(covered))
	}
}

func TestCheckSystemPromptContainsUnspecified(t *testing.T) {
	if !strings.Contains(checkSystemPrompt, "UNSPECIFIED") {
		t.Error("checkSystemPrompt should contain 'UNSPECIFIED' instruction")
	}
}

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

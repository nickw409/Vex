package check

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/nickw409/vex/internal/provider"
	"github.com/nickw409/vex/internal/report"
	"github.com/nickw409/vex/internal/spec"
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

// gapResponse: pass 1 finds a gap, triggers pass 2
var gapResponse = `{"gaps": [{"behavior": "b1", "detail": "missing", "suggestion": "add test"}], "covered": []}`

// coveredMockProvider returns a response covering whatever behavior name appears in the prompt
type coveredMockProvider struct {
	mockProvider
}

func (m *coveredMockProvider) Complete(ctx context.Context, req provider.CompletionRequest) (provider.CompletionResponse, error) {
	m.mu.Lock()
	m.calls++
	m.concurrent++
	if m.concurrent > m.maxConcurrent {
		m.maxConcurrent = m.concurrent
	}
	m.mu.Unlock()

	time.Sleep(10 * time.Millisecond)

	m.mu.Lock()
	m.concurrent--
	m.mu.Unlock()

	// Extract behavior name from prompt and return it as covered
	// Prompt contains "#### <name>\n"
	resp := `{"gaps": [], "covered": [{"behavior": "matched", "detail": "ok", "test_file": "t.go", "test_name": "T"}]}`
	for _, line := range strings.Split(req.UserPrompt, "\n") {
		if strings.HasPrefix(line, "#### ") {
			name := strings.TrimPrefix(line, "#### ")
			resp = fmt.Sprintf(`{"gaps": [], "covered": [{"behavior": %q, "detail": "ok", "test_file": "t.go", "test_name": "T"}]}`, name)
			break
		}
	}
	return provider.CompletionResponse{Content: resp}, nil
}

func TestRunProjectBasic(t *testing.T) {
	// Use a response with gaps so both passes run
	mp := &mockProvider{response: gapResponse}
	inputs := []SectionInput{makeInput("sec1"), makeInput("sec2")}
	ps := &spec.ProjectSpec{}

	rpt, err := RunProject(context.Background(), mp, ps, inputs, 2, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Should have called provider twice per section (pass 1 + pass 2)
	mp.mu.Lock()
	calls := mp.calls
	mp.mu.Unlock()
	if calls != 4 {
		t.Errorf("expected 4 provider calls (2 passes x 2 sections), got %d", calls)
	}

	// Gaps from pass 2 for both sections
	if len(rpt.Gaps) != 2 {
		t.Errorf("expected 2 gaps (1 per section from pass 2), got %d", len(rpt.Gaps))
	}

	if rpt.Summary.TotalBehaviors != 2 {
		t.Errorf("expected TotalBehaviors=2, got %d", rpt.Summary.TotalBehaviors)
	}
}

func TestRunProjectSkipsPass2WhenCovered(t *testing.T) {
	mp := &coveredMockProvider{}
	inputs := []SectionInput{makeInput("sec1"), makeInput("sec2")}
	ps := &spec.ProjectSpec{}

	rpt, err := RunProject(context.Background(), mp, ps, inputs, 2, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Only pass 1 calls — no pass 2 needed
	mp.mu.Lock()
	calls := mp.calls
	mp.mu.Unlock()
	if calls != 2 {
		t.Errorf("expected 2 provider calls (pass 1 only), got %d", calls)
	}

	if len(rpt.Gaps) != 0 {
		t.Errorf("expected 0 gaps, got %d", len(rpt.Gaps))
	}
	if len(rpt.Covered) != 2 {
		t.Errorf("expected 2 covered (1 per section), got %d", len(rpt.Covered))
	}
}

func TestRunProjectMaxConcurrencyDefaultsTo5(t *testing.T) {
	mp := &coveredMockProvider{}
	// Create 6 inputs; concurrency 0 should default to 5 and still complete
	var inputs []SectionInput
	for i := 0; i < 6; i++ {
		inputs = append(inputs, makeInput(fmt.Sprintf("s%d", i)))
	}

	rpt, err := RunProject(context.Background(), mp, &spec.ProjectSpec{}, inputs, 0, nil)
	if err != nil {
		t.Fatal(err)
	}

	mp.mu.Lock()
	calls := mp.calls
	mp.mu.Unlock()
	if calls != 6 {
		t.Errorf("expected 6 calls, got %d", calls)
	}
	if rpt.Summary.TotalBehaviors != 6 {
		t.Errorf("expected 6 behaviors checked, got %d", rpt.Summary.TotalBehaviors)
	}
}

func TestRunProjectBoundedConcurrency(t *testing.T) {
	mp := &coveredMockProvider{}
	var inputs []SectionInput
	for i := 0; i < 8; i++ {
		inputs = append(inputs, makeInput(fmt.Sprintf("s%d", i)))
	}

	_, err := RunProject(context.Background(), mp, &spec.ProjectSpec{}, inputs, 2, nil)
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
		response:    gapResponse,
	}

	inputs := []SectionInput{
		makeInput("ok"),
		makeInput("fail"),
	}

	rpt, err := RunProject(context.Background(), failProvider, &spec.ProjectSpec{}, inputs, 4, nil)
	if err == nil {
		t.Fatal("expected error when a section fails")
	}

	if !strings.Contains(err.Error(), "1 section(s)") {
		t.Errorf("error should mention 1 failing section, got: %s", err.Error())
	}

	// Partial results from the successful section should still be present
	if len(rpt.Gaps) < 1 {
		t.Errorf("expected at least 1 gap from successful section, got %d", len(rpt.Gaps))
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

	prompt, err := buildPass2Prompt(input)
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

	_, err := buildPass2Prompt(input)
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
	if !strings.Contains(pass2SystemPrompt, "UNSPECIFIED") {
		t.Error("pass2SystemPrompt should contain 'UNSPECIFIED' instruction")
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

func TestFilterFalseUnspecified(t *testing.T) {
	ps := &spec.ProjectSpec{
		Sections: []spec.Section{
			{
				Name: "CLI",
				Behaviors: []spec.Behavior{
					{Name: "check-command", Description: "runs check"},
				},
				Subsections: []spec.Subsection{
					{
						Name: "Drift",
						Behaviors: []spec.Behavior{
							{Name: "drift-command", Description: "runs drift"},
						},
					},
				},
			},
			{
				Name: "Engine",
				Behaviors: []spec.Behavior{
					{Name: "run-project", Description: "runs project"},
				},
			},
		},
	}

	gaps := []report.Gap{
		{Behavior: "UNSPECIFIED", Detail: "The drift command is not covered here"},
		{Behavior: "UNSPECIFIED", Detail: "Something totally new and unknown"},
		{Behavior: "check-command", Detail: "Missing test for X"},
	}

	filtered := filterFalseUnspecified(gaps, ps)

	// drift-command is known, so that UNSPECIFIED should be removed
	// "totally new" has no match, so it stays
	// non-UNSPECIFIED gaps always stay
	if len(filtered) != 2 {
		t.Fatalf("expected 2 gaps after filtering, got %d: %+v", len(filtered), filtered)
	}
	if filtered[0].Behavior != "UNSPECIFIED" || !strings.Contains(filtered[0].Detail, "totally new") {
		t.Errorf("expected unknown UNSPECIFIED gap first, got %+v", filtered[0])
	}
	if filtered[1].Behavior != "check-command" {
		t.Errorf("expected check-command gap second, got %+v", filtered[1])
	}
}

func TestSystemPromptsContainFormulaInstruction(t *testing.T) {
	for _, prompt := range []struct {
		name string
		text string
	}{
		{"pass1SystemPrompt", pass1SystemPrompt},
		{"pass2SystemPrompt", pass2SystemPrompt},
	} {
		if !strings.Contains(prompt.text, "mathematical formula") {
			t.Errorf("%s should contain 'mathematical formula'", prompt.name)
		}
		if !strings.Contains(prompt.text, "known inputs") {
			t.Errorf("%s should contain 'known inputs'", prompt.name)
		}
	}
}

func TestBuildPass1Prompt(t *testing.T) {
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

	prompt, err := buildPass1Prompt(input)
	if err != nil {
		t.Fatal(err)
	}

	// Pass 1 should include section, behaviors, and test files
	for _, want := range []string{"Auth", "login", "POST /login", "auth_test.go", "Authentication module"} {
		if !containsStr(prompt, want) {
			t.Errorf("pass 1 prompt should contain %q", want)
		}
	}

	// Pass 1 should NOT include source files
	if containsStr(prompt, "auth.go") {
		t.Error("pass 1 prompt should not contain source files")
	}
}

func TestBuildPass1PromptTooLarge(t *testing.T) {
	large := make([]byte, maxContentSize)
	for i := range large {
		large[i] = 'x'
	}

	input := &SectionInput{
		Section: &spec.Section{Name: "Big"},
		Behaviors: []spec.Behavior{
			{Name: "b", Description: "d"},
		},
		SourceFiles: map[string]string{},
		TestFiles:   map[string]string{"big_test.go": string(large)},
	}

	_, err := buildPass1Prompt(input)
	if err == nil {
		t.Error("expected error for oversized content")
	}
	if err != nil && !strings.Contains(err.Error(), "--diff") {
		t.Errorf("error message should contain '--diff', got: %s", err.Error())
	}
}

// rateLimitProvider fails with a rate limit error for the first N calls, then succeeds.
type rateLimitProvider struct {
	mu           sync.Mutex
	failCount    int // number of calls that should fail
	calls        int
	successResp  string
}

func (p *rateLimitProvider) Complete(ctx context.Context, req provider.CompletionRequest) (provider.CompletionResponse, error) {
	p.mu.Lock()
	p.calls++
	n := p.calls
	p.mu.Unlock()

	time.Sleep(5 * time.Millisecond)

	if n <= p.failCount {
		return provider.CompletionResponse{}, fmt.Errorf("claude cli failed: 429 rate_limit_error")
	}
	return provider.CompletionResponse{Content: p.successResp}, nil
}

func TestRunProjectRetriesOnRateLimit(t *testing.T) {
	coveredResp := `{"gaps": [], "covered": [{"behavior": "s0-b1", "detail": "ok", "test_file": "t.go", "test_name": "T"}]}`
	rp := &rateLimitProvider{failCount: 1, successResp: coveredResp}

	inputs := []SectionInput{makeInput("s0")}
	rpt, err := RunProject(context.Background(), rp, &spec.ProjectSpec{}, inputs, 2, nil)
	if err != nil {
		t.Fatalf("expected success after retry, got: %v", err)
	}

	rp.mu.Lock()
	calls := rp.calls
	rp.mu.Unlock()

	// First call fails (rate limit), second succeeds
	if calls != 2 {
		t.Errorf("expected 2 calls (1 fail + 1 retry), got %d", calls)
	}
	if len(rpt.Covered) != 1 {
		t.Errorf("expected 1 covered after retry, got %d", len(rpt.Covered))
	}
}

func TestRunProjectExhaustsRetries(t *testing.T) {
	// All calls fail with rate limit — should exhaust retries and report error
	rp := &rateLimitProvider{failCount: 100, successResp: ""}

	inputs := []SectionInput{makeInput("s0")}
	_, err := RunProject(context.Background(), rp, &spec.ProjectSpec{}, inputs, 1, nil)
	if err == nil {
		t.Fatal("expected error when retries exhausted")
	}

	rp.mu.Lock()
	calls := rp.calls
	rp.mu.Unlock()

	// 1 initial + 2 retries = 3 total
	if calls != 3 {
		t.Errorf("expected 3 calls (1 + 2 retries), got %d", calls)
	}
}

func TestFilterFalseUnspecifiedEmptyGaps(t *testing.T) {
	ps := &spec.ProjectSpec{
		Sections: []spec.Section{
			{Name: "CLI", Behaviors: []spec.Behavior{{Name: "x", Description: "y"}}},
		},
	}

	filtered := filterFalseUnspecified([]report.Gap{}, ps)
	if filtered == nil {
		t.Fatal("expected non-nil empty slice")
	}
	if len(filtered) != 0 {
		t.Fatalf("expected 0 gaps, got %d", len(filtered))
	}
}

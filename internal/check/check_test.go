package check

import (
	"testing"

	"github.com/nwiley/vex/internal/spec"
)

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

	for _, want := range []string{"Auth", "login", "POST /login", "auth.go", "auth_test.go"} {
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

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

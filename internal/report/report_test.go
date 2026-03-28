package report

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestJSON(t *testing.T) {
	r := &Report{
		Spec: ".vex/vexspec.yaml",
		Gaps: []Gap{
			{Behavior: "login", Detail: "No expiry test", Suggestion: "Add expiry assertion"},
		},
		Covered: []Covered{
			{Behavior: "login", Detail: "Valid creds return token", TestFile: "auth_test.go", TestName: "TestLoginSuccess"},
		},
	}
	r.ComputeSummary(2)

	data, err := r.JSON()
	if err != nil {
		t.Fatal(err)
	}

	// Verify 2-space indentation
	raw := string(data)
	if !strings.Contains(raw, "  \"spec\"") {
		t.Errorf("expected 2-space indentation, got:\n%s", raw)
	}

	// Covered should be a flat list of behavior names, not detailed objects
	if !strings.Contains(raw, `"covered": [`) {
		t.Errorf("expected covered as array, got:\n%s", raw)
	}
	if strings.Contains(raw, `"test_file"`) {
		t.Errorf("detailed covered entries should not appear in JSON output")
	}

	// Summary should appear before gaps in the output
	summaryIdx := strings.Index(raw, `"summary"`)
	gapsIdx := strings.Index(raw, `"gaps"`)
	if summaryIdx < 0 || gapsIdx < 0 || summaryIdx > gapsIdx {
		t.Errorf("expected summary before gaps in JSON output")
	}

	var parsed struct {
		Spec    string   `json:"spec"`
		Summary Summary  `json:"summary"`
		Gaps    []Gap    `json:"gaps"`
		Covered []string `json:"covered"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}

	if parsed.Spec != ".vex/vexspec.yaml" {
		t.Errorf("expected spec .vex/vexspec.yaml, got %s", parsed.Spec)
	}
	if len(parsed.Gaps) != 1 {
		t.Errorf("expected 1 gap, got %d", len(parsed.Gaps))
	}
	if len(parsed.Covered) != 1 || parsed.Covered[0] != "login" {
		t.Errorf("expected covered [login], got %v", parsed.Covered)
	}
}

func TestHasGaps(t *testing.T) {
	r := &Report{}
	if r.HasGaps() {
		t.Error("empty report should have no gaps")
	}

	r.Gaps = []Gap{{Behavior: "test", Detail: "missing"}}
	if !r.HasGaps() {
		t.Error("report with gaps should return true")
	}
}

func TestComputeSummary(t *testing.T) {
	r := &Report{
		Gaps: []Gap{
			{Behavior: "login", Detail: "missing expiry"},
		},
		Covered: []Covered{
			{Behavior: "login", Detail: "valid creds"},
			{Behavior: "refresh", Detail: "returns new token"},
		},
	}
	r.ComputeSummary(3)

	if r.Summary.TotalBehaviors != 3 {
		t.Errorf("expected 3 total, got %d", r.Summary.TotalBehaviors)
	}
	if r.Summary.FullyCovered != 1 {
		t.Errorf("expected 1 fully covered (refresh), got %d", r.Summary.FullyCovered)
	}
	if r.Summary.GapsFound != 1 {
		t.Errorf("expected 1 gap, got %d", r.Summary.GapsFound)
	}
	// CoveredBehaviors should be sorted and deduplicated
	if len(r.CoveredBehaviors) != 2 {
		t.Errorf("expected 2 covered behaviors, got %d", len(r.CoveredBehaviors))
	}
	if r.CoveredBehaviors[0] != "login" || r.CoveredBehaviors[1] != "refresh" {
		t.Errorf("expected [login refresh], got %v", r.CoveredBehaviors)
	}
}

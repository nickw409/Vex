package report

import (
	"encoding/json"
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

	var parsed Report
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}

	if parsed.Spec != ".vex/vexspec.yaml" {
		t.Errorf("expected spec .vex/vexspec.yaml, got %s", parsed.Spec)
	}
	if len(parsed.Gaps) != 1 {
		t.Errorf("expected 1 gap, got %d", len(parsed.Gaps))
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
}

package report

import (
	"encoding/json"
	"sort"
)

// Report is the final output of a vex check. It is designed to be short and
// actionable: summary first, then gaps (the items the agent must fix), then
// a flat list of covered behavior names for context.
type Report struct {
	Spec             string   `json:"spec"`
	Summary          Summary  `json:"summary"`
	Gaps             []Gap    `json:"gaps"`
	CoveredBehaviors []string `json:"covered"`

	// Covered holds detailed coverage entries used internally during the
	// two-pass check. It is excluded from JSON output to keep reports small.
	Covered []Covered `json:"-"`
}

type Gap struct {
	Behavior   string `json:"behavior"`
	Detail     string `json:"detail"`
	Suggestion string `json:"suggestion"`
}

type Covered struct {
	Behavior string `json:"behavior"`
	Detail   string `json:"detail"`
	TestFile string `json:"test_file"`
	TestName string `json:"test_name"`
}

type Summary struct {
	TotalBehaviors int `json:"total_behaviors"`
	FullyCovered   int `json:"fully_covered"`
	GapsFound      int `json:"gaps_found"`
}

func (r *Report) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

func (r *Report) HasGaps() bool {
	return len(r.Gaps) > 0
}

func (r *Report) ComputeSummary(totalBehaviors int) {
	gapped := make(map[string]bool)
	for _, g := range r.Gaps {
		gapped[g.Behavior] = true
	}

	covered := make(map[string]bool)
	for _, c := range r.Covered {
		covered[c.Behavior] = true
	}

	fullyCovered := 0
	for name := range covered {
		if !gapped[name] {
			fullyCovered++
		}
	}

	// Build the deduplicated, sorted list of covered behavior names.
	names := make([]string, 0, len(covered))
	for name := range covered {
		names = append(names, name)
	}
	sort.Strings(names)
	r.CoveredBehaviors = names

	r.Summary = Summary{
		TotalBehaviors: totalBehaviors,
		FullyCovered:   fullyCovered,
		GapsFound:      len(r.Gaps),
	}
}

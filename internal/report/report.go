package report

import "encoding/json"

type Report struct {
	Spec             string    `json:"spec"`
	BehaviorsChecked int       `json:"behaviors_checked"`
	Gaps             []Gap     `json:"gaps"`
	Covered          []Covered `json:"covered"`
	Summary          Summary   `json:"summary"`
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

	r.BehaviorsChecked = totalBehaviors
	r.Summary = Summary{
		TotalBehaviors: totalBehaviors,
		FullyCovered:   fullyCovered,
		GapsFound:      len(r.Gaps),
	}
}

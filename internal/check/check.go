package check

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/nwiley/vex/internal/provider"
	"github.com/nwiley/vex/internal/report"
	"github.com/nwiley/vex/internal/spec"
)

const maxContentSize = 400_000

type SectionInput struct {
	Section     *spec.Section
	Behaviors   []spec.Behavior
	SourceFiles map[string]string
	TestFiles   map[string]string
}

const pass1SystemPrompt = `You are a test coverage auditor. You will receive:
1. A component specification with named behaviors describing intended functionality
2. Test files testing the component

Your job: determine whether the tests adequately cover each behavior described in the specification based on the test code alone.

Respond with ONLY a JSON object in this exact format:
{
  "gaps": [
    {
      "behavior": "behavior-name",
      "detail": "What specific aspect is not tested",
      "suggestion": "What test to add"
    }
  ],
  "covered": [
    {
      "behavior": "behavior-name",
      "detail": "What aspect is covered",
      "test_file": "filename_test.go",
      "test_name": "TestFunctionName"
    }
  ]
}

Rules:
- A behavior can have MULTIPLE covered entries AND multiple gap entries (partial coverage).
- Only flag genuine gaps. If a test clearly exercises the behavior, mark it covered.
- Be specific about which test file and test function covers each behavior.
- For gaps, suggest concrete test names and what they should assert.
- Do NOT invent behaviors beyond what the spec describes. The spec is the scope boundary.
- If a behavior has sub-points, each sub-point can be a separate covered/gap entry, but the behavior name stays the same.`

const pass2SystemPrompt = `You are a test coverage auditor. You will receive:
1. A component specification with named behaviors describing intended functionality
2. Source code files implementing the component
3. Test files testing the component

These behaviors were flagged as potentially uncovered in a first pass that only looked at tests. Your job: confirm whether they are truly untested by examining both the source code and the tests.

Respond with ONLY a JSON object in this exact format:
{
  "gaps": [
    {
      "behavior": "behavior-name",
      "detail": "What specific aspect is not tested",
      "suggestion": "What test to add"
    }
  ],
  "covered": [
    {
      "behavior": "behavior-name",
      "detail": "What aspect is covered",
      "test_file": "filename_test.go",
      "test_name": "TestFunctionName"
    }
  ]
}

Rules:
- A behavior can have MULTIPLE covered entries AND multiple gap entries (partial coverage).
- Only flag genuine gaps. If the behavior is tested, mark it covered.
- Be specific about which test file and test function covers each behavior.
- For gaps, suggest concrete test names and what they should assert.
- Do NOT invent behaviors beyond what the spec describes. The spec is the scope boundary.
- If a behavior has sub-points, each sub-point can be a separate covered/gap entry, but the behavior name stays the same.
- If the source code contains significant observable behavior NOT described in the spec (e.g. concurrency, rate limiting, caching, retries, ordering guarantees), add a gap entry with behavior "UNSPECIFIED" and describe the missing spec coverage in the detail field. This helps keep the spec in sync with the code.`

type checkResponse struct {
	Gaps    []report.Gap     `json:"gaps"`
	Covered []report.Covered `json:"covered"`
}

type sectionResult struct {
	section string
	gaps    []report.Gap
	covered []report.Covered
	err     error
}

// RunProject checks all sections using a two-pass strategy with bounded concurrency.
// Pass 1: test files + behaviors only (cheap). Pass 2: source + tests for uncovered behaviors only.
func RunProject(ctx context.Context, p provider.Provider, ps *spec.ProjectSpec, inputs []SectionInput, maxConcurrency int) (*report.Report, error) {
	if maxConcurrency <= 0 {
		maxConcurrency = 4
	}

	// Pass 1: test files only
	pass1Results := make([]sectionResult, len(inputs))
	runParallel(ctx, p, inputs, pass1Results, maxConcurrency, true)

	// Determine which sections need pass 2
	var pass2Inputs []SectionInput
	var pass2Indices []int

	for i, sr := range pass1Results {
		if sr.err != nil {
			continue
		}
		uncovered := uncoveredBehaviors(sr.gaps, sr.covered, inputs[i].Behaviors)
		if len(uncovered) > 0 {
			pass2Inputs = append(pass2Inputs, SectionInput{
				Section:     inputs[i].Section,
				Behaviors:   uncovered,
				SourceFiles: inputs[i].SourceFiles,
				TestFiles:   inputs[i].TestFiles,
			})
			pass2Indices = append(pass2Indices, i)
		}
	}

	// Pass 2: source + test files for uncovered behaviors only
	pass2Results := make([]sectionResult, len(pass2Inputs))
	if len(pass2Inputs) > 0 {
		runParallel(ctx, p, pass2Inputs, pass2Results, maxConcurrency, false)
	}

	// Merge results
	merged := &report.Report{
		Spec:    ".vex/vexspec.yaml",
		Gaps:    []report.Gap{},
		Covered: []report.Covered{},
	}

	totalBehaviors := 0
	var errs []string

	for i, sr := range pass1Results {
		if sr.err != nil {
			errs = append(errs, sr.err.Error())
			continue
		}
		// Add covered from pass 1
		merged.Covered = append(merged.Covered, sr.covered...)

		// Check if this section had a pass 2
		pass2Idx := -1
		for j, idx := range pass2Indices {
			if idx == i {
				pass2Idx = j
				break
			}
		}

		if pass2Idx >= 0 {
			// Use pass 2 results for the uncovered behaviors
			p2 := pass2Results[pass2Idx]
			if p2.err != nil {
				errs = append(errs, p2.err.Error())
				// Fall back to pass 1 gaps for this section
				merged.Gaps = append(merged.Gaps, sr.gaps...)
			} else {
				merged.Gaps = append(merged.Gaps, p2.gaps...)
				merged.Covered = append(merged.Covered, p2.covered...)
			}
		} else {
			// All behaviors covered in pass 1, no gaps
		}
	}

	for _, input := range inputs {
		totalBehaviors += len(input.Behaviors)
	}

	merged.Gaps = filterFalseUnspecified(merged.Gaps, ps)
	merged.ComputeSummary(totalBehaviors)

	if len(errs) > 0 {
		return merged, fmt.Errorf("errors in %d section(s): %s", len(errs), strings.Join(errs, "; "))
	}

	return merged, nil
}

func runParallel(ctx context.Context, p provider.Provider, inputs []SectionInput, results []sectionResult, maxConcurrency int, testOnly bool) {
	sem := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup

	for i, input := range inputs {
		wg.Add(1)
		go func(idx int, si SectionInput) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			sr := sectionResult{section: si.Section.Name}

			var gaps []report.Gap
			var covered []report.Covered
			var err error

			if testOnly {
				gaps, covered, err = runSectionPass1(ctx, p, &si)
			} else {
				gaps, covered, err = runSectionPass2(ctx, p, &si)
			}

			if err != nil {
				sr.err = fmt.Errorf("section %q: %w", si.Section.Name, err)
			} else {
				sr.gaps = gaps
				sr.covered = covered
			}
			results[idx] = sr
		}(i, input)
	}

	wg.Wait()
}

// uncoveredBehaviors returns behaviors that have gaps but are NOT fully covered.
func uncoveredBehaviors(gaps []report.Gap, covered []report.Covered, allBehaviors []spec.Behavior) []spec.Behavior {
	coveredSet := make(map[string]bool)
	for _, c := range covered {
		coveredSet[c.Behavior] = true
	}

	gappedSet := make(map[string]bool)
	for _, g := range gaps {
		gappedSet[g.Behavior] = true
	}

	var uncovered []spec.Behavior
	for _, b := range allBehaviors {
		// Include if it has gaps or wasn't mentioned at all
		if gappedSet[b.Name] || !coveredSet[b.Name] {
			uncovered = append(uncovered, b)
		}
	}
	return uncovered
}

func runSectionPass1(ctx context.Context, p provider.Provider, input *SectionInput) ([]report.Gap, []report.Covered, error) {
	userPrompt, err := buildPass1Prompt(input)
	if err != nil {
		return nil, nil, err
	}

	req := provider.CompletionRequest{
		SystemPrompt: pass1SystemPrompt,
		UserPrompt:   userPrompt,
	}

	resp, err := p.Complete(ctx, req)
	if err != nil {
		return nil, nil, fmt.Errorf("pass 1 check failed: %w", err)
	}

	return parseSectionResponse(resp.Content)
}

func runSectionPass2(ctx context.Context, p provider.Provider, input *SectionInput) ([]report.Gap, []report.Covered, error) {
	userPrompt, err := buildPass2Prompt(input)
	if err != nil {
		return nil, nil, err
	}

	req := provider.CompletionRequest{
		SystemPrompt: pass2SystemPrompt,
		UserPrompt:   userPrompt,
	}

	resp, err := p.Complete(ctx, req)
	if err != nil {
		return nil, nil, fmt.Errorf("pass 2 check failed: %w", err)
	}

	return parseSectionResponse(resp.Content)
}

func buildPass1Prompt(input *SectionInput) (string, error) {
	var b strings.Builder

	fmt.Fprintf(&b, "## Section: %s\n\n", input.Section.Name)
	if input.Section.Description != "" {
		fmt.Fprintf(&b, "%s\n", input.Section.Description)
	}

	b.WriteString("### Behaviors\n\n")
	for _, beh := range input.Behaviors {
		fmt.Fprintf(&b, "#### %s\n%s\n\n", beh.Name, beh.Description)
	}

	b.WriteString("## Test Files\n\n")
	for name, content := range input.TestFiles {
		fmt.Fprintf(&b, "### %s\n```\n%s\n```\n\n", name, content)
	}

	result := b.String()
	if len(result) > maxContentSize {
		return "", fmt.Errorf("total file content exceeds %d chars; use --diff or a narrower target path", maxContentSize)
	}

	return result, nil
}

func buildPass2Prompt(input *SectionInput) (string, error) {
	var b strings.Builder

	fmt.Fprintf(&b, "## Section: %s\n\n", input.Section.Name)
	if input.Section.Description != "" {
		fmt.Fprintf(&b, "%s\n", input.Section.Description)
	}

	b.WriteString("### Behaviors (flagged as potentially uncovered)\n\n")
	for _, beh := range input.Behaviors {
		fmt.Fprintf(&b, "#### %s\n%s\n\n", beh.Name, beh.Description)
	}

	b.WriteString("## Source Files\n\n")
	for name, content := range input.SourceFiles {
		fmt.Fprintf(&b, "### %s\n```\n%s\n```\n\n", name, content)
	}

	b.WriteString("## Test Files\n\n")
	for name, content := range input.TestFiles {
		fmt.Fprintf(&b, "### %s\n```\n%s\n```\n\n", name, content)
	}

	result := b.String()
	if len(result) > maxContentSize {
		return "", fmt.Errorf("total file content exceeds %d chars; use --diff or a narrower target path", maxContentSize)
	}

	return result, nil
}

func parseSectionResponse(content string) ([]report.Gap, []report.Covered, error) {
	content = extractJSON(content)

	var resp checkResponse
	if err := json.Unmarshal([]byte(content), &resp); err != nil {
		return nil, nil, fmt.Errorf("parsing check response: %w\nraw response: %s", err, content)
	}

	gaps := resp.Gaps
	covered := resp.Covered
	if gaps == nil {
		gaps = []report.Gap{}
	}
	if covered == nil {
		covered = []report.Covered{}
	}

	return gaps, covered, nil
}

// filterFalseUnspecified removes UNSPECIFIED gaps that reference behaviors
// or components already covered by other sections in the spec. This happens
// because each section's LLM call only sees its own behaviors, so it flags
// code that belongs to another section as unspecified.
func filterFalseUnspecified(gaps []report.Gap, ps *spec.ProjectSpec) []report.Gap {
	// Build a set of all known names: behavior names, section names,
	// subsection names, and command names from across the full spec.
	known := make(map[string]bool)
	for _, sec := range ps.Sections {
		known[strings.ToLower(sec.Name)] = true
		for _, b := range sec.Behaviors {
			known[strings.ToLower(b.Name)] = true
		}
		for _, sub := range sec.Subsections {
			known[strings.ToLower(sub.Name)] = true
			for _, b := range sub.Behaviors {
				known[strings.ToLower(b.Name)] = true
			}
		}
	}
	for _, b := range ps.Shared {
		known[strings.ToLower(b.Name)] = true
	}

	var filtered []report.Gap
	for _, g := range gaps {
		if g.Behavior != "UNSPECIFIED" {
			filtered = append(filtered, g)
			continue
		}

		// Check if the detail references any known behavior/section name
		detailLower := strings.ToLower(g.Detail)
		coveredElsewhere := false
		for name := range known {
			if len(name) > 2 && strings.Contains(detailLower, name) {
				coveredElsewhere = true
				break
			}
		}

		if !coveredElsewhere {
			filtered = append(filtered, g)
		}
	}

	if filtered == nil {
		filtered = []report.Gap{}
	}
	return filtered
}

func extractJSON(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	s = strings.TrimSpace(s)
	if start := strings.Index(s, "{"); start >= 0 {
		if end := strings.LastIndex(s, "}"); end >= start {
			return s[start : end+1]
		}
	}
	return s
}

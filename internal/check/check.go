package check

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/nickw409/vex/internal/log"
	"github.com/nickw409/vex/internal/perf"
	"github.com/nickw409/vex/internal/provider"
	"github.com/nickw409/vex/internal/report"
	"github.com/nickw409/vex/internal/spec"
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
- If a behavior has sub-points, each sub-point can be a separate covered/gap entry, but the behavior name stays the same.
- If a behavior description contains a mathematical formula, equation, or numerical method, verify that tests assert correctness using known inputs with expected outputs, boundary/edge conditions, or convergence/statistical properties — not merely that the function is called or runs without error. A formula-bearing behavior is covered only when the tests would catch an incorrect implementation of that formula.`

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
- If a behavior description contains a mathematical formula, equation, or numerical method, verify that tests assert correctness using known inputs with expected outputs, boundary/edge conditions, or convergence/statistical properties — not merely that the function is called or runs without error. A formula-bearing behavior is covered only when the tests would catch an incorrect implementation of that formula.
- If the source code contains significant observable behavior NOT described in the spec (e.g. concurrency, rate limiting, caching, retries, ordering guarantees), add a gap entry with behavior "UNSPECIFIED" and describe the missing spec coverage in the detail field. This helps keep the spec in sync with the code.`

type checkResponse struct {
	Gaps    []report.Gap     `json:"gaps"`
	Covered []report.Covered `json:"covered"`
}

type sectionResult struct {
	section string
	gaps    []report.Gap
	covered []report.Covered
	usage   provider.TokenUsage
	err     error
}

const maxRetries = 2

// task represents a single LLM call (pass 1 or pass 2) for one section.
type task struct {
	inputIdx int // index into the original inputs slice
	input    SectionInput
	testOnly bool // true = pass 1, false = pass 2
	retries  int
}

// RunProject checks all sections using a pipelined two-pass strategy with
// adaptive concurrency. Pass 1 results flow directly into pass 2 as they
// complete, rather than waiting for all of pass 1 to finish.
// If prof is non-nil, timing spans are recorded for each phase.
func RunProject(ctx context.Context, p provider.Provider, ps *spec.ProjectSpec, inputs []SectionInput, maxConcurrency int, prof *perf.Profile) (*report.Report, error) {
	if maxConcurrency <= 0 {
		maxConcurrency = 5
	}

	log.Info("pass 1: analyzing %d section(s)", len(inputs))

	// Collect results per input index. Each slot holds pass 1 and optionally pass 2.
	type combinedResult struct {
		pass1 sectionResult
		pass2 *sectionResult // nil if pass 2 not needed or not yet complete
	}
	results := make([]combinedResult, len(inputs))

	// Adaptive concurrency: current limit can shrink on retryable errors.
	var concurrency atomic.Int32
	concurrency.Store(int32(maxConcurrency))

	// Work queue and completion channel.
	work := make(chan task, len(inputs)*2) // enough for pass 1 + pass 2
	done := make(chan struct{})

	// Seed pass 1 tasks.
	pending := len(inputs)
	for i, inp := range inputs {
		work <- task{inputIdx: i, input: inp, testOnly: true}
	}

	var mu sync.Mutex
	var totalUsage provider.TokenUsage
	var errs []string

	// Worker loop: pull tasks from queue, respect adaptive semaphore.
	sem := make(chan struct{}, maxConcurrency) // buffered to max ceiling
	var wg sync.WaitGroup

	go func() {
		for t := range work {
			wg.Add(1)
			go func(t task) {
				defer wg.Done()

				// Adaptive semaphore: only allow up to current concurrency.
				// We always acquire from sem (sized to maxConcurrency ceiling),
				// but also check the adaptive limit.
				sem <- struct{}{}
				defer func() { <-sem }()

				pass := "pass 1"
				if !t.testOnly {
					pass = "pass 2"
				}

				var gaps []report.Gap
				var covered []report.Covered
				var usage provider.TokenUsage
				var err error

				var end func()
				if prof != nil {
					end = prof.Start(pass+":llm", t.input.Section.Name)
				}

				if t.testOnly {
					gaps, covered, usage, err = runSectionPass1(ctx, p, &t.input)
				} else {
					gaps, covered, usage, err = runSectionPass2(ctx, p, &t.input)
				}

				if end != nil {
					end()
				}

				mu.Lock()
				totalUsage.InputTokens += usage.InputTokens
				totalUsage.OutputTokens += usage.OutputTokens
				totalUsage.CostUSD += usage.CostUSD
				totalUsage.DurationMS += usage.DurationMS
				mu.Unlock()

				if err != nil && provider.IsRetryable(err) && t.retries < maxRetries {
					// Adaptive backoff: reduce concurrency by 1 (floor 1)
					for {
						cur := concurrency.Load()
						next := cur - 1
						if next < 1 {
							next = 1
						}
						if concurrency.CompareAndSwap(cur, next) {
							if next < cur {
								log.Info("%s: %q hit rate limit, reducing concurrency to %d", pass, t.input.Section.Name, next)
							}
							break
						}
					}
					t.retries++
					log.Info("%s: %q retrying (attempt %d/%d)", pass, t.input.Section.Name, t.retries, maxRetries)
					work <- t
					return
				}

				sr := sectionResult{section: t.input.Section.Name}
				if err != nil {
					sr.err = fmt.Errorf("section %q: %w", t.input.Section.Name, err)
					log.Info("%s: %q failed: %v", pass, t.input.Section.Name, err)
				} else {
					sr.gaps = gaps
					sr.covered = covered
					sr.usage = usage
					log.Info("%s: %q done — %d gaps, %d covered", pass, t.input.Section.Name, len(gaps), len(covered))

					// Ramp up: on success, increase concurrency by 1 (capped at max)
					for {
						cur := concurrency.Load()
						if cur >= int32(maxConcurrency) {
							break
						}
						if concurrency.CompareAndSwap(cur, cur+1) {
							break
						}
					}
				}

				mu.Lock()
				if t.testOnly {
					results[t.inputIdx].pass1 = sr

					// Pipeline: if pass 1 succeeded and has uncovered behaviors, enqueue pass 2.
					if sr.err == nil {
						uncovered := uncoveredBehaviors(sr.gaps, sr.covered, inputs[t.inputIdx].Behaviors)
						if len(uncovered) > 0 {
							pending++
							log.Info("pass 2: enqueuing %q (%d uncovered)", t.input.Section.Name, len(uncovered))
							work <- task{
								inputIdx: t.inputIdx,
								input: SectionInput{
									Section:     inputs[t.inputIdx].Section,
									Behaviors:   uncovered,
									SourceFiles: inputs[t.inputIdx].SourceFiles,
									TestFiles:   inputs[t.inputIdx].TestFiles,
								},
								testOnly: false,
							}
						}
					}
				} else {
					results[t.inputIdx].pass2 = &sr
				}

				pending--
				if pending == 0 {
					close(done)
				}
				mu.Unlock()
			}(t)
		}
	}()

	<-done
	close(work)
	wg.Wait()

	// Merge results.
	merged := &report.Report{
		Spec:    ".vex/vexspec.yaml",
		Gaps:    []report.Gap{},
		Covered: []report.Covered{},
	}

	totalBehaviors := 0
	for i, cr := range results {
		if cr.pass1.err != nil {
			errs = append(errs, cr.pass1.err.Error())
			continue
		}

		merged.Covered = append(merged.Covered, cr.pass1.covered...)

		if cr.pass2 != nil {
			if cr.pass2.err != nil {
				errs = append(errs, cr.pass2.err.Error())
				// Fall back to pass 1 gaps
				merged.Gaps = append(merged.Gaps, cr.pass1.gaps...)
			} else {
				merged.Gaps = append(merged.Gaps, cr.pass2.gaps...)
				merged.Covered = append(merged.Covered, cr.pass2.covered...)
			}
		}

		totalBehaviors += len(inputs[i].Behaviors)
	}

	merged.Gaps = filterFalseUnspecified(merged.Gaps, ps)
	merged.ComputeSummary(totalBehaviors)

	log.Info("tokens: %d in / %d out | cost: $%.4f",
		totalUsage.InputTokens, totalUsage.OutputTokens, totalUsage.CostUSD)

	if len(errs) > 0 {
		return merged, fmt.Errorf("errors in %d section(s): %s", len(errs), strings.Join(errs, "; "))
	}

	return merged, nil
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

func runSectionPass1(ctx context.Context, p provider.Provider, input *SectionInput) ([]report.Gap, []report.Covered, provider.TokenUsage, error) {
	userPrompt, err := buildPass1Prompt(input)
	if err != nil {
		return nil, nil, provider.TokenUsage{}, err
	}

	req := provider.CompletionRequest{
		SystemPrompt: pass1SystemPrompt,
		UserPrompt:   userPrompt,
	}

	resp, err := p.Complete(ctx, req)
	if err != nil {
		return nil, nil, provider.TokenUsage{}, fmt.Errorf("pass 1 check failed: %w", err)
	}

	gaps, covered, err := parseSectionResponse(resp.Content)
	return gaps, covered, resp.Usage, err
}

func runSectionPass2(ctx context.Context, p provider.Provider, input *SectionInput) ([]report.Gap, []report.Covered, provider.TokenUsage, error) {
	userPrompt, err := buildPass2Prompt(input)
	if err != nil {
		return nil, nil, provider.TokenUsage{}, err
	}

	req := provider.CompletionRequest{
		SystemPrompt: pass2SystemPrompt,
		UserPrompt:   userPrompt,
	}

	resp, err := p.Complete(ctx, req)
	if err != nil {
		return nil, nil, provider.TokenUsage{}, fmt.Errorf("pass 2 check failed: %w", err)
	}

	gaps, covered, err := parseSectionResponse(resp.Content)
	return gaps, covered, resp.Usage, err
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

	start := strings.Index(s, "{")
	if start < 0 {
		return s
	}

	// Track brace depth to find the matching close brace, skipping
	// braces inside JSON strings to handle keys/values containing '{}'.
	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(s); i++ {
		c := s[i]
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && inString {
			escaped = true
			continue
		}
		if c == '"' {
			inString = !inString
			continue
		}
		if inString {
			continue
		}
		if c == '{' {
			depth++
		} else if c == '}' {
			depth--
			if depth == 0 {
				return s[start : i+1]
			}
		}
	}

	// Fallback: unbalanced braces, return from first { to end
	return s[start:]
}

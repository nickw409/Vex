package spec

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/nickw409/vex/internal/log"
	"github.com/nickw409/vex/internal/provider"
)

type ValidationResult struct {
	Complete    bool                   `json:"complete"`
	Suggestions []ValidationSuggestion `json:"suggestions"`
}

type ValidationSuggestion struct {
	Section      string `json:"section"`
	BehaviorName string `json:"behavior_name"`
	Description  string `json:"description"`
	Relation     string `json:"relation"`
}

const validateSystemPrompt = `You are a test specification reviewer. You will receive a project spec with sections describing components and their intended behaviors. Your job is to identify behaviors that are conspicuously absent — things a user of each component would obviously expect but that no listed behavior covers.

Each section's description defines the scope boundary. Only suggest behaviors that fall within a section's scope.

Respond with ONLY a JSON object in this exact format:
{
  "complete": true,
  "suggestions": []
}

When suggestions are needed:
{
  "complete": false,
  "suggestions": [
    {
      "section": "Section Name",
      "behavior_name": "suggested-kebab-name",
      "description": "What this behavior should cover",
      "relation": "new" or "extends <existing-behavior-name>"
    }
  ]
}

Rules:
- Suggest genuinely missing user-facing behaviors, not implementation details.
- Do NOT suggest: timeout handling, permission errors, graceful degradation, logging improvements, or internal error propagation paths. These are implementation concerns, not behavioral gaps.
- A behavior is "conspicuously absent" if a reasonable user of the component would ask "wait, what happens when I do X?" and the spec has no answer.
- If a behavior covers error cases inline (e.g. "returns error when X"), that counts. Don't re-suggest it as a separate behavior.
- Include all genuinely missing behaviors — do not artificially limit the count, but avoid low-confidence suggestions.
- Use "relation": "new" for entirely missing behaviors, or "relation": "extends <name>" when an existing behavior is missing a significant aspect.

Additionally, flag any existing behavior that is NOT actually a behavior:
- Data structure or type definitions (e.g. "Report contains these fields") are NOT behaviors
- Interface contracts (e.g. "Provider must implement Complete()") are NOT behaviors
- Lists of supported values (e.g. "Supports Go, Python, Java") are NOT behaviors
- A real behavior has observable input → output or describes something a caller does and gets a result
- Mathematical formulas and equations ARE valid behaviors — they define a correctness contract that tests must verify. Do NOT flag these as non-behavioral.
When you find non-behavioral entries, include them in suggestions with relation: "remove: not a behavior — <reason>".`

const validateMaxRetries = 2

type validateTask struct {
	idx     int
	sec     Section
	retries int
}

func ValidateProject(ctx context.Context, p provider.Provider, ps *ProjectSpec, maxConcurrency ...int) (*ValidationResult, error) {
	maxConc := 5
	if len(maxConcurrency) > 0 && maxConcurrency[0] > 0 {
		maxConc = maxConcurrency[0]
	}

	if len(ps.Sections) == 1 {
		maxConc = 1
	}

	type sectionResult struct {
		suggestions []ValidationSuggestion
		err         error
	}

	results := make([]sectionResult, len(ps.Sections))

	var concurrency atomic.Int32
	concurrency.Store(int32(maxConc))

	work := make(chan validateTask, len(ps.Sections))
	done := make(chan struct{})
	sem := make(chan struct{}, maxConc)

	// Seed work queue.
	var pending atomic.Int32
	pending.Store(int32(len(ps.Sections)))
	for i, sec := range ps.Sections {
		work <- validateTask{idx: i, sec: sec}
	}

	var mu sync.Mutex
	var wg sync.WaitGroup

	go func() {
		for t := range work {
			wg.Add(1)
			go func(t validateTask) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()

				shared := ps.ResolveShared(&t.sec)
				prompt := buildSectionValidatePrompt(ps.Project, ps.Description, &t.sec, shared)

				log.Info("validating section %q", t.sec.Name)
				req := provider.CompletionRequest{
					SystemPrompt: validateSystemPrompt,
					UserPrompt:   prompt,
				}

				resp, err := p.Complete(ctx, req)
				if err != nil && provider.IsRetryable(err) && t.retries < validateMaxRetries {
					// Adaptive backoff: reduce concurrency by 1 (floor 1).
					for {
						cur := concurrency.Load()
						next := cur - 1
						if next < 1 {
							next = 1
						}
						if concurrency.CompareAndSwap(cur, next) {
							if next < cur {
								log.Info("validate: %q hit rate limit, reducing concurrency to %d", t.sec.Name, next)
							}
							break
						}
					}
					t.retries++
					log.Info("validate: %q retrying (attempt %d/%d)", t.sec.Name, t.retries, validateMaxRetries)
					work <- t
					return
				}

				if err != nil {
					mu.Lock()
					results[t.idx] = sectionResult{err: fmt.Errorf("section %q: %w", t.sec.Name, err)}
					mu.Unlock()
				} else {
					parsed, parseErr := parseValidationResponse(resp.Content)
					mu.Lock()
					if parseErr != nil {
						results[t.idx] = sectionResult{err: fmt.Errorf("section %q: %w", t.sec.Name, parseErr)}
					} else {
						results[t.idx] = sectionResult{suggestions: parsed.Suggestions}
						log.Info("section %q done — %d suggestions", t.sec.Name, len(parsed.Suggestions))
					}
					mu.Unlock()

					// Ramp up: on success, increase concurrency by 1 (capped at max).
					for {
						cur := concurrency.Load()
						if cur >= int32(maxConc) {
							break
						}
						if concurrency.CompareAndSwap(cur, cur+1) {
							break
						}
					}
				}

				if pending.Add(-1) == 0 {
					close(done)
				}
			}(t)
		}
	}()

	<-done
	close(work)
	wg.Wait()

	merged := &ValidationResult{
		Complete:    true,
		Suggestions: []ValidationSuggestion{},
	}

	var errs []string
	for _, r := range results {
		if r.err != nil {
			errs = append(errs, r.err.Error())
			continue
		}
		if len(r.suggestions) > 0 {
			merged.Complete = false
			merged.Suggestions = append(merged.Suggestions, r.suggestions...)
		}
	}

	if len(errs) > 0 {
		return merged, fmt.Errorf("errors in %d section(s): %s", len(errs), strings.Join(errs, "; "))
	}

	return merged, nil
}

func buildSectionValidatePrompt(project, description string, sec *Section, shared []Behavior) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Project: %s\n\n", project)
	if description != "" {
		fmt.Fprintf(&b, "## Description\n%s\n\n", description)
	}

	fmt.Fprintf(&b, "## Section: %s\n", sec.Name)
	if sec.Description != "" {
		fmt.Fprintf(&b, "%s\n", sec.Description)
	}

	if len(shared) > 0 {
		b.WriteString("\n### Shared Behaviors\n\n")
		for _, beh := range shared {
			fmt.Fprintf(&b, "#### %s\n%s\n", beh.Name, beh.Description)
		}
	}

	if len(sec.Behaviors) > 0 {
		b.WriteString("\n### Behaviors\n\n")
		for _, beh := range sec.Behaviors {
			fmt.Fprintf(&b, "#### %s\n%s\n", beh.Name, beh.Description)
		}
	}

	for _, sub := range sec.Subsections {
		fmt.Fprintf(&b, "\n### Subsection: %s\n\n", sub.Name)
		for _, beh := range sub.Behaviors {
			fmt.Fprintf(&b, "#### %s\n%s\n", beh.Name, beh.Description)
		}
	}

	return b.String()
}

func parseValidationResponse(content string) (*ValidationResult, error) {
	content = extractJSON(content)

	var result ValidationResult
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return nil, fmt.Errorf("parsing validation response: %w\nraw response: %s", err, content)
	}

	return &result, nil
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

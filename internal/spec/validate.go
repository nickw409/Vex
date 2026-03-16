package spec

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/nwiley/vex/internal/provider"
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
- Prefer fewer, high-confidence suggestions over exhaustive lists.
- Use "relation": "new" for entirely missing behaviors, or "relation": "extends <name>" when an existing behavior is missing a significant aspect.

Additionally, flag any existing behavior that is NOT actually a behavior:
- Data structure or type definitions (e.g. "Report contains these fields") are NOT behaviors
- Interface contracts (e.g. "Provider must implement Complete()") are NOT behaviors
- Lists of supported values (e.g. "Supports Go, Python, Java") are NOT behaviors
- A real behavior has observable input → output or describes something a caller does and gets a result
When you find non-behavioral entries, include them in suggestions with relation: "remove: not a behavior — <reason>".`

func ValidateProject(ctx context.Context, p provider.Provider, ps *ProjectSpec) (*ValidationResult, error) {
	req := provider.CompletionRequest{
		SystemPrompt: validateSystemPrompt,
		UserPrompt:   buildProjectValidatePrompt(ps),
	}

	resp, err := p.Complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("validation request failed: %w", err)
	}

	return parseValidationResponse(resp.Content)
}

func buildProjectValidatePrompt(ps *ProjectSpec) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Project: %s\n\n", ps.Project)
	if ps.Description != "" {
		fmt.Fprintf(&b, "## Description\n%s\n", ps.Description)
	}

	if len(ps.Shared) > 0 {
		b.WriteString("## Shared Behaviors\n\n")
		for _, beh := range ps.Shared {
			fmt.Fprintf(&b, "### %s\n%s\n", beh.Name, beh.Description)
		}
	}

	for _, sec := range ps.Sections {
		fmt.Fprintf(&b, "## Section: %s\n", sec.Name)
		if sec.Description != "" {
			fmt.Fprintf(&b, "%s\n", sec.Description)
		}
		if len(sec.Shared) > 0 {
			fmt.Fprintf(&b, "Uses shared: %s\n\n", strings.Join(sec.Shared, ", "))
		}

		if len(sec.Behaviors) > 0 {
			b.WriteString("### Behaviors\n\n")
			for _, beh := range sec.Behaviors {
				fmt.Fprintf(&b, "#### %s\n%s\n", beh.Name, beh.Description)
			}
		}

		for _, sub := range sec.Subsections {
			fmt.Fprintf(&b, "### Subsection: %s\n\n", sub.Name)
			for _, beh := range sub.Behaviors {
				fmt.Fprintf(&b, "#### %s\n%s\n", beh.Name, beh.Description)
			}
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

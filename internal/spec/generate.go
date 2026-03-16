package spec

import (
	"context"
	"fmt"
	"strings"

	"github.com/nwiley/vex/internal/provider"
	"gopkg.in/yaml.v3"
)

const generateSystemPrompt = `You are a spec writer for a test coverage auditor called Vex. You will receive a task description and must generate vexspec sections describing the intended behaviors.

All paths are absolute from the project root. Never use relative paths.

Respond with ONLY YAML in this format:

- name: <Section Name>
  path: <directory/path/from/root>
  description: |
    <1-2 sentence overview>
  behaviors:
    - name: <short-kebab-name>
      description: |
        <Natural language description of observable behavior>
        <Include expected inputs, outputs, status codes>
        <Include error cases inline>
  subsections:
    - name: <Subsection Name>
      path: <directory/path/from/root>
      behaviors:
        - name: <short-kebab-name>
          description: |
            <Behavior scoped to this subdirectory or file>

Use subsections when:
- A behavior is clearly scoped to a specific subdirectory or file within the section
- The section spans a broad directory but has logically distinct sub-components

For subsections, use "path" for directories or "file" for a single file (not both).

Rules:
- Each behavior describes ONE observable external behavior
- Include error cases and edge cases inline within behaviors (e.g. "Returns 401 on invalid credentials")
- Be specific: "returns 401" not "handles errors"
- Include side effects: database writes, events emitted, files created
- Do NOT describe implementation details (which function, which pattern, which algorithm)
- Do NOT suggest timeout handling, permission errors, or internal error propagation
- Behaviors come from the task description, NOT from any existing code
- Use short-kebab-case for behavior names
- If the task spans multiple directories, use a list for path
- Only use subsections when there is a clear structural reason to scope behaviors

A behavior is NOT:
- A data structure or type definition (e.g. "Report contains these fields")
- An interface contract (e.g. "Provider must implement Complete()")
- A list of supported values (e.g. "Supports Go, Python, Java")
- A description of how something is implemented internally

A behavior IS:
- Something a user or caller does and gets a result (e.g. "Returns error when file not found")
- Something with observable input → output (e.g. "Marshals report to indented JSON")

Descriptions must be stable — describe WHAT happens, not HOW. Do not reference specific implementation mechanisms (e.g. "strips markdown fences") that may change. Describe the observable contract (e.g. "extracts JSON from LLM response even when wrapped in preamble text").`

const generateExtendSystemPrompt = `You are a spec writer for a test coverage auditor called Vex. You will receive:
1. An existing section from a vexspec
2. A task description for new functionality being added to that section

You must generate ONLY the new behaviors and/or subsections to add. Do NOT repeat existing behaviors.

All paths are absolute from the project root. Never use relative paths.

Respond with ONLY YAML in this format:

behaviors:
  - name: <new-behavior-name>
    description: |
      <New behavior description>
subsections:
  - name: <New Subsection>
    path: <directory/path/from/root>
    behaviors:
      - name: <behavior-name>
        description: |
          <Behavior description>

Omit "behaviors" or "subsections" if you have none to add for that category.

Rules:
- Each behavior describes ONE observable external behavior
- Include error cases inline (e.g. "Returns 401 on invalid credentials")
- Be specific: "returns 401" not "handles errors"
- Do NOT repeat any behavior that already exists in the section
- Do NOT describe implementation details
- Do NOT suggest timeout handling, permission errors, or internal error propagation
- Behaviors come from the task description, NOT from any existing code
- Use short-kebab-case for behavior names

A behavior is NOT:
- A data structure or type definition (e.g. "Report contains these fields")
- An interface contract (e.g. "Provider must implement Complete()")
- A list of supported values (e.g. "Supports Go, Python, Java")

Descriptions must be stable — describe WHAT happens, not HOW.`

type ExtendResult struct {
	Behaviors   []Behavior   `yaml:"behaviors,omitempty"`
	Subsections []Subsection `yaml:"subsections,omitempty"`
}

func Generate(ctx context.Context, p provider.Provider, description string) ([]Section, error) {
	req := provider.CompletionRequest{
		SystemPrompt: generateSystemPrompt,
		UserPrompt:   description,
	}

	resp, err := p.Complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("spec generation failed: %w", err)
	}

	return parseGenerateResponse(resp.Content)
}

func GenerateExtend(ctx context.Context, p provider.Provider, section *Section, description string) (*ExtendResult, error) {
	req := provider.CompletionRequest{
		SystemPrompt: generateExtendSystemPrompt,
		UserPrompt:   buildExtendPrompt(section, description),
	}

	resp, err := p.Complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("spec extension failed: %w", err)
	}

	return parseExtendResponse(resp.Content)
}

func buildExtendPrompt(section *Section, description string) string {
	var b strings.Builder

	b.WriteString("## Existing Section\n\n")
	fmt.Fprintf(&b, "Name: %s\n", section.Name)
	if section.Description != "" {
		fmt.Fprintf(&b, "Description: %s\n", section.Description)
	}
	if len(section.Path) > 0 {
		fmt.Fprintf(&b, "Path: %s\n", strings.Join(section.Path, ", "))
	}

	if len(section.Behaviors) > 0 {
		b.WriteString("\nExisting behaviors:\n")
		for _, beh := range section.Behaviors {
			fmt.Fprintf(&b, "- %s: %s\n", beh.Name, beh.Description)
		}
	}

	if len(section.Subsections) > 0 {
		b.WriteString("\nExisting subsections:\n")
		for _, sub := range section.Subsections {
			fmt.Fprintf(&b, "- %s\n", sub.Name)
			for _, beh := range sub.Behaviors {
				fmt.Fprintf(&b, "  - %s: %s\n", beh.Name, beh.Description)
			}
		}
	}

	fmt.Fprintf(&b, "\n## New Functionality\n\n%s\n", description)

	return b.String()
}

func parseGenerateResponse(content string) ([]Section, error) {
	content = trimFences(content)

	var sections []Section
	if err := yaml.Unmarshal([]byte(content), &sections); err != nil {
		return nil, fmt.Errorf("parsing generated spec: %w\nraw response: %s", err, content)
	}

	for _, sec := range sections {
		if sec.Name == "" {
			return nil, fmt.Errorf("generated section missing name")
		}
		for _, b := range sec.Behaviors {
			if b.Name == "" || b.Description == "" {
				return nil, fmt.Errorf("generated behavior in %q missing name or description", sec.Name)
			}
		}
		for _, sub := range sec.Subsections {
			for _, b := range sub.Behaviors {
				if b.Name == "" || b.Description == "" {
					return nil, fmt.Errorf("generated behavior in subsection %q missing name or description", sub.Name)
				}
			}
		}
	}

	return sections, nil
}

func parseExtendResponse(content string) (*ExtendResult, error) {
	content = trimFences(content)

	var result ExtendResult
	if err := yaml.Unmarshal([]byte(content), &result); err != nil {
		return nil, fmt.Errorf("parsing extend response: %w\nraw response: %s", err, content)
	}

	for _, b := range result.Behaviors {
		if b.Name == "" || b.Description == "" {
			return nil, fmt.Errorf("generated behavior missing name or description")
		}
	}
	for _, sub := range result.Subsections {
		for _, b := range sub.Behaviors {
			if b.Name == "" || b.Description == "" {
				return nil, fmt.Errorf("generated behavior in subsection %q missing name or description", sub.Name)
			}
		}
	}

	return &result, nil
}

func trimFences(content string) string {
	content = strings.TrimSpace(content)
	content = strings.TrimPrefix(content, "```yaml")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	return strings.TrimSpace(content)
}

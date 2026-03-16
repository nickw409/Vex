package cli

import (
	"fmt"

	"github.com/spf13/cobra"
)

const guideText = `# Writing Vex Specifications

## When to Write a Spec
Write the vexspec BEFORE or DURING implementation, from the task description.
NEVER write a spec by reading existing code — that defeats the purpose.

## Quick Start
Generate a spec from a task description:

  vex spec "Add JWT authentication with login, refresh, and token validation"

This creates .vex/vexspec.yaml with sections and behaviors. If the file
already exists, new sections are appended.

## Spec Format
The vexspec lives at .vex/vexspec.yaml. One file per project, structured
as a living design doc. All paths are absolute from the project root.

  project: MyApp
  description: |
    One-line project description.

  shared:
    - name: error-handling
      description: |
        Behaviors that apply across multiple sections.

  sections:
    - name: Auth
      path: internal/auth
      description: |
        JWT authentication module.
      shared: [error-handling]
      behaviors:
        - name: login
          description: |
            POST /login accepts credentials.
            Returns JWT on success. Returns 401 on invalid credentials.
      subsections:
        - name: Token Refresh
          file: internal/auth/refresh.go
          behaviors:
            - name: refresh
              description: |
                POST /refresh returns a new token.

## Guidelines
- Each behavior should describe ONE observable external behavior
- Include error cases inline (e.g. "Returns 401 on invalid credentials")
- Be specific: "returns 401" not "handles errors"
- Include side effects: database writes, events emitted, files created
- Do NOT describe implementation details (which function, which pattern)
- All paths are absolute from the project root, never relative

## Output
Vex writes results to the .vex/ directory:
- .vex/vexspec.yaml    — the project spec (source of truth)
- .vex/report.json     — full check report (gaps + covered behaviors)
- .vex/validation.json — spec validation results

Always read the full report from these files. Stdout output may be
truncated by your environment. The .vex/ directory is gitignored.

## Example Workflow
1. Read task/ticket description
2. Run: vex spec "description" to generate sections
3. Review and edit .vex/vexspec.yaml
4. Run: vex validate — review .vex/validation.json, update spec if needed
5. Implement code and tests
6. Run: vex check — review .vex/report.json
7. Fix gaps reported by vex
8. Repeat steps 6-7 until exit code 0

To check only one section:
  vex check --section "Section Name"
`

func newGuideCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "guide",
		Short: "Print instructions for writing vexspecs",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Fprint(cmd.OutOrStdout(), guideText)
		},
	}
}

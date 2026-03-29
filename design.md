# Vex Design Document

## Problem

AI agents write code and tests, tests pass, but behaviors are missing or not wired up. The tests verify what was written, not what was intended. There's no tool that bridges the gap between "what should this do" and "do the tests prove it works."

## Solution

Vex is a spec-driven test coverage auditor. It takes a structured spec describing intended behavior and compares it against actual test files to find gaps.

## Core Workflow

```
1. Agent gets task description
2. Agent writes/updates .vex/vexspec.yaml from task description (BEFORE or DURING coding, NOT after)
3. Agent runs: vex validate → reviews .vex/validation.json, updates spec if needed
4. Agent writes code + tests
5. Agent runs: vex check → reviews .vex/report.json
6. Agent fixes gaps → repeat step 5
7. No gaps (exit 0) → done
```

The spec MUST come from intent/task description, not from reading the implementation. Otherwise the spec just confirms what was written, defeating the purpose.

## Spec Format (.vex/vexspec.yaml)

The vexspec lives at `.vex/vexspec.yaml`. One file per project, structured as a living design doc. All paths are absolute from the project root.

### Unified Project Spec

```yaml
project: MyApp
description: |
  One-line description of the project.

shared:
  - name: error-handling
    description: |
      Behaviors that apply across multiple sections.
      Referenced by name from individual sections.

sections:
  - name: Auth
    path: internal/auth
    description: |
      JWT authentication with bcrypt password hashing.
    shared: [error-handling]
    behaviors:
      - name: login
        description: |
          POST /login accepts username and password.
          Returns a JWT token with 1 hour expiry on success.
          Returns 401 with error message on invalid credentials.

      - name: token-validation
        description: |
          Protected endpoints check Authorization header for valid JWT.
          Expired tokens are rejected with 401.
          Malformed tokens are rejected with 401.

    subsections:
      - name: Token Refresh
        file: internal/auth/refresh.go
        behaviors:
          - name: refresh
            description: |
              POST /refresh accepts a valid, non-expired token.
              Returns a new token with fresh expiry.
              Rejects expired tokens.
```

### Structure

- **`project`** — Project name.
- **`shared`** — Behaviors that apply across sections. Referenced by name via `shared: [name]` on a section.
- **`sections`** — Top-level components/domains of the project. Each has:
  - **`name`** — Component name (not "feature" — these are architectural units).
  - **`path`** — Directory path from project root. Can be a list for cross-directory components: `path: [src/handlers/auth.rs, src/auth/]`.
  - **`description`** — What this component does.
  - **`shared`** — List of shared behavior names to include when checking this section.
  - **`behaviors`** — Section-level behaviors checked against all files in the path.
  - **`subsections`** — Scoped to specific files or subdirectories within the section:
    - **`path`** — Subdirectory (absolute from project root, not relative to parent section).
    - **`file`** — Single file (absolute from project root). Use `path` or `file`, not both.
    - **`behaviors`** — Behaviors checked only against files in this subsection's scope.

### Path Rules

All paths are absolute from the project root. Never relative to a parent section. This prevents ambiguity when agents generate or modify specs.

```yaml
# CORRECT — absolute paths
- name: Simulation Engine
  path: rsimulation-core
  subsections:
    - name: Optimizer
      path: rsimulation-core/src/opt_engine/

# WRONG — relative paths
- name: Simulation Engine
  path: rsimulation-core
  subsections:
    - name: Optimizer
      path: src/opt_engine/    # relative to parent — ambiguous
```

### Design Principles

- The spec doubles as a design doc. Scanning sections shows project architecture; drilling into behaviors shows what each piece does.
- Behaviors describe observable external behavior, not implementation details.
- One LLM call per section keeps prompts focused and cost predictable.
- Regressions surface naturally — run `vex check` and any behavior that lost coverage appears as a gap.

Behaviors are natural language descriptions of what the component should do. The LLM understands the intent and checks whether tests exercise it.

## Config (vex.yaml)

```yaml
provider: claude-cli       # claude-cli (default, no API key needed)
model: opus                # provider-specific model name
max_concurrency: 4         # max parallel LLM calls (default 4)

# Language detection overrides (optional, auto-detected by default)
languages:
  go:
    test_patterns: ["*_test.go"]
    source_patterns: ["*.go"]
  typescript:
    test_patterns: ["*.test.ts", "*.spec.ts"]
    source_patterns: ["*.ts"]
```

## Two LLM Operations

### 1. Spec Validation (`vex validate`)

Input: the vexspec
Question: "Is this spec complete? Does it describe all the behaviors needed for this feature? What's missing?"
Example: spec says "add auth with JWT tokens" but never mentions token expiry, refresh, or revocation. Vex flags those gaps in the spec itself.

### 2. Gap Detection (`vex check`)

Input: the vexspec + source files + test files
Question: "Given this spec describing the full intended behavior, do the tests cover all of it?"
Output: JSON listing which behaviors are covered and which have gaps.

One LLM call per section, run in parallel. Cost scales with section count, not project size. Max concurrency is configurable (default 4) to avoid rate limiting or excessive cost spikes.

## Output Format (JSON to stdout)

```json
{
  "target": "./internal/auth/",
  "spec": "auth.vexspec.yaml",
  "behaviors_checked": 6,
  "gaps": [
    {
      "behavior": "login",
      "detail": "No test validates token expiry is set to 1 hour",
      "suggestion": "Add assertion checking token exp claim"
    }
  ],
  "covered": [
    {
      "behavior": "login",
      "detail": "Valid credentials return token",
      "test_file": "auth_test.go",
      "test_name": "TestLoginSuccess"
    }
  ],
  "summary": {
    "total_behaviors": 3,
    "fully_covered": 1,
    "gaps_found": 4
  }
}
```

## Provider Abstraction

Multi-provider support since models change fast. Interface:

```go
type Provider interface {
    Complete(ctx context.Context, req CompletionRequest) (CompletionResponse, error)
}

type CompletionRequest struct {
    SystemPrompt string
    UserPrompt   string
    MaxTokens    int
}

type CompletionResponse struct {
    Content string
    Usage   TokenUsage
}
```

Default implementation: claude-cli (shells out to `claude` CLI, no API key needed).
Future providers: anthropic (API), openai (API), ollama (local).

## Output

Results are written to the `.vex/` directory:
- `.vex/vexspec.yaml` — the project spec (source of truth)
- `.vex/report.json` — full check report (gaps + covered behaviors)
- `.vex/validation.json` — spec validation results

JSON is also printed to stdout. Errors and usage logging go to stderr.

## Usage Logging

After each LLM call, vex logs token usage and cost to stderr:
```
[vex] tokens: 28878 in / 4645 out | cost: $0.2436 | time: 66.7s
```

## Drift Detection

Drift detection is on by default (`vex check`). It skips sections where neither code files nor spec content have changed since the last check:

- Uses `git log --since` for committed changes and `git diff HEAD` for uncommitted changes
- Per-section spec checksums (SHA-256) detect spec-only edits without file modifications
- Skipped sections carry forward their gaps and covered entries from the previous report
- `--drift=false` forces a full re-check of all sections
- Old reports without checksums trigger a full re-check automatically

## Exit Codes

- 0: no gaps found
- 1: gaps found
- 2: error (bad config, LLM failure, etc.)

## Guide Command

`vex guide` prints instructions for AI agents on how to write good vexspecs. This is meant to be injected into agent context. Key guidance:
- Write the spec from the task description, NOT from the implementation
- Each behavior should describe observable external behavior
- Include error cases and edge cases
- Be specific about expected responses, status codes, side effects

## Phases

1. **scaffold** — Go module, cobra CLI skeleton, config parsing, vex.yaml format ✓
2. **provider** — LLM provider abstraction + claude-cli implementation ✓
3. **spec** — vexspec.yaml parsing + `vex validate` command ✓
4. **check** — core gap detection engine, file discovery, JSON output, exit codes ✓
5. **drift** — drift detection (default on), `vex guide` command ✓
6. **unified-spec** — unified project spec format (.vex/vexspec.yaml with sections, subsections, shared behaviors) ✓
7. **spec-gen** — `vex spec "description"` command to generate vexspec from task description ✓
8. **report** — `vex report` human-readable summary, covered overrides, carry-forward drift ✓

## Future Integration with Arc

- Arc generates vexspec from plan.md phase descriptions
- `vex check` runs as a gate assertion in Arc's phase loop
- Gap detection failure → gate fails → agent retries with gap info

# Vex Design Document

## Problem

AI agents write code and tests, tests pass, but behaviors are missing or not wired up. The tests verify what was written, not what was intended. There's no tool that bridges the gap between "what should this do" and "do the tests prove it works."

## Solution

Vex is a spec-driven test coverage auditor. It takes a structured spec describing intended behavior and compares it against actual test files to find gaps.

## Core Workflow

```
1. Agent gets task description
2. Agent writes vexspec from task description (BEFORE or DURING coding, NOT after)
3. Agent writes code + tests
4. Agent runs: vex check ./path/ --spec feature.vexspec.yaml
5. Vex reports gaps as JSON → agent fixes → repeat step 4
6. No gaps (exit 0) → done
```

The spec MUST come from intent/task description, not from reading the implementation. Otherwise the spec just confirms what was written, defeating the purpose.

## Spec Format (vexspec.yaml)

```yaml
feature: JWT Authentication
description: |
  Users authenticate with username/password and receive a JWT token.
  Tokens are used for subsequent API requests via Authorization header.

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
      Missing header returns 401.

  - name: refresh
    description: |
      POST /refresh accepts a valid, non-expired token.
      Returns a new token with fresh expiry.
      Rejects expired tokens.
```

Behaviors are natural language descriptions of what the feature should do. The LLM understands the intent and checks whether tests exercise it.

## Config (vex.yaml)

```yaml
provider: claude            # claude-cli | anthropic | openai | ollama
model: sonnet               # provider-specific model name
api_key_env: ANTHROPIC_API_KEY  # env var name, not the key itself

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

1-2 LLM calls per run. Cheap.

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

Implementations: claude-cli (shells out, no API key needed), anthropic (API), openai (API), ollama (local).
Start with claude-cli as default since it needs zero config.

## Diff Mode

`vex check --diff --spec feature.vexspec.yaml`

- Runs `git diff HEAD` to get changed files
- Filters to only source + test files matching language patterns
- Scopes the check to only those files
- Still checks ALL spec behaviors (the spec defines scope, diff just reduces file noise)

Note: `git diff HEAD` includes unstaged files. Code is committed only after it passes vex.

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

1. **scaffold** — Go module, cobra CLI skeleton, config parsing, vex.yaml format
2. **provider** — LLM provider abstraction + claude-cli implementation (first provider, no API key needed)
3. **spec** — vexspec.yaml parsing + `vex validate` command (first LLM-powered feature)
4. **check** — core gap detection engine, file discovery, JSON output, exit codes
5. **diff** — `--diff` mode, `vex guide` command, additional providers (anthropic, openai, ollama)

## Future Integration with Arc

- Arc generates vexspec from plan.md phase descriptions
- `vex check` runs as a gate assertion in Arc's phase loop
- Gap detection failure → gate fails → agent retries with gap info

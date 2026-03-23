# Vex

Spec-driven test coverage auditor. Verifies that tests fully cover intended behavior described in a spec, not just what happens to be implemented.

## The Problem

AI agents write code and tests. Tests pass. But behaviors are missing or not wired up. The tests verify what was written, not what was intended.

## How It Works

1. Write a spec describing intended behaviors (or generate one from a task description)
2. Run `vex check` — it sends spec + code + tests to an LLM
3. Get a JSON report of gaps (untested behaviors) and coverage
4. Fix the gaps, repeat until clean

Vex uses a two-pass strategy: pass 1 sends only test files (cheap triage), pass 2 sends source code only for uncovered behaviors. Well-tested codebases skip pass 2 entirely.

## Install

**One-liner** (downloads a prebuilt binary):

```bash
curl -fsSL https://raw.githubusercontent.com/nickw409/vex/main/install.sh | sh
```

Install a specific version or to a custom directory:

```bash
VEX_VERSION=v0.1.0 curl -fsSL https://raw.githubusercontent.com/nickw409/vex/main/install.sh | sh
VEX_INSTALL_DIR=~/.local/bin curl -fsSL https://raw.githubusercontent.com/nickw409/vex/main/install.sh | sh
```

**From source** (requires Go 1.24+):

```bash
go install github.com/nickw409/vex/cmd/vex@latest
```

Requires the [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) to be installed and authenticated.

## Quick Start

```bash
# Initialize config
vex init

# Generate a spec from a task description
vex spec "Add JWT authentication with login, refresh, and token validation"

# Validate the spec for completeness
vex validate

# Check test coverage against the spec
vex check

# After fixing gaps, use drift for incremental checks
vex check --drift
```

## Spec Format

The spec lives at `.vex/vexspec.yaml`. One file per project, structured as a living design doc:

```yaml
project: MyApp
description: |
  Short project description.

shared:
  - name: error-handling
    description: |
      Cross-cutting behaviors referenced by multiple sections.

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
              POST /refresh returns new access token from valid refresh token.
```

All paths are absolute from the project root. Each behavior describes one observable external behavior with error cases inline.

## Commands

| Command | Description |
|---------|-------------|
| `vex check` | Check test coverage against spec |
| `vex check --section Name` | Check a single section |
| `vex check --drift` | Only check sections changed since last check |
| `vex validate` | Validate spec for missing behaviors |
| `vex spec "description"` | Generate spec sections from task description |
| `vex spec "desc" --extend Name` | Add behaviors to existing section |
| `vex drift` | Show which sections have changed since last check |
| `vex lang list` | List built-in and configured languages |
| `vex lang add name --test-patterns ... --source-patterns ...` | Add a custom language |
| `vex lang remove name` | Remove a configured language |
| `vex init` | Create default vex.yaml config |
| `vex guide` | Print agent instructions for writing specs |
| `vex version` | Print version, commit, and build date |

## Output

JSON to stdout, diagnostics to stderr. Reports are also written to `.vex/`:

- `.vex/report.json` — full check report (gaps + covered behaviors)
- `.vex/validation.json` — spec validation results
- `.vex/drift.json` — drift detection results

Exit codes: 0 = clean, 1 = gaps/suggestions found, 2 = fatal error.

## Cost Optimization

Vex minimizes LLM cost through three mechanisms:

- **Two-pass check**: Pass 1 sends only tests (cheap). Pass 2 sends source only for uncovered behaviors. Skipped entirely when everything is covered.
- **Drift detection**: `--drift` skips sections with no changes since the last check. Uses git log + uncommitted changes.
- **Section scoping**: `--section` checks only one section at a time.

Cost converges toward zero as test coverage improves and code stabilizes.

## Configuration

`vex.yaml` in the project root:

```yaml
provider: claude-cli
model: opus
max_concurrency: 4
```

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | `claude-cli` | LLM provider |
| `model` | `opus` | Model name passed to provider |
| `max_concurrency` | `4` | Max parallel LLM calls during check |
| `languages` | auto-detect | Override language detection patterns |

## Supported Languages

Auto-detected by file extensions. Multi-language projects are fully supported — vex detects all languages present and collects source and test files across all of them.

**Built-in:** Go, TypeScript, JavaScript, Python, Java, Rust, C, C++, C#, Ruby, Kotlin, Swift, PHP, CUDA

Add custom languages via CLI or config:

```bash
vex lang add mylang --test-patterns "*_test.x" --source-patterns "*.x"
```

```yaml
languages:
  mylang:
    test_patterns: ["*_test.x"]
    source_patterns: ["*.x"]
```

## License

MIT

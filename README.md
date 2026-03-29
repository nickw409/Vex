# Vex

Spec-driven test coverage auditor for AI agent workflows. Verifies that tests fully cover intended behavior described in a spec — not just what happens to be implemented.

Built in Go. ~3,000 lines of application code, ~3,500 lines of tests. Zero external dependencies beyond the standard library and Cobra for CLI.

## The Problem

AI agents write code and tests in a tight loop. Tests pass. But behaviors are missing, partially implemented, or not wired up. Traditional coverage tools measure lines executed, not whether the *intended behavior* is actually tested. The tests verify what was written, not what was intended.

Vex closes that gap. It takes a behavioral spec — a structured YAML document describing what the software *should* do — and uses an LLM to audit whether the test suite actually exercises each described behavior.

## Architecture

```
cmd/vex/              Entry point
internal/
  cli/                Cobra command definitions (check, report, spec, validate, drift, lang, init, guide)
  config/             vex.yaml parsing and validation
  provider/           LLM provider abstraction (currently claude-cli)
  spec/               vexspec.yaml parsing, validation, and generation
  check/              Core gap detection engine (two-pass LLM analysis)
  diff/               Git diff parsing and drift detection
  lang/               Multi-language detection and test file discovery
  report/             Structured JSON output formatting
  log/                Timestamped diagnostic logging (stderr)
  version/            Build-time version injection via ldflags
install.sh            Standalone binary installer (no Go toolchain required)
Makefile              Build, test, cross-compile, and release targets
```

### Two-Pass Gap Detection

The check engine uses a two-pass strategy to minimize LLM cost:

1. **Pass 1 (test-only)**: Sends the behavioral spec and test files to the LLM. Cheap triage — most well-tested behaviors are confirmed covered here and never reach pass 2.
2. **Pass 2 (source + tests)**: Only behaviors flagged as potentially uncovered in pass 1 are re-analyzed with both source code and test files. This deeper pass catches indirect coverage that isn't obvious from tests alone.

Sections are checked concurrently with bounded parallelism. Well-tested codebases skip pass 2 entirely, converging cost toward zero.

### Drift Detection

Drift detection is on by default. Vex uses git log and uncommitted change detection to skip sections with no code or spec changes since the last check. Per-section spec checksums detect edits to behavior descriptions even when no files changed. Skipped sections carry forward their gaps and covered entries from the previous report so unfixed gaps are not lost. Combined with two-pass, this means incremental checks on stable codebases are near-free.

### Multi-Language Detection

Vex auto-detects all languages present in a project simultaneously. A Rust + CUDA project discovers `.rs`, `.cu`, and `.cuh` files in a single directory walk and classifies them correctly across language boundaries. Custom languages can be added at runtime without modifying source.

**Built-in languages (14):** Go, TypeScript, JavaScript, Python, Java, Rust, C, C++, C#, Ruby, Kotlin, Swift, PHP, CUDA

### Agent-First Design

Vex is designed to be invoked by AI agents, not humans directly:

- **JSON output to stdout** — structured reports that agents parse, not human-readable prose
- **Deterministic exit codes** — 0 (clean), 1 (gaps found), 2 (fatal error)
- **Config files over CLI flags** — `vex.yaml` for project config, `.vex/vexspec.yaml` for the spec
- **Guide command** — `vex guide` prints instructions that agents use to write well-formed specs
- **Spec generation** — `vex spec "description"` generates spec sections from natural language task descriptions

## Install

**One-liner** (downloads a prebuilt binary):

```bash
curl -fsSL https://raw.githubusercontent.com/nickw409/vex/main/install.sh | sh
```

Install a specific version or to a custom directory:

```bash
VEX_VERSION=v1.3.0 curl -fsSL https://raw.githubusercontent.com/nickw409/vex/main/install.sh | sh
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

# Check test coverage against the spec (drift detection on by default)
vex check

# View a formatted summary of the last check
vex report
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
    covered:
      - behavior: session-persistence
        reason: tested via e2e binary spawn in tests/e2e/
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
| `vex check` | Check test coverage against spec (drift on by default) |
| `vex check --section Name` | Check a single section |
| `vex check --drift=false` | Force full re-check of all sections |
| `vex report` | Formatted summary of last check |
| `vex validate` | Validate spec for missing behaviors |
| `vex spec "description"` | Generate spec sections from task description |
| `vex spec "desc" --extend Name` | Add behaviors to existing section |
| `vex drift` | Show which sections have changed since last check |
| `vex lang list` | List built-in and configured languages |
| `vex lang add name --test-patterns ... --source-patterns ...` | Add a custom language |
| `vex lang remove name` | Remove a configured language |
| `vex init` | Create default vex.yaml config |
| `vex update` | Update vex to the latest version |
| `vex guide` | Print agent instructions for writing specs |
| `vex version` | Print version, commit, and build date |

## Output

JSON to stdout, diagnostics to stderr. Reports are also written to `.vex/`:

- `.vex/report.json` — full check report (gaps + covered behaviors)
- `.vex/validation.json` — spec validation results
- `.vex/drift.json` — drift detection results

Exit codes: 0 = clean, 1 = gaps/suggestions found, 2 = fatal error.

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

## Releases

Cross-compiled binaries for linux/darwin (amd64/arm64) are published to GitHub Releases. Version, commit hash, and build date are injected at build time via ldflags.

```bash
make publish VERSION=v1.5.0 NOTES="Release notes here"
```

This runs tests, tags, pushes, builds binaries, creates the GitHub release, and updates the Go module proxy in one command.

## License

MIT

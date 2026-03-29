# Vex

Test coverage auditor that verifies tests fully cover intended behavior described in a spec. Designed for AI agent consumption, not direct human use.

## Quick Reference

```bash
go test ./...
make build              # build with version info
make install            # install to $GOPATH/bin
```

## Project Structure

```
cmd/vex/          Entry point (main.go)
internal/
  cli/            Cobra command definitions
  config/         vex.yaml parsing
  provider/       LLM provider abstraction (claude-cli)
  spec/           vexspec.yaml parsing, validation, generation
  check/          Core gap detection engine (two-pass)
  diff/           Git diff and drift detection
  lang/           Language detection and test file discovery
  report/         JSON output formatting
  version/        Build-time version info (set via ldflags)
install.sh        Standalone binary installer (no Go required)
Makefile          Build, test, install, and release targets
```

## Key Conventions

- Go 1.24, module `github.com/nickw409/vex`
- CLI built with `spf13/cobra`
- Tests use stdlib `testing` only — no external test frameworks
- Test files live alongside source (`*_test.go`)
- JSON output to stdout by default (agents consume it, not humans)
- Exit code 0 = no gaps, exit code 1 = gaps found, exit code 2 = fatal error

## Commands

```bash
vex check                                        # check (drift detection on by default)
vex check --drift=false                          # force full re-check of all sections
vex check --section Config                       # check single section
vex report                                       # formatted summary of last check
vex validate                                     # validate spec completeness (concurrent)
vex spec "description"                           # generate spec sections from task
vex spec "description" --extend Config           # add behaviors to existing section
vex drift                                        # check for code changes since last check
vex lang add rust --test-patterns "*_test.rs" --source-patterns "*.rs"
vex lang list                                    # list built-in + configured languages
vex lang remove rust                             # remove a configured language
vex init                                         # create vex.yaml config
vex update                                       # update to latest version
vex guide                                        # print agent instructions for writing specs
vex version                                      # print version, commit, build date
```

## Design Principles

- **Spec-driven** — spec is the source of truth, not the code
- **Two-pass check** — pass 1 sends only tests (cheap triage), pass 2 sends source only for uncovered behaviors; sections run concurrently with adaptive concurrency
- **Drift-aware** — drift detection is on by default, skipping sections where neither code nor spec changed since the last check
- **Language agnostic** — auto-detects Go, TypeScript, JavaScript, Python, Java, Rust, C, C++, C#, Ruby, Kotlin, Swift, PHP, CUDA; multi-language projects are fully supported; custom languages can be added via `vex lang add`
- **Agent-first** — JSON output, config files over CLI flags, guide command for agent instructions
- **Bounded** — spec defines the scope, no infinite nitpicking

## Versioning & Releases

Version is injected at build time via ldflags into `internal/version`. Never hardcode version strings.

**Cutting a release:**

```bash
make publish VERSION=v1.5.0 NOTES="Release notes here"
```

This single command runs tests, tags, pushes, builds cross-compiled binaries, creates the GitHub release with artifacts, and updates the Go module proxy. It fails fast if the working tree is dirty or tests fail.

**What `make release` produces:** cross-compiled tarballs in `dist/` for linux/darwin amd64/arm64. Each contains a single `vex` binary.

**Users install via:** `curl -fsSL https://raw.githubusercontent.com/nickw409/vex/main/install.sh | sh` — downloads the right binary for their OS/arch from GitHub releases. Supports `VEX_VERSION` and `VEX_INSTALL_DIR` env vars.

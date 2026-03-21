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
vex check                                        # check test coverage against spec
vex check --section Config                       # check single section
vex check --drift                                # only check sections changed since last check
vex validate                                     # validate spec completeness
vex spec "description"                           # generate spec sections from task
vex spec "description" --extend Config           # add behaviors to existing section
vex drift                                        # check for code changes since last check
vex lang add rust --test-patterns "*_test.rs" --source-patterns "*.rs"
vex lang list                                    # list built-in + configured languages
vex lang remove rust                             # remove a configured language
vex init                                         # create vex.yaml config
vex guide                                        # print agent instructions for writing specs
vex version                                      # print version, commit, build date
```

## Design Principles

- **Spec-driven** — spec is the source of truth, not the code
- **Two-pass check** — pass 1 sends only tests (cheap triage), pass 2 sends source only for uncovered behaviors
- **Drift-aware** — `--drift` skips clean sections, converging cost toward zero for stable code
- **Language agnostic** — auto-detects Go, TypeScript, JavaScript, Python, Java, Rust, C, C++, C#, Ruby, Kotlin, Swift, PHP, CUDA; multi-language projects are fully supported; custom languages can be added via `vex lang add`
- **Agent-first** — JSON output, config files over CLI flags, guide command for agent instructions
- **Bounded** — spec defines the scope, no infinite nitpicking

## Versioning & Releases

Version is injected at build time via ldflags into `internal/version`. Never hardcode version strings.

**Cutting a release:**

```bash
git tag v0.X.0
git push origin v0.X.0
make VERSION=v0.X.0 release
# Upload dist/*.tar.gz to the GitHub release for the tag
gh release create v0.X.0 dist/*.tar.gz --title "v0.X.0" --notes "Release notes"
```

**What `make release` produces:** cross-compiled tarballs in `dist/` for linux/darwin amd64/arm64. Each contains a single `vex` binary.

**Users install via:** `curl -fsSL https://raw.githubusercontent.com/nickw409/vex/main/install.sh | sh` — downloads the right binary for their OS/arch from GitHub releases. Supports `VEX_VERSION` and `VEX_INSTALL_DIR` env vars.

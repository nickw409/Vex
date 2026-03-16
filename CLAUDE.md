# Vex

Test coverage auditor that verifies tests fully cover intended behavior described in a spec. Designed for AI agent consumption, not direct human use.

## Quick Reference

```bash
go test ./...
go build -o vex ./cmd/vex/
```

## Project Structure

```
cmd/vex/          Entry point (main.go)
internal/
  cli/            Cobra command definitions
  config/         vex.yaml parsing
  provider/       Multi-provider LLM abstraction
  spec/           vexspec.yaml parsing
  check/          Core gap detection engine
  diff/           Git diff scoping
  lang/           Language detection and test file discovery
  report/         JSON output formatting
```

## Key Conventions

- Go 1.24, module `github.com/nwiley/vex`
- CLI built with `spf13/cobra`
- Tests use stdlib `testing` only — no external test frameworks
- Test files live alongside source (`*_test.go`)
- JSON output to stdout by default (agents consume it, not humans)
- Exit code 0 = no gaps, exit code 1 = gaps found

## Commands

```bash
vex check                                        # check test coverage against spec
vex check --section Config                       # check single section
vex check --diff                                 # check only changed files
vex validate                                     # validate spec completeness
vex spec "description"                           # generate spec sections from task
vex spec "description" --extend Config           # add behaviors to existing section
vex drift                                        # check for code changes since last check
vex init                                         # create vex.yaml
vex guide                                        # print agent instructions for writing specs
```

## Design Principles

- **Spec-driven only** — no code-driven behavior extraction. Spec is the source of truth.
- **Language agnostic** — auto-detects language, config overrides for weird projects.
- **Agent-first** — JSON output, config files over CLI flags, guide command for agent instructions.
- **Bounded** — spec defines the scope. No infinite nitpicking.
- **Cheap** — 1-2 LLM calls per run max.

# Vex Optimization Options

Backlog of optimization ideas ranked by impact/effort. Options 1 and 3 were implemented first.

## Option 2: Sonnet for pass 1 (implemented, opt-in)

`pass1_model: sonnet` in vex.yaml. 5x cheaper per token but 27% slower wall-clock.
Sonnet takes longer per call on large prompts. Already plumbed, users opt in when cost > speed.

## Option 4: Prompt cache ordering

Structure user prompts so stable/shared content appears first, maximizing Claude's prefix
cache hits across calls. System prompt caching is already automatic. User prompt content
varies per section, but if large shared test files are placed at the prompt start, prefix
caching can reuse them across calls. Cache reads are 90% cheaper than fresh tokens.

Low effort, ~10-20% cost reduction depending on overlap between sections.

## Option 5: Behavior-scoped file filtering

Instead of sending all files under a section's path, filter to files relevant to specific
behaviors using name/content heuristics. Large sections with broad paths pull in every file.

Highly variable impact. Risk of false gap reports if heuristic misses relevant files.
Needs conservative filtering (include more, not less).

## Option 6: Incremental behavior checking

Track which behaviors map to which files. On re-check, only send behaviors whose relevant
code/tests changed, even within a drifted section. Current drift is section-level; this
makes it behavior-level.

High effort, high impact on incremental runs (60-80% fewer behaviors). No impact on full runs.
Risk: inexact behavior-to-file mapping could produce stale results.

## Option 7: Streaming output format

Use `--output-format stream-json` for faster time-to-first-token. Could enable early
cancellation if pass 1 clearly shows "all covered" early in the stream.

Modest wall-clock improvement. More complex parsing. LLM still generates all tokens.

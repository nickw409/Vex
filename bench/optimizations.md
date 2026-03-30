# Vex Optimization Options

Backlog ranked by impact on large codebases (e.g., Fracture: 26 sections, 167
behaviors, 37K lines Rust+CUDA, estimated $12-18/full run).

Key structural problem: multiple sections share broad paths. In Fracture, 9
sections all use `path: backends/fracture-cuda` and 4 use
`path: crates/fracture-engine/src`. Each section walks, reads, and sends the
*entire* directory to the LLM — the RMSNorm section receives the attention
kernel code, matmul code, memory management code, etc. This is the dominant
cost multiplier.

---

## Option 5: Behavior-scoped file filtering [HIGHEST IMPACT]

**What:** When a section has a broad path, filter to files relevant to the
section's behaviors instead of sending everything. Use filename heuristics
(e.g., section "RMSNorm Kernel" matches files containing "rmsnorm", "rms_norm",
"rms") and optionally grep for behavior keywords in file content.

**Why it matters now:** In Fracture, `backends/fracture-cuda` contains every
kernel, every test, the full backend implementation. Each of 9 sections sends
ALL of it. If each kernel section only sent its own ~2-3 source files and ~1-2
test files instead of all ~30+ files, pass 1 token count drops by ~80% for
those sections. That's 9 sections × 80% reduction = massive savings.

**Estimated impact:** 50-70% total token reduction on Fracture. The 9 CUDA
sections go from sending ~full backend each to sending ~2-4 files each.

**Risk:** A naive heuristic could miss files. Mitigation: fall back to full
directory when no files match, and allow section-level `file:` overrides for
precision. The user can always narrow the spec if the heuristic misses.

**Effort:** Medium. Needs a matching algorithm (section name keywords vs
filenames/content), plus fallback logic.

---

## Option 4: Prompt cache ordering [HIGH IMPACT]

**What:** Structure user prompts so shared/stable content appears at the start,
maximizing Claude's automatic prefix cache. When multiple sections send the same
files, put those files first in the prompt.

**Why it matters now:** If option 5 isn't implemented (or as a complement to
it), 9 sections send the same CUDA backend files. Claude's prefix cache works
on token-prefix matches — if the first N tokens of the prompt are identical
across calls, the cache hits. Currently, section-specific content (name,
description, behaviors) is at the top of the prompt, guaranteeing cache misses.
Moving shared file content to the top would let subsequent CUDA sections reuse
cached tokens.

**Estimated impact:** If 9 CUDA sections each send the same ~50KB of files,
and 8 of 9 hit cache (90% cheaper): saves ~360KB worth of full-price tokens.
Combined with option 5, this becomes less impactful (less shared content to
cache). Best as a fallback if option 5 isn't done.

**Risk:** Low. Prompt reordering has no quality impact. But the cache is
prefix-based, so even small differences in the prompt prefix break it. Needs
careful structuring.

**Effort:** Low-medium. Reorder prompt sections in buildPass1Prompt/
buildPass2Prompt. May need a shared file deduplication layer.

---

## Option 6: Incremental behavior checking [HIGH IMPACT on reruns]

**What:** Track which files are relevant to which behaviors (from previous run
results). On re-check, skip behaviors where none of their relevant files
changed, even within a drifted section.

**Why it matters now:** Fracture has 167 behaviors across 26 sections. During
active development, you might change one file in the CUDA backend. Current
drift detection flags the entire section (e.g., all of "Attention Kernel"
with 6 behaviors). Behavior-level tracking would skip the 5 unchanged
behaviors and only re-check the 1 whose test file changed.

**Estimated impact:** On incremental runs (the common case during development),
60-80% fewer LLM calls. A typical change touching 1-2 files would trigger
2-4 LLM calls instead of 20+.

**Risk:** Behavior-to-file mapping is derived from LLM responses (test_file
field in covered entries). If the mapping is wrong, stale results persist until
a full re-check. Mitigation: periodic full re-check flag, conservative mapping
(if unsure, re-check the behavior).

**Effort:** High. Needs a behavior→file index stored in report.json, diff
integration at the behavior level, and carry-forward logic.

---

## Option 2: Sonnet for pass 1 [IMPLEMENTED, opt-in]

`pass1_model: sonnet` in vex.yaml. 5x cheaper per token.

Measured: 30% cost reduction, 27% slower wall-clock on vex repo. On Fracture
with 26 pass 1 calls, the cost savings scale proportionally but the wall-clock
penalty is worse (more total calls serialized behind the concurrency limit).

Best for CI/scheduled runs where wall-clock is less critical than cost.

---

## Option 7: Streaming output format [LOW IMPACT]

Use `--output-format stream-json` for faster time-to-first-token.

Marginal wall-clock improvement. LLM still generates all tokens. More complex
parsing. Not worth the effort until the bigger wins are captured.

---

## Priority order for Fracture-scale repos

1. **Option 5** — File filtering eliminates the root cause (sending irrelevant
   files). Biggest single win.
2. **Option 6** — Incremental behaviors eliminates redundant re-checks on the
   most common operation (iterative development).
3. **Option 4** — Prompt cache ordering helps when files can't be filtered
   (or as a complement).
4. **Option 2** — Already implemented, opt-in for cost-sensitive runs.
5. **Option 7** — Marginal, defer.

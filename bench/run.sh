#!/usr/bin/env bash
#
# bench/run.sh — Run a profiled vex check against a fixed 3-section subset.
#
# Usage:
#   make bench            # build + run
#   bench/run.sh          # run with existing binary
#   bench/run.sh --diff   # compare against previous baseline
#
# Costs ~$0.30-0.50 per run (5 LLM calls vs ~25 for full repo).
# Writes profile to bench/profile.json and summary to stderr.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BENCH_DIR=bench
PROFILE="$BENCH_DIR/profile.json"
BASELINE="$BENCH_DIR/baseline.json"
SPEC="$BENCH_DIR/benchspec.yaml"

# Ensure binary is built
if [ ! -f ./vex ]; then
    echo "Building vex..." >&2
    go build -o vex ./cmd/vex/
fi

# Run the check with profiling, forcing full re-check (no drift skip)
echo "Running bench (3 sections, drift=false)..." >&2
./vex check --spec "$SPEC" --drift=false --profile 2>&1 >/dev/null || true

# Move profile from .vex/ to bench/
if [ -f .vex/profile.json ]; then
    cp .vex/profile.json "$PROFILE"
fi

echo "" >&2
echo "=== Bench Results ===" >&2

# Parse and display summary
python3 -c "
import json, sys

with open('$PROFILE') as f:
    spans = json.load(f)

# Aggregate
total = 0
pass1_total = 0
pass2_total = 0
input_total = 0
drift_total = 0
sections_p1 = []
sections_p2 = []

for s in spans:
    name, dur = s['name'], s['duration_ms']
    parent = s.get('parent', '')

    if name == 'check:total':
        total = dur
    elif name == 'inputs:total':
        input_total = dur
    elif name == 'drift:total':
        drift_total = dur
    elif name == 'pass 1:llm':
        pass1_total += dur
        sections_p1.append((parent, dur))
    elif name == 'pass 2:llm':
        pass2_total += dur
        sections_p2.append((parent, dur))

print(f'  Total wall-clock:  {total/1000:.1f}s')
print(f'  Input building:    {input_total:.1f}ms')
print(f'  Drift detection:   {drift_total:.1f}ms')
print(f'  Pass 1 LLM total:  {pass1_total/1000:.1f}s ({len(sections_p1)} calls)')
print(f'  Pass 2 LLM total:  {pass2_total/1000:.1f}s ({len(sections_p2)} calls)')
print()

print('  Pass 1 breakdown:')
for name, dur in sorted(sections_p1, key=lambda x: -x[1]):
    print(f'    {name:25s} {dur/1000:.1f}s')

if sections_p2:
    print('  Pass 2 breakdown:')
    for name, dur in sorted(sections_p2, key=lambda x: -x[1]):
        print(f'    {name:25s} {dur/1000:.1f}s')
" >&2

# Compare against baseline if requested
if [ "${1:-}" = "--diff" ] && [ -f "$BASELINE" ]; then
    echo "" >&2
    echo "=== vs Baseline ===" >&2
    python3 -c "
import json

with open('$BASELINE') as f:
    base = json.load(f)
with open('$PROFILE') as f:
    curr = json.load(f)

def extract(spans):
    d = {}
    for s in spans:
        name = s['name']
        parent = s.get('parent', '')
        key = f'{parent}:{name}' if parent else name
        if name in ('check:total', 'inputs:total', 'pass 1:llm', 'pass 2:llm'):
            d[key] = d.get(key, 0) + s['duration_ms']
    return d

b, c = extract(base), extract(curr)
for key in sorted(set(b) | set(c)):
    bv = b.get(key, 0)
    cv = c.get(key, 0)
    if bv == 0:
        continue
    delta = ((cv - bv) / bv) * 100
    arrow = '+' if delta > 0 else ''
    print(f'  {key:40s} {bv/1000:7.1f}s -> {cv/1000:7.1f}s  ({arrow}{delta:.0f}%)')
" >&2
fi

echo "" >&2
echo "Profile: $PROFILE" >&2
echo "To save as baseline: cp $PROFILE $BASELINE" >&2

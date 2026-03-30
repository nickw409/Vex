package check

import (
	"testing"

	"github.com/nickw409/vex/internal/spec"
)

func TestSplitCamelCase(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"RMSNorm", []string{"RMS", "Norm"}},
		{"kvCache", []string{"kv", "Cache"}},
		{"Attention", []string{"Attention"}},
		{"SiLU", []string{"Si", "LU"}},
		{"abc", []string{"abc"}},
		{"", nil},
		{"HTTPServer", []string{"HTTP", "Server"}},
	}

	for _, tt := range tests {
		got := splitCamelCase(tt.input)
		if len(got) != len(tt.want) {
			t.Errorf("splitCamelCase(%q) = %v, want %v", tt.input, got, tt.want)
			continue
		}
		for i := range got {
			if got[i] != tt.want[i] {
				t.Errorf("splitCamelCase(%q)[%d] = %q, want %q", tt.input, i, got[i], tt.want[i])
			}
		}
	}
}

func TestSplitIdentifier(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"RMSNorm Kernel", []string{"RMS", "Norm", "Kernel"}},
		{"rmsnorm-forward", []string{"rmsnorm", "forward"}},
		{"kv_cache_manager", []string{"kv", "cache", "manager"}},
		{"CLI Report", []string{"CLI", "Report"}},
		{"CUDA Memory Management", []string{"CUDA", "Memory", "Management"}},
	}

	for _, tt := range tests {
		got := splitIdentifier(tt.input)
		if len(got) != len(tt.want) {
			t.Errorf("splitIdentifier(%q) = %v, want %v", tt.input, got, tt.want)
			continue
		}
		for i := range got {
			if got[i] != tt.want[i] {
				t.Errorf("splitIdentifier(%q)[%d] = %q, want %q", tt.input, i, got[i], tt.want[i])
			}
		}
	}
}

func TestExtractSectionKeywords(t *testing.T) {
	sec := &spec.Section{Name: "RMSNorm Kernel"}
	behaviors := []spec.Behavior{
		{Name: "rmsnorm-forward"},
		{Name: "rmsnorm-numerical-stability"},
	}

	keywords := extractSectionKeywords(sec, behaviors)

	// Should include "rmsnorm" and "norm" from section name, "rmsnorm" from behaviors (deduped)
	found := make(map[string]bool)
	for _, kw := range keywords {
		found[kw] = true
	}

	if !found["rmsnorm"] {
		t.Error("expected keyword 'rmsnorm'")
	}
	if !found["rms"] {
		t.Error("expected keyword 'rms'")
	}
	// "kernel" is a noise word
	if found["kernel"] {
		t.Error("'kernel' should be filtered as noise")
	}
}

func TestExtractSectionKeywordsAttention(t *testing.T) {
	sec := &spec.Section{Name: "Attention Kernel"}
	behaviors := []spec.Behavior{
		{Name: "attention-prefill"},
		{Name: "attention-decode"},
		{Name: "attention-score-scaling"},
	}

	keywords := extractSectionKeywords(sec, behaviors)
	found := make(map[string]bool)
	for _, kw := range keywords {
		found[kw] = true
	}

	if !found["attention"] {
		t.Error("expected keyword 'attention'")
	}
}

func TestExtractSectionKeywordsKVCache(t *testing.T) {
	sec := &spec.Section{Name: "KV Cache Manager"}
	behaviors := []spec.Behavior{
		{Name: "cache-allocation"},
		{Name: "cache-append-prefill"},
	}

	keywords := extractSectionKeywords(sec, behaviors)
	found := make(map[string]bool)
	for _, kw := range keywords {
		found[kw] = true
	}

	if !found["cache"] {
		t.Error("expected keyword 'cache'")
	}
}

func TestFilterRelevantFiles(t *testing.T) {
	sec := &spec.Section{Name: "RMSNorm Kernel"}
	behaviors := []spec.Behavior{
		{Name: "rmsnorm-forward"},
		{Name: "rmsnorm-numerical-stability"},
	}

	// Simulate a broad CUDA backend path with many files
	files := []string{
		"backends/fracture-cuda/src/rmsnorm.cu",
		"backends/fracture-cuda/src/rmsnorm.rs",
		"backends/fracture-cuda/src/rope.cu",
		"backends/fracture-cuda/src/rope.rs",
		"backends/fracture-cuda/src/attention.cu",
		"backends/fracture-cuda/src/attention.rs",
		"backends/fracture-cuda/src/silu.cu",
		"backends/fracture-cuda/src/embedding.cu",
		"backends/fracture-cuda/src/matmul.cu",
		"backends/fracture-cuda/src/memory.rs",
		"backends/fracture-cuda/src/backend.rs",
		"backends/fracture-cuda/src/lib.rs",
	}

	filtered := FilterRelevantFiles(sec, behaviors, files)

	if len(filtered) >= len(files) {
		t.Fatalf("expected filtering to reduce files, got %d/%d", len(filtered), len(files))
	}

	// Should include rmsnorm files
	hasRmsnorm := false
	for _, f := range filtered {
		if contains(f, "rmsnorm") {
			hasRmsnorm = true
		}
	}
	if !hasRmsnorm {
		t.Error("filtered files should include rmsnorm files")
	}

	// Should NOT include unrelated kernels
	for _, f := range filtered {
		if contains(f, "attention") || contains(f, "rope") || contains(f, "silu") ||
			contains(f, "embedding") || contains(f, "matmul") {
			t.Errorf("filtered files should not include unrelated file: %s", f)
		}
	}
}

func TestFilterRelevantFilesBelowThreshold(t *testing.T) {
	sec := &spec.Section{Name: "Small Section"}
	behaviors := []spec.Behavior{{Name: "some-behavior"}}

	// Fewer files than threshold — should return all
	files := []string{"a.go", "b.go", "c.go"}
	filtered := FilterRelevantFiles(sec, behaviors, files)

	if len(filtered) != len(files) {
		t.Errorf("expected all files below threshold, got %d/%d", len(filtered), len(files))
	}
}

func TestFilterRelevantFilesFallback(t *testing.T) {
	sec := &spec.Section{Name: "Obscure Section"}
	behaviors := []spec.Behavior{{Name: "xyz-behavior"}}

	// Many files but none match keywords — should return all
	files := make([]string, 15)
	for i := range files {
		files[i] = "unrelated_" + string(rune('a'+i)) + ".go"
	}

	filtered := FilterRelevantFiles(sec, behaviors, files)
	if len(filtered) != len(files) {
		t.Errorf("expected fallback to all files, got %d/%d", len(filtered), len(files))
	}
}

func TestFilterRelevantFilesTestDirectory(t *testing.T) {
	sec := &spec.Section{Name: "Attention Kernel"}
	behaviors := []spec.Behavior{
		{Name: "attention-prefill"},
		{Name: "attention-decode"},
	}

	// Test files in a tests/ subdirectory
	files := []string{
		"backends/fracture-cuda/tests/test_rmsnorm.rs",
		"backends/fracture-cuda/tests/test_attention.rs",
		"backends/fracture-cuda/tests/test_rope.rs",
		"backends/fracture-cuda/tests/test_silu.rs",
		"backends/fracture-cuda/tests/test_embedding.rs",
		"backends/fracture-cuda/tests/test_matmul.rs",
		"backends/fracture-cuda/tests/test_memory.rs",
		"backends/fracture-cuda/tests/test_profiling.rs",
		"backends/fracture-cuda/tests/test_backend.rs",
		"backends/fracture-cuda/tests/helpers.rs",
		"backends/fracture-cuda/tests/reference.rs",
	}

	filtered := FilterRelevantFiles(sec, behaviors, files)

	if len(filtered) >= len(files) {
		t.Fatalf("expected filtering, got %d/%d", len(filtered), len(files))
	}

	// Should include attention test
	hasAttention := false
	for _, f := range filtered {
		if contains(f, "attention") {
			hasAttention = true
		}
	}
	if !hasAttention {
		t.Error("should include test_attention.rs")
	}

	// Should not include unrelated tests
	for _, f := range filtered {
		if contains(f, "rmsnorm") || contains(f, "rope") || contains(f, "silu") {
			t.Errorf("should not include unrelated test: %s", f)
		}
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) > 0 && containsLower(s, substr))
}

func containsLower(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

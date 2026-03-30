package check

import (
	"path/filepath"
	"strings"
	"unicode"

	"github.com/nickw409/vex/internal/spec"
)

// fileFilterThreshold is the minimum number of files in a path before
// filtering is attempted. Below this, the path is narrow enough that
// sending everything is fine.
const fileFilterThreshold = 10

// FilterRelevantFiles reduces a file list to those relevant to the given
// section when a broad path yields many files. Uses section name and
// behavior names as keywords matched against filenames.
//
// Returns the original list when:
// - total file count is at or below the threshold
// - no keywords could be extracted
// - no files matched any keyword (conservative fallback)
func FilterRelevantFiles(sec *spec.Section, behaviors []spec.Behavior, files []string) []string {
	if len(files) <= fileFilterThreshold {
		return files
	}

	keywords := extractSectionKeywords(sec, behaviors)
	if len(keywords) == 0 {
		return files
	}

	var matched []string
	for _, f := range files {
		if fileMatchesKeywords(f, keywords) {
			matched = append(matched, f)
		}
	}

	if len(matched) == 0 {
		return files
	}
	return matched
}

// extractSectionKeywords derives search keywords from the section name
// and its behavior names. Returns lowercase keywords of length >= 3,
// excluding common noise words.
func extractSectionKeywords(sec *spec.Section, behaviors []spec.Behavior) []string {
	seen := make(map[string]bool)
	var keywords []string

	add := func(word string) {
		w := strings.ToLower(word)
		if len(w) < 3 || noiseWords[w] || seen[w] {
			return
		}
		seen[w] = true
		keywords = append(keywords, w)
	}

	// Split section name on spaces, camelCase boundaries, etc.
	for _, w := range splitIdentifier(sec.Name) {
		add(w)
	}

	// Extract the common prefix from behavior names — often the strongest signal.
	// e.g., "rmsnorm-forward", "rmsnorm-numerical-stability" → "rmsnorm"
	for _, b := range behaviors {
		parts := splitIdentifier(b.Name)
		if len(parts) > 0 {
			add(parts[0])
		}
	}

	return keywords
}

// splitIdentifier breaks a string on spaces, hyphens, underscores, and
// camelCase boundaries. "RMSNorm Kernel" → ["RMSNorm", "Kernel"],
// "rmsnorm-forward" → ["rmsnorm", "forward"].
func splitIdentifier(s string) []string {
	// First split on common delimiters.
	s = strings.NewReplacer("-", " ", "_", " ").Replace(s)
	words := strings.Fields(s)

	// Then split camelCase within each word.
	var result []string
	for _, w := range words {
		result = append(result, splitCamelCase(w)...)
	}
	return result
}

// splitCamelCase splits "RMSNorm" → ["RMS", "Norm"], "kvCache" → ["kv", "Cache"].
func splitCamelCase(s string) []string {
	if len(s) == 0 {
		return nil
	}

	var parts []string
	runes := []rune(s)
	start := 0

	for i := 1; i < len(runes); i++ {
		// Split before an uppercase letter that follows a lowercase letter: "kvCache" → "kv" + "Cache"
		if unicode.IsUpper(runes[i]) && unicode.IsLower(runes[i-1]) {
			parts = append(parts, string(runes[start:i]))
			start = i
			continue
		}
		// Split before a lowercase letter that follows an uppercase run: "RMSNorm" → "RMS" + "Norm"
		if i > start+1 && unicode.IsUpper(runes[i-1]) && unicode.IsLower(runes[i]) {
			parts = append(parts, string(runes[start:i-1]))
			start = i - 1
		}
	}
	parts = append(parts, string(runes[start:]))
	return parts
}

// fileMatchesKeywords checks if a filename (or its parent directory) contains
// any of the keywords. Matching is done on the lowercased path components.
func fileMatchesKeywords(path string, keywords []string) bool {
	// Check filename without extension.
	base := strings.ToLower(strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)))
	// Also check the immediate parent directory name.
	dir := strings.ToLower(filepath.Base(filepath.Dir(path)))

	for _, kw := range keywords {
		if strings.Contains(base, kw) || strings.Contains(dir, kw) {
			return true
		}
	}
	return false
}

// noiseWords are common words that appear in many section/behavior names
// but carry no file-matching signal.
var noiseWords = map[string]bool{
	"the": true, "and": true, "for": true, "with": true,
	"from": true, "that": true, "this": true, "into": true,
	"when": true, "each": true, "all": true, "any": true,
	// Common spec/behavior terms that don't help match files.
	"kernel":      true,
	"manager":     true,
	"handler":     true,
	"validation":  true,
	"correctness": true,
	"forward":     true,
	"error":       true,
	"errors":      true,
	"status":      true,
	"test":        true,
	"tests":       true,
}

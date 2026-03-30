package check

import (
	"path/filepath"
	"strings"
)

// extractTestSignatures returns a condensed version of a test file containing
// only function signatures, subtest declarations, and assertion lines. This
// reduces pass 1 prompt size by ~70% while preserving enough structure for the
// LLM to determine which behaviors are covered.
//
// Falls back to full content for unrecognized file types.
func extractTestSignatures(filename, content string) string {
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".go":
		return extractGo(content)
	case ".rs":
		return extractRust(content)
	case ".py":
		return extractPython(content)
	case ".ts", ".js":
		return extractJS(content)
	case ".java", ".kt":
		return extractJava(content)
	default:
		return content
	}
}

// extractGo extracts Go test signatures, subtests, and assertions.
func extractGo(content string) string {
	var out []string
	for _, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		if shouldKeepGo(trimmed) {
			out = append(out, line)
		}
	}
	if len(out) == 0 {
		return content
	}
	return strings.Join(out, "\n")
}

func shouldKeepGo(line string) bool {
	// Function signatures
	if strings.HasPrefix(line, "func Test") || strings.HasPrefix(line, "func Benchmark") {
		return true
	}
	// Subtests
	if strings.Contains(line, "t.Run(") || strings.Contains(line, ".Run(") {
		return true
	}
	// Assertions and failures
	for _, pattern := range []string{
		"t.Error", "t.Fatal", "t.Fail", "t.Skip",
		"t.Helper()",
		"assert.", "require.",
		"if err", "!= nil",
	} {
		if strings.Contains(line, pattern) {
			return true
		}
	}
	return false
}

// extractRust extracts Rust test signatures and assertions.
func extractRust(content string) string {
	var out []string
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if shouldKeepRust(trimmed) {
			out = append(out, line)
			continue
		}
		// Keep fn line after #[test] or #[tokio::test]
		if i > 0 {
			prevTrimmed := strings.TrimSpace(lines[i-1])
			if (strings.HasPrefix(prevTrimmed, "#[test]") || strings.Contains(prevTrimmed, "::test]")) &&
				strings.HasPrefix(trimmed, "fn ") {
				out = append(out, line)
			}
		}
	}
	if len(out) == 0 {
		return content
	}
	return strings.Join(out, "\n")
}

func shouldKeepRust(line string) bool {
	if strings.HasPrefix(line, "#[test]") || strings.HasPrefix(line, "#[tokio::test]") {
		return true
	}
	if strings.HasPrefix(line, "#[cfg(test)]") {
		return true
	}
	if strings.HasPrefix(line, "mod ") && strings.Contains(line, "test") {
		return true
	}
	for _, pattern := range []string{
		"assert!", "assert_eq!", "assert_ne!",
		"panic!", ".unwrap()", ".expect(",
		".is_err()", ".is_ok()",
	} {
		if strings.Contains(line, pattern) {
			return true
		}
	}
	return false
}

// extractPython extracts Python test signatures and assertions.
func extractPython(content string) string {
	var out []string
	for _, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		if shouldKeepPython(trimmed) {
			out = append(out, line)
		}
	}
	if len(out) == 0 {
		return content
	}
	return strings.Join(out, "\n")
}

func shouldKeepPython(line string) bool {
	if strings.HasPrefix(line, "def test_") || strings.HasPrefix(line, "async def test_") {
		return true
	}
	if strings.HasPrefix(line, "class Test") {
		return true
	}
	for _, pattern := range []string{
		"assert ", "assert_", "self.assert",
		"pytest.raises", "pytest.mark",
		"@pytest", "@mock",
	} {
		if strings.Contains(line, pattern) {
			return true
		}
	}
	return false
}

// extractJS extracts JS/TS test signatures and assertions.
func extractJS(content string) string {
	var out []string
	for _, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		if shouldKeepJS(trimmed) {
			out = append(out, line)
		}
	}
	if len(out) == 0 {
		return content
	}
	return strings.Join(out, "\n")
}

func shouldKeepJS(line string) bool {
	for _, pattern := range []string{
		"describe(", "it(", "test(",
		"expect(", "assert.",
		"beforeEach(", "afterEach(",
		"beforeAll(", "afterAll(",
		"jest.mock(", "jest.spy",
	} {
		if strings.Contains(line, pattern) {
			return true
		}
	}
	return false
}

// extractJava extracts Java/Kotlin test signatures and assertions.
func extractJava(content string) string {
	var out []string
	for _, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		if shouldKeepJava(trimmed) {
			out = append(out, line)
		}
	}
	if len(out) == 0 {
		return content
	}
	return strings.Join(out, "\n")
}

func shouldKeepJava(line string) bool {
	if strings.HasPrefix(line, "@Test") || strings.HasPrefix(line, "@ParameterizedTest") {
		return true
	}
	if strings.HasPrefix(line, "@Before") || strings.HasPrefix(line, "@After") {
		return true
	}
	for _, pattern := range []string{
		"assert", "Assert.",
		"verify(", "when(",
		"public void test", "fun test",
	} {
		if strings.Contains(line, pattern) {
			return true
		}
	}
	return false
}

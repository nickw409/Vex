package lang

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/nickw409/vex/internal/config"
)

type Language struct {
	Name           string
	TestPatterns   []string
	SourcePatterns []string
}

var builtinLanguages = map[string]Language{
	"go": {
		Name:           "go",
		TestPatterns:   []string{"*_test.go"},
		SourcePatterns: []string{"*.go"},
	},
	"typescript": {
		Name:           "typescript",
		TestPatterns:   []string{"*.test.ts", "*.spec.ts"},
		SourcePatterns: []string{"*.ts"},
	},
	"javascript": {
		Name:           "javascript",
		TestPatterns:   []string{"*.test.js", "*.spec.js"},
		SourcePatterns: []string{"*.js"},
	},
	"python": {
		Name:           "python",
		TestPatterns:   []string{"test_*.py", "*_test.py"},
		SourcePatterns: []string{"*.py"},
	},
	"java": {
		Name:           "java",
		TestPatterns:   []string{"*Test.java"},
		SourcePatterns: []string{"*.java"},
	},
	"rust": {
		Name:           "rust",
		TestPatterns:   []string{"*_test.rs"},
		SourcePatterns: []string{"*.rs"},
	},
	"c": {
		Name:           "c",
		TestPatterns:   []string{"test_*.c", "*_test.c"},
		SourcePatterns: []string{"*.c", "*.h"},
	},
	"cpp": {
		Name:           "cpp",
		TestPatterns:   []string{"test_*.cpp", "*_test.cpp", "test_*.cc", "*_test.cc"},
		SourcePatterns: []string{"*.cpp", "*.cc", "*.hpp"},
	},
	"csharp": {
		Name:           "csharp",
		TestPatterns:   []string{"*Tests.cs", "*Test.cs"},
		SourcePatterns: []string{"*.cs"},
	},
	"ruby": {
		Name:           "ruby",
		TestPatterns:   []string{"test_*.rb", "*_test.rb", "*_spec.rb"},
		SourcePatterns: []string{"*.rb"},
	},
	"kotlin": {
		Name:           "kotlin",
		TestPatterns:   []string{"*Test.kt", "*Tests.kt"},
		SourcePatterns: []string{"*.kt"},
	},
	"swift": {
		Name:           "swift",
		TestPatterns:   []string{"*Tests.swift", "*Test.swift"},
		SourcePatterns: []string{"*.swift"},
	},
	"php": {
		Name:           "php",
		TestPatterns:   []string{"*Test.php", "*_test.php"},
		SourcePatterns: []string{"*.php"},
	},
	"cuda": {
		Name:           "cuda",
		TestPatterns:   []string{"test_*.cu", "*_test.cu"},
		SourcePatterns: []string{"*.cu", "*.cuh"},
	},
}

// BuiltinLanguages returns a copy of the built-in language definitions.
func BuiltinLanguages() map[string]Language {
	out := make(map[string]Language, len(builtinLanguages))
	for k, v := range builtinLanguages {
		out[k] = v
	}
	return out
}

// DetectAll returns all languages found in the directory, sorted by file count
// (most files first). Returns an error only if no languages are detected.
func DetectAll(dir string, overrides map[string]config.LanguageConfig) ([]*Language, error) {
	// Build the full set of known languages: builtins + overrides.
	// Overrides replace builtins with the same name.
	all := make(map[string]Language, len(builtinLanguages)+len(overrides))
	for k, v := range builtinLanguages {
		all[k] = v
	}
	for name, lc := range overrides {
		all[name] = Language{
			Name:           name,
			TestPatterns:   lc.TestPatterns,
			SourcePatterns: lc.SourcePatterns,
		}
	}

	// Build extension-to-language map from source patterns.
	// Patterns like "*.rs" map extension ".rs" to language name.
	// Overrides are applied second so they take priority over builtins.
	extMap := make(map[string]string)
	for name, l := range builtinLanguages {
		for _, p := range l.SourcePatterns {
			ext := filepath.Ext(p)
			if ext != "" {
				extMap[strings.ToLower(ext)] = name
			}
		}
	}
	for name, lc := range overrides {
		for _, p := range lc.SourcePatterns {
			ext := filepath.Ext(p)
			if ext != "" {
				extMap[strings.ToLower(ext)] = name
			}
		}
	}

	hasPackageJSON := false
	if _, err := os.Stat(filepath.Join(dir, "package.json")); err == nil {
		hasPackageJSON = true
	}

	counts := make(map[string]int)

	filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			base := d.Name()
			if base == "node_modules" || base == "vendor" || base == ".git" {
				return filepath.SkipDir
			}
			return nil
		}

		ext := strings.ToLower(filepath.Ext(d.Name()))
		if langName, ok := extMap[ext]; ok {
			// JavaScript requires package.json to avoid false positives.
			if langName == "javascript" && !hasPackageJSON {
				return nil
			}
			counts[langName]++
		}
		return nil
	})

	if len(counts) == 0 {
		return nil, fmt.Errorf("no supported language detected in %s", dir)
	}

	type langCount struct {
		name  string
		count int
	}
	var sorted []langCount
	for name, count := range counts {
		sorted = append(sorted, langCount{name, count})
	}
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].count != sorted[j].count {
			return sorted[i].count > sorted[j].count
		}
		return sorted[i].name < sorted[j].name
	})

	var result []*Language
	for _, lc := range sorted {
		l := all[lc.name]
		result = append(result, &l)
	}
	return result, nil
}

// Detect returns the single most-prevalent language in the directory.
// For multi-language projects, use DetectAll instead.
func Detect(dir string, overrides map[string]config.LanguageConfig) (*Language, error) {
	langs, err := DetectAll(dir, overrides)
	if err != nil {
		return nil, err
	}
	return langs[0], nil
}

func FindFiles(dir string, lang *Language) (sourceFiles []string, testFiles []string, err error) {
	err = filepath.WalkDir(dir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			base := d.Name()
			if base == "node_modules" || base == "vendor" || base == ".git" {
				return filepath.SkipDir
			}
			return nil
		}

		name := d.Name()

		if isTest(name, lang.TestPatterns) {
			testFiles = append(testFiles, path)
			return nil
		}

		if matchesAny(name, lang.SourcePatterns) {
			sourceFiles = append(sourceFiles, path)
		}

		return nil
	})

	return
}

// DetectAndFind performs language detection and file discovery in a single
// directory walk, avoiding the cost of two separate traversals. It returns
// the detected languages (sorted by file count, most first) along with
// classified source and test file lists.
func DetectAndFind(dir string, overrides map[string]config.LanguageConfig) (langs []*Language, sourceFiles []string, testFiles []string, err error) {
	// Build the full set of known languages: builtins + overrides.
	all := make(map[string]Language, len(builtinLanguages)+len(overrides))
	for k, v := range builtinLanguages {
		all[k] = v
	}
	for name, lc := range overrides {
		all[name] = Language{
			Name:           name,
			TestPatterns:   lc.TestPatterns,
			SourcePatterns: lc.SourcePatterns,
		}
	}

	// Build extension-to-language map from source patterns.
	extMap := make(map[string]string)
	for name, l := range builtinLanguages {
		for _, p := range l.SourcePatterns {
			ext := filepath.Ext(p)
			if ext != "" {
				extMap[strings.ToLower(ext)] = name
			}
		}
	}
	for name, lc := range overrides {
		for _, p := range lc.SourcePatterns {
			ext := filepath.Ext(p)
			if ext != "" {
				extMap[strings.ToLower(ext)] = name
			}
		}
	}

	hasPackageJSON := false
	if _, err := os.Stat(filepath.Join(dir, "package.json")); err == nil {
		hasPackageJSON = true
	}

	// First pass: count files per language to determine which languages are present.
	counts := make(map[string]int)
	// Collect all file paths for classification in the second phase.
	var allFiles []string

	filepath.WalkDir(dir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			base := d.Name()
			if base == "node_modules" || base == "vendor" || base == ".git" {
				return filepath.SkipDir
			}
			return nil
		}

		ext := strings.ToLower(filepath.Ext(d.Name()))
		if langName, ok := extMap[ext]; ok {
			if langName == "javascript" && !hasPackageJSON {
				return nil
			}
			counts[langName]++
			allFiles = append(allFiles, path)
		}
		return nil
	})

	if len(counts) == 0 {
		return nil, nil, nil, fmt.Errorf("no supported language detected in %s", dir)
	}

	// Sort languages by file count (most first).
	type langCount struct {
		name  string
		count int
	}
	var sorted []langCount
	for name, count := range counts {
		sorted = append(sorted, langCount{name, count})
	}
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].count != sorted[j].count {
			return sorted[i].count > sorted[j].count
		}
		return sorted[i].name < sorted[j].name
	})

	for _, lc := range sorted {
		l := all[lc.name]
		langs = append(langs, &l)
	}

	// Classify collected files using the detected languages.
	for _, path := range allFiles {
		name := filepath.Base(path)
		isTest := false
		for _, l := range langs {
			if matchesAny(name, l.TestPatterns) {
				testFiles = append(testFiles, path)
				isTest = true
				break
			}
		}
		if !isTest {
			for _, l := range langs {
				if matchesAny(name, l.SourcePatterns) {
					sourceFiles = append(sourceFiles, path)
					break
				}
			}
		}
	}

	return langs, sourceFiles, testFiles, nil
}

// FindFilesMulti discovers source and test files for multiple languages in a single walk.
func FindFilesMulti(dir string, langs []*Language) (sourceFiles []string, testFiles []string, err error) {
	err = filepath.WalkDir(dir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			base := d.Name()
			if base == "node_modules" || base == "vendor" || base == ".git" {
				return filepath.SkipDir
			}
			return nil
		}

		name := d.Name()

		// Check test patterns across all languages first.
		for _, l := range langs {
			if matchesAny(name, l.TestPatterns) {
				testFiles = append(testFiles, path)
				return nil
			}
		}

		// Then check source patterns.
		for _, l := range langs {
			if matchesAny(name, l.SourcePatterns) {
				sourceFiles = append(sourceFiles, path)
				return nil
			}
		}

		return nil
	})

	return
}

// IsTestFile reports whether the given filename matches the language's test patterns.
func IsTestFile(filename string, lang *Language) bool {
	return matchesAny(filepath.Base(filename), lang.TestPatterns)
}

// IsTestFileMulti reports whether the given filename matches any language's test patterns.
func IsTestFileMulti(filename string, langs []*Language) bool {
	base := filepath.Base(filename)
	for _, l := range langs {
		if matchesAny(base, l.TestPatterns) {
			return true
		}
	}
	return false
}

func isTest(name string, patterns []string) bool {
	return matchesAny(name, patterns)
}

func matchesAny(name string, patterns []string) bool {
	for _, p := range patterns {
		if matched, _ := filepath.Match(p, name); matched {
			return true
		}
	}
	return false
}

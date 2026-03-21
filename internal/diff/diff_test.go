package diff

import (
	"path/filepath"
	"testing"

	"github.com/nickw409/vex/internal/lang"
)

func TestFilterByLanguageGo(t *testing.T) {
	langs := []*lang.Language{{
		Name:           "go",
		TestPatterns:   []string{"*_test.go"},
		SourcePatterns: []string{"*.go"},
	}}

	files := []string{
		"/repo/main.go",
		"/repo/main_test.go",
		"/repo/handler.go",
		"/repo/README.md",
	}

	src, tests := FilterByLanguage(files, langs)

	if len(src) != 2 {
		t.Errorf("expected 2 source files, got %d: %v", len(src), src)
	}
	if len(tests) != 1 {
		t.Errorf("expected 1 test file, got %d: %v", len(tests), tests)
	}
}

func TestFilterByLanguageTS(t *testing.T) {
	langs := []*lang.Language{{
		Name:           "typescript",
		TestPatterns:   []string{"*.test.ts", "*.spec.ts"},
		SourcePatterns: []string{"*.ts"},
	}}

	files := []string{
		"/repo/app.ts",
		"/repo/app.test.ts",
		"/repo/utils.spec.ts",
		"/repo/style.css",
	}

	src, tests := FilterByLanguage(files, langs)

	if len(src) != 1 {
		t.Errorf("expected 1 source file, got %d: %v", len(src), src)
	}
	if len(tests) != 2 {
		t.Errorf("expected 2 test files, got %d: %v", len(tests), tests)
	}
}

func TestFilterByLanguageNoMatches(t *testing.T) {
	langs := []*lang.Language{{
		Name:           "go",
		TestPatterns:   []string{"*_test.go"},
		SourcePatterns: []string{"*.go"},
	}}

	files := []string{"/repo/README.md", "/repo/Makefile"}

	src, tests := FilterByLanguage(files, langs)

	if len(src) != 0 {
		t.Errorf("expected 0 source files, got %d", len(src))
	}
	if len(tests) != 0 {
		t.Errorf("expected 0 test files, got %d", len(tests))
	}
}

func TestFilterByLanguageMulti(t *testing.T) {
	langs := []*lang.Language{
		{
			Name:           "rust",
			TestPatterns:   []string{"*_test.rs"},
			SourcePatterns: []string{"*.rs"},
		},
		{
			Name:           "cuda",
			TestPatterns:   []string{"test_*.cu", "*_test.cu"},
			SourcePatterns: []string{"*.cu", "*.cuh"},
		},
	}

	files := []string{
		"/repo/main.rs",
		"/repo/lib_test.rs",
		"/repo/kernel.cu",
		"/repo/kernel.cuh",
		"/repo/test_kernel.cu",
		"/repo/README.md",
	}

	src, tests := FilterByLanguage(files, langs)

	if len(src) != 3 {
		t.Errorf("expected 3 source files, got %d: %v", len(src), src)
	}
	if len(tests) != 2 {
		t.Errorf("expected 2 test files, got %d: %v", len(tests), tests)
	}
}

func TestChangedFilesNonGitDir(t *testing.T) {
	dir := t.TempDir()
	_, err := ChangedFiles(dir)
	if err == nil {
		t.Error("expected error for non-git directory")
	}
}

func TestChangedFilesAbsolutePaths(t *testing.T) {
	dir := setupGitRepo(t)

	// Create and commit a file
	writeFile(t, filepath.Join(dir, "main.go"), "package main")
	gitAdd(t, dir, ".")
	gitCommit(t, dir, "initial")

	// Modify the file so it shows up in diff
	writeFile(t, filepath.Join(dir, "main.go"), "package main\n// changed")

	files, err := ChangedFiles(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("expected changed files, got none")
	}
	for _, f := range files {
		if !filepath.IsAbs(f) {
			t.Errorf("expected absolute path, got %q", f)
		}
	}
}

func TestChangedFilesNoChanges(t *testing.T) {
	dir := setupGitRepo(t)

	// Create and commit a file, then don't modify anything
	writeFile(t, filepath.Join(dir, "main.go"), "package main")
	gitAdd(t, dir, ".")
	gitCommit(t, dir, "initial")

	files, err := ChangedFiles(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if files != nil {
		t.Errorf("expected nil for no changes, got %v", files)
	}
}

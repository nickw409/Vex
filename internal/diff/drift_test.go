package diff

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDrift_NoChanges(t *testing.T) {
	dir := setupGitRepo(t)

	// Create a file and commit it
	writeFile(t, filepath.Join(dir, "src", "main.go"), "package main")
	gitAdd(t, dir, ".")
	gitCommit(t, dir, "initial")

	// Check drift since 1 second in the future — nothing should match
	result, err := Drift(dir, []string{"src"}, time.Now().Add(1*time.Second))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Fatalf("expected no drift, got %v", result)
	}
}

func TestDrift_WithChanges(t *testing.T) {
	dir := setupGitRepo(t)

	writeFile(t, filepath.Join(dir, "src", "main.go"), "package main")
	gitAdd(t, dir, ".")
	gitCommit(t, dir, "initial")

	since := time.Now().Add(-1 * time.Second)

	// Make a change and commit
	writeFile(t, filepath.Join(dir, "src", "main.go"), "package main\n// changed")
	gitAdd(t, dir, ".")
	gitCommit(t, dir, "update")

	result, err := Drift(dir, []string{"src"}, since)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected drift, got nil")
	}
	if len(result.ChangedFiles) == 0 {
		t.Fatal("expected changed files")
	}
}

func TestDrift_DifferentPaths(t *testing.T) {
	dir := setupGitRepo(t)

	writeFile(t, filepath.Join(dir, "src", "main.go"), "package main")
	gitAdd(t, dir, ".")
	gitCommit(t, dir, "initial src")

	since := time.Now().Add(-1 * time.Second)

	// Only lib changes after since
	writeFile(t, filepath.Join(dir, "lib", "util.go"), "package lib")
	gitAdd(t, dir, ".")
	gitCommit(t, dir, "add lib")

	// src should not have drifted
	result, err := Drift(dir, []string{"src"}, time.Now().Add(1*time.Second))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Fatalf("expected no drift for src, got %v", result)
	}

	// lib should have drifted
	result, err = Drift(dir, []string{"lib"}, since)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected drift for lib")
	}
}

func TestDrift_DeduplicatesOverlappingPaths(t *testing.T) {
	dir := setupGitRepo(t)

	// Create a file under src/sub/
	writeFile(t, filepath.Join(dir, "src", "sub", "main.go"), "package main")
	gitAdd(t, dir, ".")
	gitCommit(t, dir, "initial")

	since := time.Now().Add(-1 * time.Second)

	// Modify the file
	writeFile(t, filepath.Join(dir, "src", "sub", "main.go"), "package main\n// changed")
	gitAdd(t, dir, ".")
	gitCommit(t, dir, "update")

	// Pass overlapping paths: "src" and "src/sub" both cover the same file
	result, err := Drift(dir, []string{"src", "src/sub"}, since)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected drift result, got nil")
	}

	// The file should appear only once despite overlapping paths
	count := 0
	for _, f := range result.ChangedFiles {
		if strings.Contains(f, "main.go") {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected main.go to appear once (deduped), got %d times in %v", count, result.ChangedFiles)
	}
}

func TestReportModTime_NoFile(t *testing.T) {
	dir := t.TempDir()
	mt := ReportModTime(dir)
	if !mt.IsZero() {
		t.Fatalf("expected zero time, got %v", mt)
	}
}

func TestReportModTime_WithFile(t *testing.T) {
	dir := t.TempDir()
	vexDir := filepath.Join(dir, ".vex")
	os.MkdirAll(vexDir, 0755)
	os.WriteFile(filepath.Join(vexDir, "report.json"), []byte("{}"), 0644)

	mt := ReportModTime(dir)
	if mt.IsZero() {
		t.Fatal("expected non-zero time")
	}
}

func setupGitRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	run(t, dir, "git", "init")
	run(t, dir, "git", "config", "user.email", "test@test.com")
	run(t, dir, "git", "config", "user.name", "Test")
	return dir
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	os.MkdirAll(filepath.Dir(path), 0755)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
}

func gitAdd(t *testing.T, dir, path string) {
	t.Helper()
	run(t, dir, "git", "add", path)
}

func gitCommit(t *testing.T, dir, msg string) {
	t.Helper()
	run(t, dir, "git", "commit", "-m", msg)
}

func run(t *testing.T, dir string, name string, args ...string) {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s %v failed: %v\n%s", name, args, err, out)
	}
}

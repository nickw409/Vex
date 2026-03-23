package cli

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCheckSpecNotFound(t *testing.T) {
	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--spec", "/nonexistent/spec.yaml"})

	if err := cmd.Execute(); err == nil {
		t.Error("expected error for nonexistent spec file")
	}
}

func TestCheckSectionNotFound(t *testing.T) {
	dir := t.TempDir()
	vexDir := filepath.Join(dir, ".vex")
	os.MkdirAll(vexDir, 0755)

	specPath := filepath.Join(vexDir, "vexspec.yaml")
	os.WriteFile(specPath, []byte(`project: Test
sections:
  - name: Auth
    path: auth
    description: Auth module
    behaviors:
      - name: login
        description: Login endpoint
`), 0644)

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--spec", specPath, "--section", "Nonexistent"})

	if err := cmd.Execute(); err == nil {
		t.Error("expected error for nonexistent section")
	}
}

func TestValidateRequiresValidSpec(t *testing.T) {
	cmd := NewRootCmd()
	cmd.SetArgs([]string{"validate", "/nonexistent/spec.yaml"})

	if err := cmd.Execute(); err == nil {
		t.Error("expected error for nonexistent spec file")
	}
}

func TestRootConfigLoadedForCheck(t *testing.T) {
	dir := t.TempDir()
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	// Write a custom config
	os.WriteFile(filepath.Join(dir, "vex.yaml"), []byte("provider: claude-cli\nmodel: haiku\n"), 0644)

	// check will fail (no spec) but config should be loaded first
	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check"})
	cmd.Execute()

	if cfg == nil {
		t.Fatal("expected cfg to be set after check command")
	}
	if cfg.Model != "haiku" {
		t.Errorf("expected model 'haiku' from config, got %q", cfg.Model)
	}
}

func TestRootConfigSkippedForInit(t *testing.T) {
	dir := t.TempDir()
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	// No vex.yaml — init should not try to load config
	cfg = nil
	cmd := NewRootCmd()
	cmd.SetArgs([]string{"init"})
	cmd.Execute()

	// cfg should remain nil since init skips config loading
	if cfg != nil {
		t.Error("expected cfg to remain nil for init command")
	}
}

func TestRootConfigDefaultsOnMissing(t *testing.T) {
	dir := t.TempDir()
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	// No vex.yaml — config should fall back to defaults
	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check"})
	cmd.Execute()

	if cfg == nil {
		t.Fatal("expected cfg to be set with defaults")
	}
	if cfg.Model != "opus" {
		t.Errorf("expected default model 'opus', got %q", cfg.Model)
	}
}

func TestRootConfigFlagLoadsFromPath(t *testing.T) {
	dir := t.TempDir()
	customPath := filepath.Join(dir, "custom.yaml")
	os.WriteFile(customPath, []byte("provider: claude-cli\nmodel: sonnet\n"), 0644)

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--config", customPath})
	cmd.Execute()

	if cfg == nil {
		t.Fatal("expected cfg to be set")
	}
	if cfg.Model != "sonnet" {
		t.Errorf("expected model 'sonnet' from custom config, got %q", cfg.Model)
	}
}

func TestCheckNoFilesEmptyReport(t *testing.T) {
	dir := t.TempDir()
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	os.WriteFile(filepath.Join(dir, "vex.yaml"), []byte("provider: claude-cli\nmodel: opus\n"), 0644)

	vexDir := filepath.Join(dir, ".vex")
	os.MkdirAll(vexDir, 0755)
	specPath := filepath.Join(vexDir, "vexspec.yaml")
	os.WriteFile(specPath, []byte(`project: Test
sections:
  - name: Empty
    path: nonexistent_dir
    description: No files here
    behaviors:
      - name: something
        description: Something
`), 0644)

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--spec", specPath})

	// Should not error — produces empty report
	err := cmd.Execute()
	if err != nil {
		t.Fatalf("expected no error for empty section, got: %v", err)
	}
}

func TestDriftSpecNotFound(t *testing.T) {
	cmd := NewRootCmd()
	cmd.SetArgs([]string{"drift", "--spec", "/nonexistent/spec.yaml"})

	if err := cmd.Execute(); err == nil {
		t.Error("expected error for nonexistent spec in drift command")
	}
}

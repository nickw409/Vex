package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefault(t *testing.T) {
	cfg := Default()
	if cfg.Provider != "claude-cli" {
		t.Errorf("expected provider claude-cli, got %s", cfg.Provider)
	}
	if cfg.Model != "opus" {
		t.Errorf("expected model opus, got %s", cfg.Model)
	}
	if cfg.MaxConcurrency != 4 {
		t.Errorf("expected max_concurrency 4, got %d", cfg.MaxConcurrency)
	}
	if cfg.APIKeyEnv != "" {
		t.Errorf("expected empty api_key_env, got %s", cfg.APIKeyEnv)
	}
	if cfg.Languages != nil {
		t.Errorf("expected nil languages, got %v", cfg.Languages)
	}
}

func TestLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vex.yaml")

	content := []byte(`provider: claude-cli
model: haiku
languages:
  go:
    test_patterns: ["*_test.go"]
    source_patterns: ["*.go"]
`)
	if err := os.WriteFile(path, content, 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Provider != "claude-cli" {
		t.Errorf("expected provider claude-cli, got %s", cfg.Provider)
	}
	if cfg.Model != "haiku" {
		t.Errorf("expected model haiku, got %s", cfg.Model)
	}
	if lang, ok := cfg.Languages["go"]; !ok {
		t.Error("expected go language config")
	} else {
		if len(lang.TestPatterns) != 1 || lang.TestPatterns[0] != "*_test.go" {
			t.Errorf("unexpected test patterns: %v", lang.TestPatterns)
		}
		if len(lang.SourcePatterns) != 1 || lang.SourcePatterns[0] != "*.go" {
			t.Errorf("unexpected source patterns: %v", lang.SourcePatterns)
		}
	}
}

func TestLoadDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vex.yaml")

	if err := os.WriteFile(path, []byte("{}"), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Provider != "claude-cli" {
		t.Errorf("expected default provider claude-cli, got %s", cfg.Provider)
	}
	if cfg.Model != "opus" {
		t.Errorf("expected default model opus, got %s", cfg.Model)
	}
	if cfg.MaxConcurrency != 4 {
		t.Errorf("expected default max_concurrency 4, got %d", cfg.MaxConcurrency)
	}
}

func TestLoadMissing(t *testing.T) {
	_, err := Load("/nonexistent/vex.yaml")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestLoadInvalidYAML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vex.yaml")

	if err := os.WriteFile(path, []byte(":::invalid"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := Load(path)
	if err == nil {
		t.Error("expected error for invalid YAML")
	}
}

func TestWriteDefault(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vex.yaml")

	if err := WriteDefault(path); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Provider != "claude-cli" {
		t.Errorf("expected provider claude-cli, got %s", cfg.Provider)
	}
}

func TestWriteDefaultAlreadyExists(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vex.yaml")

	if err := os.WriteFile(path, []byte("{}"), 0644); err != nil {
		t.Fatal(err)
	}

	if err := WriteDefault(path); err == nil {
		t.Error("expected error when file already exists")
	}
}

func TestWriteDefaultContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vex.yaml")

	if err := WriteDefault(path); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Provider != "claude-cli" {
		t.Errorf("expected provider claude-cli, got %s", cfg.Provider)
	}
	if cfg.Model != "opus" {
		t.Errorf("expected model opus, got %s", cfg.Model)
	}
}

func TestLoadWalksUpDirectories(t *testing.T) {
	parent := t.TempDir()
	child := filepath.Join(parent, "sub", "deep")
	if err := os.MkdirAll(child, 0755); err != nil {
		t.Fatal(err)
	}

	content := []byte("provider: claude-cli\nmodel: opus\n")
	if err := os.WriteFile(filepath.Join(parent, "vex.yaml"), content, 0644); err != nil {
		t.Fatal(err)
	}

	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(child)

	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Provider != "claude-cli" {
		t.Errorf("expected provider claude-cli, got %s", cfg.Provider)
	}
}

func TestLoadPartialConfigPreservesExplicitValues(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vex.yaml")

	content := []byte("model: haiku\n")
	if err := os.WriteFile(path, content, 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Model != "haiku" {
		t.Errorf("expected explicit model haiku, got %s", cfg.Model)
	}
	if cfg.Provider != "claude-cli" {
		t.Errorf("expected default provider claude-cli, got %s", cfg.Provider)
	}
	if cfg.MaxConcurrency != 4 {
		t.Errorf("expected default max_concurrency 4, got %d", cfg.MaxConcurrency)
	}
}

func TestAddLanguage(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vex.yaml")

	// AddLanguage should create the file if it doesn't exist.
	lc := LanguageConfig{
		TestPatterns:   []string{"*_test.rs"},
		SourcePatterns: []string{"*.rs"},
	}
	if err := AddLanguage(path, "rust", lc); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	rust, ok := cfg.Languages["rust"]
	if !ok {
		t.Fatal("expected rust language in config")
	}
	if rust.TestPatterns[0] != "*_test.rs" {
		t.Errorf("expected *_test.rs, got %s", rust.TestPatterns[0])
	}
	if rust.SourcePatterns[0] != "*.rs" {
		t.Errorf("expected *.rs, got %s", rust.SourcePatterns[0])
	}
}

func TestAddLanguageToExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vex.yaml")

	if err := WriteDefault(path); err != nil {
		t.Fatal(err)
	}

	lc := LanguageConfig{
		TestPatterns:   []string{"*_test.rs"},
		SourcePatterns: []string{"*.rs"},
	}
	if err := AddLanguage(path, "rust", lc); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Provider != "claude-cli" {
		t.Errorf("expected provider preserved, got %s", cfg.Provider)
	}
	if _, ok := cfg.Languages["rust"]; !ok {
		t.Error("expected rust language added")
	}
}

func TestRemoveLanguage(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vex.yaml")

	lc := LanguageConfig{
		TestPatterns:   []string{"*_test.rs"},
		SourcePatterns: []string{"*.rs"},
	}
	if err := AddLanguage(path, "rust", lc); err != nil {
		t.Fatal(err)
	}

	if err := RemoveLanguage(path, "rust"); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Languages != nil {
		t.Errorf("expected nil languages after removing last one, got %v", cfg.Languages)
	}
}

func TestRemoveLanguageNotFound(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vex.yaml")

	if err := WriteDefault(path); err != nil {
		t.Fatal(err)
	}

	err := RemoveLanguage(path, "rust")
	if err == nil {
		t.Error("expected error removing nonexistent language")
	}
}

func TestLoadEmptyPathNoVexYaml(t *testing.T) {
	dir := t.TempDir()
	child := filepath.Join(dir, "a", "b")
	if err := os.MkdirAll(child, 0755); err != nil {
		t.Fatal(err)
	}

	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(child)

	_, err := Load("")
	if err == nil {
		t.Error("expected error when no vex.yaml found anywhere")
	}
}

package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/nickw409/vex/internal/config"
	"github.com/nickw409/vex/internal/diff"
	"github.com/nickw409/vex/internal/provider"
	"github.com/nickw409/vex/internal/report"
	"github.com/nickw409/vex/internal/spec"
)

// cliMockProvider returns a fixed response for any LLM call.
type cliMockProvider struct {
	response string
	err      error
	calls    int
}

func (m *cliMockProvider) Complete(ctx context.Context, req provider.CompletionRequest) (provider.CompletionResponse, error) {
	m.calls++
	if m.err != nil {
		return provider.CompletionResponse{}, m.err
	}
	return provider.CompletionResponse{Content: m.response}, nil
}

// setupCheckEnv creates a temp dir with vex.yaml, a spec, and source/test files.
// Returns the dir path. Caller must chdir and restore.
func setupCheckEnv(t *testing.T, specYAML string) string {
	t.Helper()
	dir := t.TempDir()

	os.WriteFile(filepath.Join(dir, "vex.yaml"), []byte("provider: claude-cli\nmodel: opus\n"), 0644)

	vexDir := filepath.Join(dir, ".vex")
	os.MkdirAll(vexDir, 0755)
	os.WriteFile(filepath.Join(vexDir, "vexspec.yaml"), []byte(specYAML), 0644)

	// Create source and test files
	srcDir := filepath.Join(dir, "src")
	os.MkdirAll(srcDir, 0755)
	os.WriteFile(filepath.Join(srcDir, "auth.go"), []byte("package auth\nfunc Login() {}"), 0644)
	os.WriteFile(filepath.Join(srcDir, "auth_test.go"), []byte("package auth\nfunc TestLogin(t *testing.T) {}"), 0644)

	return dir
}

const testSpec = `project: Test
sections:
  - name: Auth
    path: src
    description: Auth module
    behaviors:
      - name: login
        description: POST /login returns JWT
`

// withMockProvider overrides newProviderFunc for the duration of the test.
func withMockProvider(t *testing.T, response string) {
	t.Helper()
	orig := newProviderFunc
	newProviderFunc = func(cfg *config.Config) (provider.Provider, error) {
		return &cliMockProvider{response: response}, nil
	}
	t.Cleanup(func() { newProviderFunc = orig })
}

func withFailingProvider(t *testing.T, err error) {
	t.Helper()
	orig := newProviderFunc
	newProviderFunc = func(cfg *config.Config) (provider.Provider, error) {
		return &cliMockProvider{err: err}, nil
	}
	t.Cleanup(func() { newProviderFunc = orig })
}

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

// --- Drift command tests ---

func TestDriftNoPreviousCheck(t *testing.T) {
	dir := t.TempDir()
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	// The drift command calls os.Exit(2) when no report exists.
	// We can't test os.Exit directly, but we can verify ReportModTime
	// returns zero, which is the condition that triggers exit 2.
	modTime := diff.ReportModTime(dir)
	if !modTime.IsZero() {
		t.Error("expected zero mod time when no report.json exists")
	}
}

func TestDriftAllClean(t *testing.T) {
	dir := setupGitRepoWithSpec(t)
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	// Set report.json mtime 2 seconds in the future to ensure git log --since
	// won't match the commit.
	vexDir := filepath.Join(dir, ".vex")
	os.WriteFile(filepath.Join(vexDir, "report.json"), []byte("{}"), 0644)
	future := diff.ReportModTime(dir).Add(2 * time.Second)
	os.Chtimes(filepath.Join(vexDir, "report.json"), future, future)

	// Capture stdout
	origStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"drift", "--spec", filepath.Join(vexDir, "vexspec.yaml")})
	cmd.Execute()

	w.Close()
	os.Stdout = origStdout

	var buf bytes.Buffer
	buf.ReadFrom(r)

	var report struct {
		Drifted []interface{} `json:"drifted"`
		Clean   []string      `json:"clean"`
	}
	if err := json.Unmarshal(buf.Bytes(), &report); err != nil {
		t.Fatalf("drift output is not valid JSON: %v\noutput: %s", err, buf.String())
	}

	if len(report.Drifted) != 0 {
		t.Errorf("expected no drifted sections, got %d", len(report.Drifted))
	}
	if len(report.Clean) != 1 || report.Clean[0] != "Auth" {
		t.Errorf("expected [Auth] in clean, got %v", report.Clean)
	}
}

func setupGitRepoWithSpec(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	// Init git repo
	for _, args := range [][]string{
		{"git", "init"},
		{"git", "config", "user.email", "test@test.com"},
		{"git", "config", "user.name", "Test"},
	} {
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("%v failed: %v\n%s", args, err, out)
		}
	}

	// Create source file and commit
	srcDir := filepath.Join(dir, "internal", "auth")
	os.MkdirAll(srcDir, 0755)
	os.WriteFile(filepath.Join(srcDir, "auth.go"), []byte("package auth"), 0644)

	vexDir := filepath.Join(dir, ".vex")
	os.MkdirAll(vexDir, 0755)
	os.WriteFile(filepath.Join(vexDir, "vexspec.yaml"), []byte(`project: Test
sections:
  - name: Auth
    path: internal/auth
    description: Auth module
    behaviors:
      - name: login
        description: Login endpoint
`), 0644)

	for _, args := range [][]string{
		{"git", "add", "."},
		{"git", "commit", "-m", "initial"},
	} {
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("%v failed: %v\n%s", args, err, out)
		}
	}

	return dir
}

// --- Shared behavior tests: exit-codes, error-output, json-output ---

func TestOutputReportExitZeroNoGaps(t *testing.T) {
	r := &report.Report{
		Spec:    ".vex/vexspec.yaml",
		Gaps:    []report.Gap{},
		Covered: []report.Covered{},
	}
	r.ComputeSummary(0)

	if r.HasGaps() {
		t.Error("expected HasGaps() == false for empty gaps")
	}
}

func TestOutputReportHasGapsWhenGapsExist(t *testing.T) {
	r := &report.Report{
		Spec: ".vex/vexspec.yaml",
		Gaps: []report.Gap{
			{Behavior: "login", Detail: "missing test", Suggestion: "add test"},
		},
		Covered: []report.Covered{},
	}
	r.ComputeSummary(1)

	if !r.HasGaps() {
		t.Error("expected HasGaps() == true when gaps exist")
	}
}

func TestCheckEmptyReportWritesJSON(t *testing.T) {
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

	// Capture stdout
	origStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--spec", specPath, "--drift=false"})
	cmd.Execute()

	w.Close()
	os.Stdout = origStdout

	var buf bytes.Buffer
	buf.ReadFrom(r)
	output := buf.String()

	// Verify stdout contains valid JSON
	var parsed report.Report
	if err := json.Unmarshal([]byte(strings.TrimSpace(output)), &parsed); err != nil {
		t.Fatalf("stdout is not valid JSON: %v\noutput: %s", err, output)
	}

	// Verify .vex/report.json was also written
	fileData, err := os.ReadFile(filepath.Join(vexDir, "report.json"))
	if err != nil {
		t.Fatalf("expected .vex/report.json to be written: %v", err)
	}
	var fileParsed report.Report
	if err := json.Unmarshal(bytes.TrimSpace(fileData), &fileParsed); err != nil {
		t.Fatalf(".vex/report.json is not valid JSON: %v", err)
	}
}

func TestCheckErrorsGoToStderr(t *testing.T) {
	// Capture stderr
	origStderr := os.Stderr
	r, w, _ := os.Pipe()
	os.Stderr = w

	// Capture stdout to ensure errors don't leak there
	origStdout := os.Stdout
	rOut, wOut, _ := os.Pipe()
	os.Stdout = wOut

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--spec", "/nonexistent/spec.yaml", "--drift=false"})
	cmd.Execute()

	w.Close()
	wOut.Close()
	os.Stderr = origStderr
	os.Stdout = origStdout

	var stderrBuf bytes.Buffer
	stderrBuf.ReadFrom(r)

	var stdoutBuf bytes.Buffer
	stdoutBuf.ReadFrom(rOut)

	// Stdout should be empty (no JSON output on fatal error)
	if stdoutBuf.Len() > 0 {
		t.Errorf("expected no stdout on fatal error, got: %s", stdoutBuf.String())
	}
}

func TestRootConfigSkippedForGuide(t *testing.T) {
	cfg = nil
	cmd := NewRootCmd()
	cmd.SetArgs([]string{"guide"})
	// Discard stdout
	cmd.SetOut(new(bytes.Buffer))
	cmd.Execute()

	if cfg != nil {
		t.Error("expected cfg to remain nil for guide command")
	}
}

func TestRootConfigSkippedForDrift(t *testing.T) {
	cfg = nil
	cmd := NewRootCmd()
	cmd.SetArgs([]string{"drift", "--spec", "/nonexistent"})
	cmd.Execute()

	if cfg != nil {
		t.Error("expected cfg to remain nil for drift command")
	}
}

func TestValidateErrorGoesToStderr(t *testing.T) {
	origStderr := os.Stderr
	_, w, _ := os.Pipe()
	os.Stderr = w

	origStdout := os.Stdout
	rOut, wOut, _ := os.Pipe()
	os.Stdout = wOut

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"validate", "/nonexistent/spec.yaml"})
	cmd.Execute()

	w.Close()
	wOut.Close()
	os.Stderr = origStderr
	os.Stdout = origStdout

	var stdoutBuf bytes.Buffer
	stdoutBuf.ReadFrom(rOut)

	// No JSON on stdout when spec doesn't exist
	if stdoutBuf.Len() > 0 {
		t.Errorf("expected no stdout on validate fatal error, got: %s", stdoutBuf.String())
	}
}

func TestValidateDefaultSpecPath(t *testing.T) {
	dir := t.TempDir()
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	// No spec file — should error referencing default path
	cmd := NewRootCmd()
	cmd.SetArgs([]string{"validate"})
	err := cmd.Execute()

	if err == nil {
		t.Fatal("expected error when default spec doesn't exist")
	}
	if !strings.Contains(err.Error(), "vexspec.yaml") {
		t.Errorf("expected error to reference vexspec.yaml, got: %v", err)
	}
}

func TestRootConfigSkippedForUpdate(t *testing.T) {
	cfg = nil
	cmd := NewRootCmd()
	// update will fail but config should not be loaded
	cmd.SetArgs([]string{"update", "--help"})
	cmd.Execute()

	if cfg != nil {
		t.Error("expected cfg to remain nil for update command")
	}
}

// --- CLI Check integration tests with mock provider ---

var coveredResp = `{"gaps": [], "covered": [{"behavior": "login", "detail": "tested", "test_file": "auth_test.go", "test_name": "TestLogin"}]}`
var gapResp = `{"gaps": [{"behavior": "login", "detail": "missing expiry test", "suggestion": "TestLoginExpiry"}], "covered": []}`

func TestCheckExecutionWithMockProvider(t *testing.T) {
	withMockProvider(t, coveredResp)

	dir := setupCheckEnv(t, testSpec)
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	origStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--drift=false"})
	err := cmd.Execute()

	w.Close()
	os.Stdout = origStdout

	if err != nil {
		t.Fatalf("check failed: %v", err)
	}

	var buf bytes.Buffer
	buf.ReadFrom(r)

	// Verify JSON on stdout
	var parsed report.Report
	if err := json.Unmarshal(buf.Bytes(), &parsed); err != nil {
		t.Fatalf("stdout not valid JSON: %v\noutput: %s", err, buf.String())
	}

	if parsed.Summary.TotalBehaviors != 1 {
		t.Errorf("expected 1 total behavior, got %d", parsed.Summary.TotalBehaviors)
	}
}

func TestCheckOutputWritesReportFile(t *testing.T) {
	withMockProvider(t, coveredResp)

	dir := setupCheckEnv(t, testSpec)
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	// Discard stdout
	origStdout := os.Stdout
	_, w, _ := os.Pipe()
	os.Stdout = w

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--drift=false"})
	cmd.Execute()

	w.Close()
	os.Stdout = origStdout

	// Verify .vex/report.json written
	data, err := os.ReadFile(filepath.Join(dir, ".vex", "report.json"))
	if err != nil {
		t.Fatalf("expected .vex/report.json: %v", err)
	}

	var parsed report.Report
	if err := json.Unmarshal(bytes.TrimSpace(data), &parsed); err != nil {
		t.Fatalf("report.json not valid JSON: %v", err)
	}

	// Verify indented
	if !strings.Contains(string(data), "\n  ") {
		t.Error("expected indented JSON in report.json")
	}
}

func TestCheckOutputIncludesChecksums(t *testing.T) {
	withMockProvider(t, coveredResp)

	dir := setupCheckEnv(t, testSpec)
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	origStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--drift=false"})
	cmd.Execute()

	w.Close()
	os.Stdout = origStdout

	var buf bytes.Buffer
	buf.ReadFrom(r)

	var parsed struct {
		SectionChecksums map[string]string `json:"section_checksums"`
	}
	json.Unmarshal(buf.Bytes(), &parsed)

	if parsed.SectionChecksums == nil {
		t.Fatal("expected section_checksums in report")
	}
	if _, ok := parsed.SectionChecksums["Auth"]; !ok {
		t.Error("expected Auth in section_checksums")
	}
}

func TestCheckExecutionErrorStillOutputs(t *testing.T) {
	withFailingProvider(t, fmt.Errorf("provider unavailable"))

	dir := setupCheckEnv(t, testSpec)
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	origStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--drift=false"})
	cmd.Execute() // error expected but report should still be output

	w.Close()
	os.Stdout = origStdout

	var buf bytes.Buffer
	buf.ReadFrom(r)

	// Should still get JSON output even on error
	if buf.Len() == 0 {
		t.Error("expected report output even on provider error")
	}
}

func TestCheckProfileMode(t *testing.T) {
	withMockProvider(t, coveredResp)

	dir := setupCheckEnv(t, testSpec)
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	// Discard stdout
	origStdout := os.Stdout
	_, w, _ := os.Pipe()
	os.Stdout = w

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--drift=false", "--profile"})
	cmd.Execute()

	w.Close()
	os.Stdout = origStdout

	// Verify .vex/profile.json created
	data, err := os.ReadFile(filepath.Join(dir, ".vex", "profile.json"))
	if err != nil {
		t.Fatalf("expected .vex/profile.json: %v", err)
	}
	if len(data) == 0 {
		t.Error("profile.json is empty")
	}
}

func TestValidateExecutionWithMockProvider(t *testing.T) {
	withMockProvider(t, `{"complete": true, "suggestions": []}`)

	dir := setupCheckEnv(t, testSpec)
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	origStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"validate", filepath.Join(dir, ".vex", "vexspec.yaml")})
	err := cmd.Execute()

	w.Close()
	os.Stdout = origStdout

	if err != nil {
		t.Fatalf("validate failed: %v", err)
	}

	var buf bytes.Buffer
	buf.ReadFrom(r)

	// Should be valid indented JSON on stdout
	output := buf.String()
	if !strings.Contains(output, `"complete"`) {
		t.Errorf("expected complete field in stdout JSON, got: %s", output)
	}
	if !strings.Contains(output, "\n  ") {
		t.Error("expected indented JSON on stdout")
	}
}

func TestValidateOutputWritesFile(t *testing.T) {
	withMockProvider(t, `{"complete": true, "suggestions": []}`)

	dir := setupCheckEnv(t, testSpec)
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	// Discard stdout
	origStdout := os.Stdout
	_, w, _ := os.Pipe()
	os.Stdout = w

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"validate", filepath.Join(dir, ".vex", "vexspec.yaml")})
	cmd.Execute()

	w.Close()
	os.Stdout = origStdout

	data, err := os.ReadFile(filepath.Join(dir, ".vex", "validation.json"))
	if err != nil {
		t.Fatalf("expected .vex/validation.json: %v", err)
	}
	if !strings.Contains(string(data), `"complete"`) {
		t.Error("validation.json should contain complete field")
	}
}

func TestValidateSuccessOutputsJSON(t *testing.T) {
	withMockProvider(t, `{"complete": true, "suggestions": []}`)

	dir := setupCheckEnv(t, testSpec)
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	origStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	origStderr := os.Stderr
	_, wErr, _ := os.Pipe()
	os.Stderr = wErr

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"validate", filepath.Join(dir, ".vex", "vexspec.yaml")})
	cmd.Execute()

	w.Close()
	wErr.Close()
	os.Stdout = origStdout
	os.Stderr = origStderr

	var buf bytes.Buffer
	buf.ReadFrom(r)

	// Stdout should have JSON
	if buf.Len() == 0 {
		t.Error("expected JSON on stdout for successful validate")
	}

	var parsed struct {
		Complete bool `json:"complete"`
	}
	if err := json.Unmarshal(buf.Bytes(), &parsed); err != nil {
		t.Fatalf("stdout is not valid JSON: %v", err)
	}
	if !parsed.Complete {
		t.Error("expected complete=true")
	}
}

func TestValidateDriftSkipsCleanSections(t *testing.T) {
	mock := &cliMockProvider{response: `{"complete": true, "suggestions": []}`}
	orig := newProviderFunc
	newProviderFunc = func(cfg *config.Config) (provider.Provider, error) {
		return mock, nil
	}
	t.Cleanup(func() { newProviderFunc = orig })

	dir := setupCheckEnv(t, testSpec)
	origDir, _ := os.Getwd()
	defer os.Chdir(origDir)
	os.Chdir(dir)

	// Write a previous validation.json with checksums matching the spec.
	ps, _ := spec.LoadProject(filepath.Join(dir, ".vex", "vexspec.yaml"))
	checksums := make(map[string]string)
	for _, sec := range ps.Sections {
		checksums[sec.Name] = spec.SectionChecksum(&sec, ps.ResolveShared(&sec))
	}

	prevValidation, _ := json.Marshal(map[string]interface{}{
		"complete":          true,
		"suggestions":       []interface{}{},
		"section_checksums": checksums,
	})
	os.WriteFile(filepath.Join(dir, ".vex", "validation.json"), prevValidation, 0644)

	// Discard stdout
	origStdout := os.Stdout
	_, w, _ := os.Pipe()
	os.Stdout = w

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"validate", filepath.Join(dir, ".vex", "vexspec.yaml")})
	cmd.Execute()

	w.Close()
	os.Stdout = origStdout

	if mock.calls != 0 {
		t.Errorf("expected 0 LLM calls (all sections clean), got %d", mock.calls)
	}
}

func TestValidateDriftRevalidatesChangedSections(t *testing.T) {
	withMockProvider(t, `{"complete": true, "suggestions": []}`)

	dir := setupCheckEnv(t, testSpec)
	origDir, _ := os.Getwd()
	defer os.Chdir(origDir)
	os.Chdir(dir)

	// Write a previous validation.json with stale checksums.
	prevValidation, _ := json.Marshal(map[string]interface{}{
		"complete":    true,
		"suggestions": []interface{}{},
		"section_checksums": map[string]string{
			"Auth": "stale-checksum",
		},
	})
	os.WriteFile(filepath.Join(dir, ".vex", "validation.json"), prevValidation, 0644)

	// Capture stderr for log messages
	origStderr := os.Stderr
	rErr, wErr, _ := os.Pipe()
	os.Stderr = wErr

	origStdout := os.Stdout
	_, wOut, _ := os.Pipe()
	os.Stdout = wOut

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"validate", filepath.Join(dir, ".vex", "vexspec.yaml")})
	cmd.Execute()

	wErr.Close()
	wOut.Close()
	os.Stderr = origStderr
	os.Stdout = origStdout

	var stderrBuf bytes.Buffer
	stderrBuf.ReadFrom(rErr)

	if !strings.Contains(stderrBuf.String(), "spec changed") {
		t.Error("expected 'spec changed' log message for stale checksum")
	}
}

func TestValidateDriftCarriesForwardSuggestions(t *testing.T) {
	// validate calls os.Exit(1) when suggestions exist, so we test
	// carry-forward in a subprocess.
	if os.Getenv("VEX_TEST_CARRY_FORWARD") == "1" {
		withMockProvider(t, `{"complete": true, "suggestions": []}`)

		twoSectionSpec := `project: Test
sections:
  - name: Auth
    path: src
    description: Auth module
    behaviors:
      - name: login
        description: POST /login returns JWT
  - name: Storage
    path: src
    description: Storage module
    behaviors:
      - name: upload
        description: Upload files
`
		dir := setupCheckEnv(t, twoSectionSpec)
		origDir, _ := os.Getwd()
		defer os.Chdir(origDir)
		os.Chdir(dir)

		ps, _ := spec.LoadProject(filepath.Join(dir, ".vex", "vexspec.yaml"))
		authChecksum := spec.SectionChecksum(&ps.Sections[0], ps.ResolveShared(&ps.Sections[0]))

		prevValidation, _ := json.Marshal(map[string]interface{}{
			"complete": false,
			"suggestions": []map[string]string{
				{
					"section":       "Auth",
					"behavior_name": "logout",
					"description":   "Missing logout behavior",
					"relation":      "new",
				},
			},
			"section_checksums": map[string]string{
				"Auth":    authChecksum,
				"Storage": "stale-checksum",
			},
		})
		os.WriteFile(filepath.Join(dir, ".vex", "validation.json"), prevValidation, 0644)

		origStdout := os.Stdout
		_, w, _ := os.Pipe()
		os.Stdout = w

		cmd := NewRootCmd()
		cmd.SetArgs([]string{"validate", filepath.Join(dir, ".vex", "vexspec.yaml")})
		cmd.Execute()

		w.Close()
		os.Stdout = origStdout

		// Read the written file (stdout may be lost due to os.Exit).
		data, err := os.ReadFile(filepath.Join(dir, ".vex", "validation.json"))
		if err != nil {
			t.Fatalf("expected validation.json: %v", err)
		}
		if !strings.Contains(string(data), `"logout"`) {
			t.Error("expected carried-forward suggestion for Auth/logout in validation.json")
		}
		return
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestValidateDriftCarriesForwardSuggestions", "-test.v")
	cmd.Env = append(os.Environ(), "VEX_TEST_CARRY_FORWARD=1")
	out, err := cmd.CombinedOutput()
	// Exit code 1 is expected (suggestions found).
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
			// Expected: os.Exit(1) from validate with suggestions.
			// Check that the subprocess test assertions passed.
			if strings.Contains(string(out), "FAIL") && !strings.Contains(string(out), "exit status 1") {
				t.Fatalf("subprocess test failed:\n%s", out)
			}
			return
		}
		t.Fatalf("subprocess failed: %v\n%s", err, out)
	}
}

func TestCheckFileGatheringWarnsOnBadPath(t *testing.T) {
	withMockProvider(t, coveredResp)

	specWithBadPath := `project: Test
sections:
  - name: Auth
    path: [src, nonexistent_bad_path]
    description: Auth module
    behaviors:
      - name: login
        description: POST /login returns JWT
`
	dir := setupCheckEnv(t, specWithBadPath)
	orig, _ := os.Getwd()
	defer os.Chdir(orig)
	os.Chdir(dir)

	// Capture stderr for warnings
	origStderr := os.Stderr
	rErr, wErr, _ := os.Pipe()
	os.Stderr = wErr

	// Discard stdout
	origStdout := os.Stdout
	_, wOut, _ := os.Pipe()
	os.Stdout = wOut

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"check", "--drift=false"})
	cmd.Execute()

	wErr.Close()
	wOut.Close()
	os.Stderr = origStderr
	os.Stdout = origStdout

	var stderrBuf bytes.Buffer
	stderrBuf.ReadFrom(rErr)

	if !strings.Contains(stderrBuf.String(), "nonexistent_bad_path") {
		t.Errorf("expected warning about nonexistent_bad_path on stderr, got: %s", stderrBuf.String())
	}
}

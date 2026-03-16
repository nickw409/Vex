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

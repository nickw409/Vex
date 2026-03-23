package spec

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeSpec(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "vexspec.yaml")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadProject(t *testing.T) {
	path := writeSpec(t, `project: MyApp
description: Test project
sections:
  - name: Auth
    path: internal/auth
    description: Authentication module
    behaviors:
      - name: login
        description: POST /login returns JWT
      - name: logout
        description: POST /logout invalidates session
`)

	ps, err := LoadProject(path)
	if err != nil {
		t.Fatal(err)
	}

	if ps.Project != "MyApp" {
		t.Errorf("expected project 'MyApp', got %q", ps.Project)
	}
	if len(ps.Sections) != 1 {
		t.Errorf("expected 1 section, got %d", len(ps.Sections))
	}
	if ps.Sections[0].Name != "Auth" {
		t.Errorf("expected section name 'Auth', got %q", ps.Sections[0].Name)
	}
	if len(ps.Sections[0].Behaviors) != 2 {
		t.Errorf("expected 2 behaviors, got %d", len(ps.Sections[0].Behaviors))
	}
}

func TestLoadProjectWithShared(t *testing.T) {
	path := writeSpec(t, `project: MyApp
shared:
  - name: error-handling
    description: All errors return structured JSON
sections:
  - name: Auth
    path: internal/auth
    description: Auth module
    shared: [error-handling]
    behaviors:
      - name: login
        description: Login endpoint
`)

	ps, err := LoadProject(path)
	if err != nil {
		t.Fatal(err)
	}

	all := ps.AllBehaviors(&ps.Sections[0])
	if len(all) != 2 {
		t.Errorf("expected 2 behaviors (1 shared + 1 own), got %d", len(all))
	}
	if all[0].Name != "error-handling" {
		t.Errorf("expected shared behavior first, got %q", all[0].Name)
	}
}

func TestLoadProjectWithSubsections(t *testing.T) {
	path := writeSpec(t, `project: MyApp
sections:
  - name: App Server
    path: app
    description: Main server
    behaviors:
      - name: websocket
        description: WebSocket handling
    subsections:
      - name: Auth Handlers
        file: app/handlers/auth.go
        behaviors:
          - name: login
            description: Login handler
      - name: Analysis
        path: [app/handlers/analysis/, app/db/queries/]
        behaviors:
          - name: result-queries
            description: Query results
`)

	ps, err := LoadProject(path)
	if err != nil {
		t.Fatal(err)
	}

	sec := &ps.Sections[0]
	all := ps.AllBehaviors(sec)
	if len(all) != 3 {
		t.Errorf("expected 3 behaviors (1 section + 2 subsection), got %d", len(all))
	}

	paths := SectionPaths(sec)
	// app (section path), app/handlers/analysis/, app/db/queries/ (subsection paths)
	if len(paths) != 3 {
		t.Errorf("expected 3 paths, got %d: %v", len(paths), paths)
	}

	files := SectionFiles(sec)
	// app/handlers/auth.go (subsection file)
	if len(files) != 1 || files[0] != "app/handlers/auth.go" {
		t.Errorf("expected [app/handlers/auth.go], got %v", files)
	}

	allPaths := SectionAllPaths(sec)
	// app, app/handlers (from file dir), app/handlers/analysis/, app/db/queries/
	if len(allPaths) != 4 {
		t.Errorf("expected 4 allPaths, got %d: %v", len(allPaths), allPaths)
	}
}

func TestLoadProjectPathAsString(t *testing.T) {
	path := writeSpec(t, `project: MyApp
sections:
  - name: Core
    path: internal/core
    description: Core module
    behaviors:
      - name: process
        description: Process data
`)

	ps, err := LoadProject(path)
	if err != nil {
		t.Fatal(err)
	}

	if len(ps.Sections[0].Path) != 1 || ps.Sections[0].Path[0] != "internal/core" {
		t.Errorf("expected path ['internal/core'], got %v", ps.Sections[0].Path)
	}
}

func TestLoadProjectPathAsList(t *testing.T) {
	path := writeSpec(t, `project: MyApp
sections:
  - name: Auth
    path: [src/handlers/auth.go, src/auth/]
    description: Auth module
    behaviors:
      - name: login
        description: Login
`)

	ps, err := LoadProject(path)
	if err != nil {
		t.Fatal(err)
	}

	if len(ps.Sections[0].Path) != 2 {
		t.Errorf("expected 2 paths, got %d", len(ps.Sections[0].Path))
	}
}

func TestLoadProjectMissingProject(t *testing.T) {
	path := writeSpec(t, `sections:
  - name: Auth
    description: Auth
    behaviors:
      - name: login
        description: Login
`)

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error for missing project name")
	}
}

func TestLoadProjectNoSections(t *testing.T) {
	path := writeSpec(t, `project: MyApp
sections: []
`)

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error for empty sections")
	}
}

func TestLoadProjectUnknownSharedRef(t *testing.T) {
	path := writeSpec(t, `project: MyApp
sections:
  - name: Auth
    description: Auth
    shared: [nonexistent]
    behaviors:
      - name: login
        description: Login
`)

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error for unknown shared reference")
	}
}

func TestLoadProjectSubsectionPathAndFile(t *testing.T) {
	path := writeSpec(t, `project: MyApp
sections:
  - name: Auth
    description: Auth
    behaviors:
      - name: login
        description: Login
    subsections:
      - name: Bad
        path: some/dir
        file: some/file.go
        behaviors:
          - name: thing
            description: Thing
`)

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error when subsection has both path and file")
	}
}

func TestLoadProjectBehaviorMissingName(t *testing.T) {
	path := writeSpec(t, `project: MyApp
sections:
  - name: Auth
    description: Auth
    behaviors:
      - description: Login
`)

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error for behavior missing name")
	}
}

func TestLoadProjectBehaviorMissingDescription(t *testing.T) {
	path := writeSpec(t, `project: MyApp
sections:
  - name: Auth
    description: Auth
    behaviors:
      - name: login
`)

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error for behavior missing description")
	}
}

func TestLoadProjectInvalidYAML(t *testing.T) {
	path := writeSpec(t, ":::invalid yaml\n\t{{{")

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error for invalid YAML")
	}
}

func TestLoadProjectNonexistentFile(t *testing.T) {
	_, err := LoadProject("/nonexistent/spec.yaml")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestLoadProjectDefaultPath(t *testing.T) {
	// LoadProject("") should look for .vex/vexspec.yaml
	_, err := LoadProject("")
	if err == nil {
		t.Error("expected error since .vex/vexspec.yaml doesn't exist in test dir")
	}
}

func TestLoadProjectDuplicateSharedBehaviorNames(t *testing.T) {
	path := writeSpec(t, `project: MyApp
shared:
  - name: error-handling
    description: Structured errors
  - name: error-handling
    description: Duplicate shared behavior
sections:
  - name: Auth
    description: Auth
    behaviors:
      - name: login
        description: Login
`)

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error for duplicate shared behavior names")
	}
	if err != nil && !strings.Contains(err.Error(), "duplicate") {
		t.Errorf("expected error to mention 'duplicate', got: %s", err.Error())
	}
}

func TestLoadProjectSharedBehaviorMissingDescription(t *testing.T) {
	path := writeSpec(t, `project: MyApp
shared:
  - name: error-handling
sections:
  - name: Auth
    description: Auth
    behaviors:
      - name: login
        description: Login
`)

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error for shared behavior missing description")
	}
}

func TestLoadProjectSharedBehaviorMissingName(t *testing.T) {
	path := writeSpec(t, `project: MyApp
shared:
  - description: Structured errors
sections:
  - name: Auth
    description: Auth
    behaviors:
      - name: login
        description: Login
`)

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error for shared behavior missing name")
	}
}

func TestLoadProjectSubsectionMissingName(t *testing.T) {
	path := writeSpec(t, `project: MyApp
sections:
  - name: Auth
    description: Auth
    behaviors:
      - name: login
        description: Login
    subsections:
      - path: some/dir
        behaviors:
          - name: thing
            description: Thing
`)

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error for subsection missing name")
	}
}

func TestLoadProjectSectionMissingName(t *testing.T) {
	path := writeSpec(t, `project: MyApp
sections:
  - description: Auth module
    behaviors:
      - name: login
        description: Login
`)

	_, err := LoadProject(path)
	if err == nil {
		t.Error("expected error for section missing name")
	}
}

func TestSectionPathsDeduplication(t *testing.T) {
	sec := &Section{
		Path: PathList{"internal/auth", "internal/core"},
		Subsections: []Subsection{
			{
				Name: "Sub1",
				Path: PathList{"internal/auth"},
			},
			{
				Name: "Sub2",
				Path: PathList{"internal/core", "internal/api"},
			},
		},
	}

	paths := SectionPaths(sec)
	// internal/auth, internal/core, internal/api — no duplicates
	if len(paths) != 3 {
		t.Errorf("expected 3 deduplicated paths, got %d: %v", len(paths), paths)
	}
	expected := []string{"internal/auth", "internal/core", "internal/api"}
	for i, want := range expected {
		if i >= len(paths) || paths[i] != want {
			t.Errorf("expected paths[%d]=%q, got %v", i, want, paths)
		}
	}
}

package lang

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/nickw409/vex/internal/config"
)

func TestDetectGo(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "main.go")
	touch(t, dir, "main_test.go")
	touch(t, dir, "handler.go")

	l, err := Detect(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if l.Name != "go" {
		t.Errorf("expected go, got %s", l.Name)
	}
}

func TestDetectPython(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "app.py")
	touch(t, dir, "test_app.py")

	l, err := Detect(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if l.Name != "python" {
		t.Errorf("expected python, got %s", l.Name)
	}
}

func TestDetectWithOverride(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "main.go")

	overrides := map[string]config.LanguageConfig{
		"custom": {
			TestPatterns:   []string{"*_spec.go"},
			SourcePatterns: []string{"*.go"},
		},
	}

	l, err := Detect(dir, overrides)
	if err != nil {
		t.Fatal(err)
	}
	if l.Name != "custom" {
		t.Errorf("expected custom, got %s", l.Name)
	}
	if l.TestPatterns[0] != "*_spec.go" {
		t.Errorf("expected *_spec.go pattern, got %s", l.TestPatterns[0])
	}
}

func TestDetectEmpty(t *testing.T) {
	dir := t.TempDir()
	_, err := Detect(dir, nil)
	if err == nil {
		t.Error("expected error for empty directory")
	}
}

func TestFindFiles(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "main.go")
	touch(t, dir, "handler.go")
	touch(t, dir, "main_test.go")
	touch(t, dir, "handler_test.go")
	touch(t, dir, "README.md")

	l := &Language{
		Name:           "go",
		TestPatterns:   []string{"*_test.go"},
		SourcePatterns: []string{"*.go"},
	}

	src, tests, err := FindFiles(dir, l)
	if err != nil {
		t.Fatal(err)
	}

	if len(src) != 2 {
		t.Errorf("expected 2 source files, got %d", len(src))
	}
	if len(tests) != 2 {
		t.Errorf("expected 2 test files, got %d", len(tests))
	}
}

func TestFindFilesSkipsVendor(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "main.go")

	vendorDir := filepath.Join(dir, "vendor")
	os.MkdirAll(vendorDir, 0755)
	touch(t, vendorDir, "dep.go")

	l := &Language{
		Name:           "go",
		TestPatterns:   []string{"*_test.go"},
		SourcePatterns: []string{"*.go"},
	}

	src, _, err := FindFiles(dir, l)
	if err != nil {
		t.Fatal(err)
	}

	if len(src) != 1 {
		t.Errorf("expected 1 source file (vendor excluded), got %d", len(src))
	}
}

func TestDetectTypeScript(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "app.ts")
	touch(t, dir, "app.test.ts")
	touch(t, dir, "utils.ts")

	l, err := Detect(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if l.Name != "typescript" {
		t.Errorf("expected typescript, got %s", l.Name)
	}
}

func TestDetectJava(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "App.java")
	touch(t, dir, "AppTest.java")

	l, err := Detect(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if l.Name != "java" {
		t.Errorf("expected java, got %s", l.Name)
	}
}

func TestDetectMultipleLanguagesPicksMostFiles(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "main.go")
	touch(t, dir, "handler.go")
	touch(t, dir, "utils.go")
	touch(t, dir, "script.py")

	l, err := Detect(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if l.Name != "go" {
		t.Errorf("expected go (most files), got %s", l.Name)
	}
}

func TestDetectSkipsNodeModules(t *testing.T) {
	dir := t.TempDir()
	nmDir := filepath.Join(dir, "node_modules")
	os.MkdirAll(nmDir, 0755)
	touch(t, nmDir, "dep.js")

	_, err := Detect(dir, nil)
	if err == nil {
		t.Error("expected error when files only in node_modules")
	}
}

func TestDetectSkipsGitDir(t *testing.T) {
	dir := t.TempDir()
	gitDir := filepath.Join(dir, ".git")
	os.MkdirAll(gitDir, 0755)
	touch(t, gitDir, "config.py")

	_, err := Detect(dir, nil)
	if err == nil {
		t.Error("expected error when files only in .git")
	}
}

func TestFindFilesSkipsNodeModules(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "main.go")

	nmDir := filepath.Join(dir, "node_modules")
	os.MkdirAll(nmDir, 0755)
	touch(t, nmDir, "dep.go")

	l := &Language{
		Name:           "go",
		TestPatterns:   []string{"*_test.go"},
		SourcePatterns: []string{"*.go"},
	}

	src, _, err := FindFiles(dir, l)
	if err != nil {
		t.Fatal(err)
	}
	if len(src) != 1 {
		t.Errorf("expected 1 source file (node_modules excluded), got %d", len(src))
	}
}

func TestFindFilesSkipsGitDir(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "main.go")

	gitDir := filepath.Join(dir, ".git")
	os.MkdirAll(gitDir, 0755)
	touch(t, gitDir, "hook.go")

	l := &Language{
		Name:           "go",
		TestPatterns:   []string{"*_test.go"},
		SourcePatterns: []string{"*.go"},
	}

	src, _, err := FindFiles(dir, l)
	if err != nil {
		t.Fatal(err)
	}
	if len(src) != 1 {
		t.Errorf("expected 1 source file (.git excluded), got %d", len(src))
	}
}

func TestDetectJavaScriptWithPackageJSON(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "app.js")
	touch(t, dir, "app.test.js")
	touch(t, dir, "package.json")

	l, err := Detect(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if l.Name != "javascript" {
		t.Errorf("expected javascript, got %s", l.Name)
	}
}

func TestDetectJavaScriptWithoutPackageJSON(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "app.js")
	touch(t, dir, "utils.js")

	_, err := Detect(dir, nil)
	if err == nil {
		t.Error("expected error when .js files present but no package.json")
	}
}

func TestDetectSkipsVendor(t *testing.T) {
	dir := t.TempDir()
	vendorDir := filepath.Join(dir, "vendor")
	os.MkdirAll(vendorDir, 0755)
	touch(t, vendorDir, "dep.go")

	_, err := Detect(dir, nil)
	if err == nil {
		t.Error("expected error when files only in vendor")
	}
}

func TestDetectRust(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "main.rs")
	touch(t, dir, "lib.rs")
	touch(t, dir, "lib_test.rs")

	l, err := Detect(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if l.Name != "rust" {
		t.Errorf("expected rust, got %s", l.Name)
	}
}

func TestDetectConfiguredLanguage(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "main.rb")
	touch(t, dir, "test_main.rb")

	overrides := map[string]config.LanguageConfig{
		"ruby": {
			TestPatterns:   []string{"test_*.rb", "*_test.rb"},
			SourcePatterns: []string{"*.rb"},
		},
	}

	l, err := Detect(dir, overrides)
	if err != nil {
		t.Fatal(err)
	}
	if l.Name != "ruby" {
		t.Errorf("expected ruby, got %s", l.Name)
	}
	if l.TestPatterns[0] != "test_*.rb" {
		t.Errorf("expected test_*.rb pattern, got %s", l.TestPatterns[0])
	}
}

func TestBuiltinLanguagesIncludesRust(t *testing.T) {
	langs := BuiltinLanguages()
	if _, ok := langs["rust"]; !ok {
		t.Error("expected rust in builtin languages")
	}
}

func TestBuiltinLanguagesIncludesCUDA(t *testing.T) {
	langs := BuiltinLanguages()
	if _, ok := langs["cuda"]; !ok {
		t.Error("expected cuda in builtin languages")
	}
}

func TestDetectCUDA(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "kernel.cu")
	touch(t, dir, "helpers.cuh")
	touch(t, dir, "test_kernel.cu")

	l, err := Detect(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if l.Name != "cuda" {
		t.Errorf("expected cuda, got %s", l.Name)
	}
}

func TestDetectAllMultipleLanguages(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "main.rs")
	touch(t, dir, "lib.rs")
	touch(t, dir, "kernel.cu")
	touch(t, dir, "helpers.cuh")

	langs, err := DetectAll(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(langs) != 2 {
		t.Fatalf("expected 2 languages, got %d", len(langs))
	}

	names := map[string]bool{}
	for _, l := range langs {
		names[l.Name] = true
	}
	if !names["rust"] {
		t.Error("expected rust in detected languages")
	}
	if !names["cuda"] {
		t.Error("expected cuda in detected languages")
	}
}

func TestDetectAllSortedByCount(t *testing.T) {
	dir := t.TempDir()
	// 3 rust files, 1 cuda file — rust should be first
	touch(t, dir, "main.rs")
	touch(t, dir, "lib.rs")
	touch(t, dir, "utils.rs")
	touch(t, dir, "kernel.cu")

	langs, err := DetectAll(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(langs) < 2 {
		t.Fatalf("expected at least 2 languages, got %d", len(langs))
	}
	if langs[0].Name != "rust" {
		t.Errorf("expected rust first (most files), got %s", langs[0].Name)
	}
}

func TestDetectAllEmpty(t *testing.T) {
	dir := t.TempDir()
	_, err := DetectAll(dir, nil)
	if err == nil {
		t.Error("expected error for empty directory")
	}
}

func TestFindFilesMulti(t *testing.T) {
	dir := t.TempDir()
	touch(t, dir, "main.rs")
	touch(t, dir, "lib_test.rs")
	touch(t, dir, "kernel.cu")
	touch(t, dir, "helpers.cuh")
	touch(t, dir, "test_kernel.cu")
	touch(t, dir, "README.md")

	langs := []*Language{
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

	src, tests, err := FindFilesMulti(dir, langs)
	if err != nil {
		t.Fatal(err)
	}
	if len(src) != 3 {
		t.Errorf("expected 3 source files (main.rs, kernel.cu, helpers.cuh), got %d", len(src))
	}
	if len(tests) != 2 {
		t.Errorf("expected 2 test files (lib_test.rs, test_kernel.cu), got %d", len(tests))
	}
}

func TestIsTestFileMulti(t *testing.T) {
	langs := []*Language{
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

	if !IsTestFileMulti("lib_test.rs", langs) {
		t.Error("expected lib_test.rs to be a test file")
	}
	if !IsTestFileMulti("test_kernel.cu", langs) {
		t.Error("expected test_kernel.cu to be a test file")
	}
	if IsTestFileMulti("main.rs", langs) {
		t.Error("expected main.rs to not be a test file")
	}
	if IsTestFileMulti("kernel.cu", langs) {
		t.Error("expected kernel.cu to not be a test file")
	}
}

func touch(t *testing.T, dir, name string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(""), 0644); err != nil {
		t.Fatal(err)
	}
}

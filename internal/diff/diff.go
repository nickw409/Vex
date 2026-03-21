package diff

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/nickw409/vex/internal/lang"
)

func ChangedFiles(dir string) ([]string, error) {
	cmd := exec.Command("git", "diff", "HEAD", "--name-only")
	cmd.Dir = dir

	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("running git diff: %w", err)
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 1 && lines[0] == "" {
		return nil, nil
	}

	var files []string
	for _, line := range lines {
		abs := filepath.Join(dir, line)
		files = append(files, abs)
	}

	return files, nil
}

func FilterByLanguage(files []string, langs []*lang.Language) (sourceFiles []string, testFiles []string) {
	for _, f := range files {
		name := filepath.Base(f)

		isTest := false
		for _, l := range langs {
			if matchesAny(name, l.TestPatterns) {
				testFiles = append(testFiles, f)
				isTest = true
				break
			}
		}
		if isTest {
			continue
		}

		for _, l := range langs {
			if matchesAny(name, l.SourcePatterns) {
				sourceFiles = append(sourceFiles, f)
				break
			}
		}
	}
	return
}

func matchesAny(name string, patterns []string) bool {
	for _, p := range patterns {
		if matched, _ := filepath.Match(p, name); matched {
			return true
		}
	}
	return false
}

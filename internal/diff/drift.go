package diff

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type DriftResult struct {
	Section      string   `json:"section"`
	ChangedFiles []string `json:"changed_files"`
}

// Drift checks if files under the given paths have changed since the
// given timestamp. Returns nil if no files changed.
func Drift(dir string, paths []string, since time.Time) (*DriftResult, error) {
	sinceStr := since.Format(time.RFC3339)

	var allChanged []string
	for _, p := range paths {
		absPath := filepath.Join(dir, p)

		cmd := exec.Command("git", "log", "--since="+sinceStr, "--name-only", "--pretty=format:", "--", absPath)
		cmd.Dir = dir

		out, err := cmd.Output()
		if err != nil {
			return nil, fmt.Errorf("running git log for %s: %w", p, err)
		}

		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line != "" {
				allChanged = append(allChanged, line)
			}
		}
	}

	if len(allChanged) == 0 {
		return nil, nil
	}

	// Deduplicate
	seen := make(map[string]bool)
	var unique []string
	for _, f := range allChanged {
		if !seen[f] {
			seen[f] = true
			unique = append(unique, f)
		}
	}

	return &DriftResult{ChangedFiles: unique}, nil
}

// ReportModTime returns the modification time of .vex/report.json,
// or zero time if the file doesn't exist.
func ReportModTime(dir string) time.Time {
	info, err := os.Stat(filepath.Join(dir, ".vex", "report.json"))
	if err != nil {
		return time.Time{}
	}
	return info.ModTime()
}

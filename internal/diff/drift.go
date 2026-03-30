package diff

import (
	"encoding/json"
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

	absPaths := make([]string, len(paths))
	for i, p := range paths {
		absPaths[i] = filepath.Join(dir, p)
	}

	var allChanged []string

	// Committed changes since last check — single git log for all paths.
	args := append([]string{"log", "--since=" + sinceStr, "--name-only", "--pretty=format:", "--"}, absPaths...)
	cmd := exec.Command("git", args...)
	cmd.Dir = dir

	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("running git log: %w", err)
	}

	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			allChanged = append(allChanged, line)
		}
	}

	// Uncommitted changes (staged + unstaged) — single git diff for all paths.
	args = append([]string{"diff", "HEAD", "--name-only", "--"}, absPaths...)
	cmd = exec.Command("git", args...)
	cmd.Dir = dir

	out, err = cmd.Output()
	if err == nil {
		for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			line = strings.TrimSpace(line)
			if line != "" {
				allChanged = append(allChanged, line)
			}
		}
	}
	// HEAD may not exist in empty repos, ignore error

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

// PreviousReport holds fields from a previous report.json needed for
// drift detection and result carry-forward.
type PreviousReport struct {
	SectionChecksums map[string]string `json:"section_checksums"`
	Gaps             []struct {
		Behavior   string `json:"behavior"`
		Detail     string `json:"detail"`
		Suggestion string `json:"suggestion"`
	} `json:"gaps"`
	Covered []string `json:"covered"`
}

// LoadPreviousReport reads .vex/report.json and returns its contents.
// Returns nil if the file doesn't exist or is unreadable.
func LoadPreviousReport(dir string) *PreviousReport {
	data, err := os.ReadFile(filepath.Join(dir, ".vex", "report.json"))
	if err != nil {
		return nil
	}
	var prev PreviousReport
	if err := json.Unmarshal(data, &prev); err != nil {
		return nil
	}
	return &prev
}

// ReportChecksums reads section_checksums from .vex/report.json.
// Returns nil if the file doesn't exist or has no checksums.
func ReportChecksums(dir string) map[string]string {
	prev := LoadPreviousReport(dir)
	if prev == nil {
		return nil
	}
	return prev.SectionChecksums
}

// PreviousValidation holds fields from a previous validation.json
// needed for drift detection and result carry-forward.
type PreviousValidation struct {
	SectionChecksums map[string]string `json:"section_checksums"`
	Suggestions      []struct {
		Section      string `json:"section"`
		BehaviorName string `json:"behavior_name"`
		Description  string `json:"description"`
		Relation     string `json:"relation"`
	} `json:"suggestions"`
}

// LoadPreviousValidation reads .vex/validation.json and returns its contents.
// Returns nil if the file doesn't exist or is unreadable.
func LoadPreviousValidation(dir string) *PreviousValidation {
	data, err := os.ReadFile(filepath.Join(dir, ".vex", "validation.json"))
	if err != nil {
		return nil
	}
	var prev PreviousValidation
	if err := json.Unmarshal(data, &prev); err != nil {
		return nil
	}
	return &prev
}

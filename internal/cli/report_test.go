package cli

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/nickw409/vex/internal/report"
	"github.com/spf13/cobra"
)

func TestPrintReportNoGaps(t *testing.T) {
	r := &report.Report{
		Summary: report.Summary{TotalBehaviors: 5, FullyCovered: 5, GapsFound: 0},
		Gaps:    []report.Gap{},
	}

	cmd := &cobra.Command{}
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)

	printReport(cmd, r)
	output := buf.String()

	if !strings.Contains(output, "5 behaviors: 5 covered, 0 gaps") {
		t.Errorf("expected summary line, got: %s", output)
	}
	if !strings.Contains(output, "No gaps found.") {
		t.Errorf("expected 'No gaps found.', got: %s", output)
	}
}

func TestPrintReportWithGaps(t *testing.T) {
	r := &report.Report{
		Summary: report.Summary{TotalBehaviors: 10, FullyCovered: 7, GapsFound: 5},
		Gaps: []report.Gap{
			{Behavior: "login", Detail: "No test for invalid credentials", Suggestion: "Add test"},
			{Behavior: "login", Detail: "No test for expired token", Suggestion: "Add test"},
			{Behavior: "logout", Detail: "Missing test", Suggestion: "Add test"},
		},
	}

	cmd := &cobra.Command{}
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)

	printReport(cmd, r)
	output := buf.String()

	if !strings.Contains(output, "10 behaviors: 7 covered, 5 gaps") {
		t.Errorf("expected summary, got: %s", output)
	}
	if !strings.Contains(output, "login (2)") {
		t.Errorf("expected 'login (2)' group header, got: %s", output)
	}
	if !strings.Contains(output, "logout (1)") {
		t.Errorf("expected 'logout (1)' group header, got: %s", output)
	}
	if !strings.Contains(output, "No test for invalid credentials") {
		t.Errorf("expected gap detail, got: %s", output)
	}
}

func TestPrintReportTruncatesLongDetails(t *testing.T) {
	long := strings.Repeat("x", 200)
	r := &report.Report{
		Summary: report.Summary{TotalBehaviors: 1, FullyCovered: 0, GapsFound: 1},
		Gaps: []report.Gap{
			{Behavior: "test", Detail: long, Suggestion: "fix"},
		},
	}

	cmd := &cobra.Command{}
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)

	printReport(cmd, r)
	output := buf.String()

	if strings.Contains(output, long) {
		t.Error("expected long detail to be truncated")
	}
	if !strings.Contains(output, "...") {
		t.Error("expected truncation marker '...'")
	}
}

func TestPrintReportReplacesNewlines(t *testing.T) {
	r := &report.Report{
		Summary: report.Summary{TotalBehaviors: 1, FullyCovered: 0, GapsFound: 1},
		Gaps: []report.Gap{
			{Behavior: "test", Detail: "line one\nline two\nline three", Suggestion: "fix"},
		},
	}

	cmd := &cobra.Command{}
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)

	printReport(cmd, r)
	output := buf.String()

	if strings.Contains(output, "\nline two") {
		t.Error("expected newlines in detail to be replaced with spaces")
	}
	if !strings.Contains(output, "line one line two line three") {
		t.Errorf("expected newlines replaced with spaces, got: %s", output)
	}
}

func TestPrintReportAlphabeticalOrder(t *testing.T) {
	r := &report.Report{
		Summary: report.Summary{TotalBehaviors: 3, FullyCovered: 0, GapsFound: 3},
		Gaps: []report.Gap{
			{Behavior: "zebra", Detail: "z gap", Suggestion: "fix"},
			{Behavior: "alpha", Detail: "a gap", Suggestion: "fix"},
			{Behavior: "middle", Detail: "m gap", Suggestion: "fix"},
		},
	}

	cmd := &cobra.Command{}
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)

	printReport(cmd, r)
	output := buf.String()

	alphaIdx := strings.Index(output, "alpha")
	middleIdx := strings.Index(output, "middle")
	zebraIdx := strings.Index(output, "zebra")

	if alphaIdx > middleIdx || middleIdx > zebraIdx {
		t.Errorf("expected alphabetical order (alpha < middle < zebra), got: %s", output)
	}
}

func TestReportCommandNoFile(t *testing.T) {
	cmd := NewRootCmd()
	cmd.SetArgs([]string{"report"})

	err := cmd.Execute()
	if err == nil {
		t.Error("expected error when no report.json exists")
	}
	if err != nil && !strings.Contains(err.Error(), "run vex check first") {
		t.Errorf("expected 'run vex check first' hint in error, got: %s", err.Error())
	}
}

func TestReportCommandCorruptFile(t *testing.T) {
	dir := t.TempDir()
	vexPath := dir + "/.vex"
	os.MkdirAll(vexPath, 0755)
	os.WriteFile(vexPath+"/report.json", []byte("not valid json{{{"), 0644)

	origDir, _ := os.Getwd()
	defer os.Chdir(origDir)
	os.Chdir(dir)

	cmd := NewRootCmd()
	cmd.SetArgs([]string{"report"})

	err := cmd.Execute()
	if err == nil {
		t.Error("expected error for corrupt report.json")
	}
	if err != nil && !strings.Contains(err.Error(), "parsing report") {
		t.Errorf("expected 'parsing report' in error, got: %s", err.Error())
	}
}

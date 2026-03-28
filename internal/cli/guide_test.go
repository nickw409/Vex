package cli

import (
	"bytes"
	"strings"
	"testing"
)

func TestGuideOutput(t *testing.T) {
	cmd := NewRootCmd()
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)
	cmd.SetArgs([]string{"guide"})

	if err := cmd.Execute(); err != nil {
		t.Fatal(err)
	}

	output := buf.String()
	if output == "" {
		t.Error("guide should produce output")
	}

	for _, want := range []string{
		"vexspec",
		"behaviors",
		"Workflow",
		"Section Sizing",
		"under 10 behaviors",
		".vex/report.json",
		".vex/validation.json",
		"vex validate",
		"Formulas and Equations",
		"geometric-brownian-motion",
		"black-scholes-call",
	} {
		if !strings.Contains(output, want) {
			t.Errorf("guide output should contain %q", want)
		}
	}
}

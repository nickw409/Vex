package cli

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/nwiley/vex/internal/diff"
	"github.com/nwiley/vex/internal/spec"
	"github.com/spf13/cobra"
)

type driftReport struct {
	Drifted []diff.DriftResult `json:"drifted"`
	Clean   []string           `json:"clean"`
}

func newDriftCmd() *cobra.Command {
	var specPath string

	cmd := &cobra.Command{
		Use:   "drift",
		Short: "Check if code has changed since last vex check",
		RunE: func(cmd *cobra.Command, args []string) error {
			ps, err := spec.LoadProject(specPath)
			if err != nil {
				return fmt.Errorf("loading spec: %w", err)
			}

			cwd, err := os.Getwd()
			if err != nil {
				return fmt.Errorf("getting working directory: %w", err)
			}

			since := diff.ReportModTime(cwd)
			if since.IsZero() {
				fmt.Fprintln(os.Stderr, "no previous check found — run vex check first")
				os.Exit(2)
			}

			fmt.Fprintf(os.Stderr, "checking changes since last check (%s)\n", since.Format("2006-01-02 15:04:05"))

			report := driftReport{
				Drifted: []diff.DriftResult{},
				Clean:   []string{},
			}

			for _, section := range ps.Sections {
				paths := spec.SectionPaths(&section)
				if len(paths) == 0 {
					continue
				}

				result, err := diff.Drift(cwd, paths, since)
				if err != nil {
					fmt.Fprintf(os.Stderr, "warning: drift check failed for %s: %v\n", section.Name, err)
					continue
				}

				if result == nil {
					report.Clean = append(report.Clean, section.Name)
				} else {
					result.Section = section.Name
					report.Drifted = append(report.Drifted, *result)
				}
			}

			out, err := json.MarshalIndent(report, "", "  ")
			if err != nil {
				return fmt.Errorf("marshaling drift report: %w", err)
			}

			fmt.Println(string(out))

			if len(report.Drifted) > 0 {
				writeOutput("drift.json", out)
				os.Exit(1)
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&specPath, "spec", ".vex/vexspec.yaml", "path to vexspec file")

	return cmd
}

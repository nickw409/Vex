package cli

import (
	"fmt"
	"os"

	"github.com/nwiley/vex/internal/check"
	"github.com/nwiley/vex/internal/diff"
	"github.com/nwiley/vex/internal/lang"
	"github.com/nwiley/vex/internal/provider"
	"github.com/nwiley/vex/internal/report"
	"github.com/nwiley/vex/internal/spec"
	"github.com/spf13/cobra"
)

func newCheckCmd() *cobra.Command {
	var specPath string
	var section string
	var useDrift bool

	cmd := &cobra.Command{
		Use:   "check",
		Short: "Check test coverage against a vexspec",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			ps, err := spec.LoadProject(specPath)
			if err != nil {
				return err
			}

			p, err := provider.New(cfg)
			if err != nil {
				return err
			}

			sections := ps.Sections
			if section != "" {
				found := false
				for _, sec := range ps.Sections {
					if sec.Name == section {
						sections = []spec.Section{sec}
						found = true
						break
					}
				}
				if !found {
					return fmt.Errorf("section %q not found in spec", section)
				}
			}

			if useDrift {
				cwd, err := os.Getwd()
				if err != nil {
					return fmt.Errorf("getting working directory: %w", err)
				}
				since := diff.ReportModTime(cwd)
				if since.IsZero() {
					fmt.Fprintln(os.Stderr, "no previous check found, checking all sections")
				} else {
					var drifted []spec.Section
					for _, sec := range sections {
						paths := spec.SectionPaths(&sec)
						result, err := diff.Drift(cwd, paths, since)
						if err != nil {
							fmt.Fprintf(os.Stderr, "warning: drift check failed for %s: %v\n", sec.Name, err)
							drifted = append(drifted, sec)
							continue
						}
						if result != nil {
							drifted = append(drifted, sec)
						} else {
							fmt.Fprintf(os.Stderr, "skipping clean section %q\n", sec.Name)
						}
					}
					sections = drifted
					if len(sections) == 0 {
						fmt.Fprintln(os.Stderr, "all sections clean, nothing to check")
						return emptyReport(ps)
					}
				}
			}

			var inputs []check.SectionInput
			for i := range sections {
				sec := &sections[i]
				behaviors := ps.AllBehaviors(sec)
				if len(behaviors) == 0 {
					continue
				}

				paths := spec.SectionPaths(sec)
				if len(paths) == 0 {
					continue
				}

				srcMap := make(map[string]string)
				testMap := make(map[string]string)

				for _, dir := range paths {
					l, err := lang.Detect(dir, cfg.Languages)
					if err != nil {
						fmt.Fprintf(os.Stderr, "warning: skipping path %s: %v\n", dir, err)
						continue
					}

					sourceFiles, testFiles, err := lang.FindFiles(dir, l)
					if err != nil {
						fmt.Fprintf(os.Stderr, "warning: skipping path %s: %v\n", dir, err)
						continue
					}

					for _, f := range sourceFiles {
						data, err := os.ReadFile(f)
						if err != nil {
							return fmt.Errorf("reading %s: %w", f, err)
						}
						srcMap[f] = string(data)
					}
					for _, f := range testFiles {
						data, err := os.ReadFile(f)
						if err != nil {
							return fmt.Errorf("reading %s: %w", f, err)
						}
						testMap[f] = string(data)
					}
				}

				if len(testMap) == 0 && len(srcMap) == 0 {
					fmt.Fprintf(os.Stderr, "warning: no files found for section %q\n", sec.Name)
					continue
				}

				inputs = append(inputs, check.SectionInput{
					Section:     sec,
					Behaviors:   behaviors,
					SourceFiles: srcMap,
					TestFiles:   testMap,
				})
			}

			if len(inputs) == 0 {
				return emptyReport(ps)
			}

			r, err := check.RunProject(cmd.Context(), p, ps, inputs, cfg.MaxConcurrency)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
			}

			return outputReport(r)
		},
	}

	cmd.Flags().StringVar(&specPath, "spec", "", "path to vexspec.yaml (default: .vex/vexspec.yaml)")
	cmd.Flags().StringVar(&section, "section", "", "check only this section")
	cmd.Flags().BoolVar(&useDrift, "drift", false, "only check sections with changes since last check")

	return cmd
}

func emptyReport(ps *spec.ProjectSpec) error {
	totalBehaviors := 0
	for _, sec := range ps.Sections {
		totalBehaviors += len(ps.AllBehaviors(&sec))
	}

	r := &report.Report{
		Spec:    ".vex/vexspec.yaml",
		Gaps:    []report.Gap{},
		Covered: []report.Covered{},
	}
	r.ComputeSummary(totalBehaviors)
	return outputReport(r)
}

func outputReport(r *report.Report) error {
	out, err := r.JSON()
	if err != nil {
		return fmt.Errorf("marshaling report: %w", err)
	}

	fmt.Fprintln(os.Stdout, string(out))

	if err := writeOutput("report.json", out); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}

	if r.HasGaps() {
		os.Exit(1)
	}
	return nil
}

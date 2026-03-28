package cli

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/nickw409/vex/internal/check"
	"github.com/nickw409/vex/internal/diff"
	"github.com/nickw409/vex/internal/lang"
	"github.com/nickw409/vex/internal/log"
	"github.com/nickw409/vex/internal/perf"
	"github.com/nickw409/vex/internal/provider"
	"github.com/nickw409/vex/internal/report"
	"github.com/nickw409/vex/internal/spec"
	"github.com/spf13/cobra"
)

func newCheckCmd() *cobra.Command {
	var specPath string
	var section string
	var useDrift bool
	var useProfile bool

	cmd := &cobra.Command{
		Use:   "check",
		Short: "Check test coverage against a vexspec",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			var prof *perf.Profile
			if useProfile {
				prof = perf.New()
			}

			endLoad := profStart(prof, "spec:load", "")
			ps, err := spec.LoadProject(specPath)
			endLoad()
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
				endDrift := profStart(prof, "drift:total", "")
				cwd, err := os.Getwd()
				if err != nil {
					endDrift()
					return fmt.Errorf("getting working directory: %w", err)
				}
				since := diff.ReportModTime(cwd)
				if since.IsZero() {
					log.Info("no previous check found, checking all sections")
				} else {
					log.Info("checking drift since %s", since.Format("2006-01-02 15:04:05"))
					var drifted []spec.Section
					for _, sec := range sections {
						paths := spec.SectionAllPaths(&sec)
						endSec := profStart(prof, "drift:section", sec.Name)
						result, err := diff.Drift(cwd, paths, since)
						endSec()
						if err != nil {
							log.Info("warning: drift check failed for %s: %v", sec.Name, err)
							drifted = append(drifted, sec)
							continue
						}
						if result != nil {
							drifted = append(drifted, sec)
						} else {
							log.Info("skipping clean section %q", sec.Name)
						}
					}
					sections = drifted
					if len(sections) == 0 {
						log.Info("all sections clean, nothing to check")
						endDrift()
						return emptyReport(ps)
					}
				}
				endDrift()
			}

			endInputs := profStart(prof, "inputs:total", "")
			var inputs []check.SectionInput
			for i := range sections {
				sec := &sections[i]
				behaviors := ps.AllBehaviors(sec)
				if len(behaviors) == 0 {
					continue
				}

				paths := spec.SectionPaths(sec)
				files := spec.SectionFiles(sec)

				if len(paths) == 0 && len(files) == 0 {
					continue
				}

				srcMap := make(map[string]string)
				testMap := make(map[string]string)

				// path: entries — walk directories for all source and test files
				for _, dir := range paths {
					endDetect := profStart(prof, "inputs:detect", sec.Name)
					langs, err := lang.DetectAll(dir, cfg.Languages)
					endDetect()
					if err != nil {
						log.Info("warning: skipping path %s: %v", dir, err)
						continue
					}

					endFind := profStart(prof, "inputs:find_files", sec.Name)
					sourceFiles, testFiles, err := lang.FindFilesMulti(dir, langs)
					endFind()
					if err != nil {
						log.Info("warning: skipping path %s: %v", dir, err)
						continue
					}

					endRead := profStart(prof, "inputs:read_source", sec.Name)
					for _, f := range sourceFiles {
						data, err := os.ReadFile(f)
						if err != nil {
							endRead()
							return fmt.Errorf("reading %s: %w", f, err)
						}
						srcMap[f] = string(data)
					}
					endRead()

					endReadTest := profStart(prof, "inputs:read_tests", sec.Name)
					for _, f := range testFiles {
						data, err := os.ReadFile(f)
						if err != nil {
							endReadTest()
							return fmt.Errorf("reading %s: %w", f, err)
						}
						testMap[f] = string(data)
					}
					endReadTest()
				}

				// file: entries — classify as source or test, then read
				if len(files) > 0 {
					endFiles := profStart(prof, "inputs:read_explicit", sec.Name)
					// Detect language from the first file's directory
					langs, langErr := lang.DetectAll(filepath.Dir(files[0]), cfg.Languages)

					for _, f := range files {
						data, err := os.ReadFile(f)
						if err != nil {
							endFiles()
							return fmt.Errorf("reading %s: %w", f, err)
						}
						if langErr == nil && lang.IsTestFileMulti(f, langs) {
							testMap[f] = string(data)
						} else {
							srcMap[f] = string(data)
						}
					}
					endFiles()
				}

				if len(testMap) == 0 && len(srcMap) == 0 {
					log.Info("warning: no files found for section %q", sec.Name)
					continue
				}

				inputs = append(inputs, check.SectionInput{
					Section:     sec,
					Behaviors:   behaviors,
					SourceFiles: srcMap,
					TestFiles:   testMap,
				})
			}
			endInputs()

			if len(inputs) == 0 {
				return emptyReport(ps)
			}

			log.Info("checking %d section(s)", len(inputs))
			endCheck := profStart(prof, "check:total", "")
			r, err := check.RunProject(cmd.Context(), p, ps, inputs, cfg.MaxConcurrency, prof)
			endCheck()
			if err != nil {
				log.Info("check done with errors: %v", err)
			}

			if prof != nil {
				prof.Print()
				if writeErr := prof.WriteFile(filepath.Join(vexDir, "profile.json")); writeErr != nil {
					fmt.Fprintln(os.Stderr, writeErr)
				} else {
					fmt.Fprintln(os.Stderr, ".vex/profile.json")
				}
			}

			return outputReport(r)
		},
	}

	cmd.Flags().StringVar(&specPath, "spec", "", "path to vexspec.yaml (default: .vex/vexspec.yaml)")
	cmd.Flags().StringVar(&section, "section", "", "check only this section")
	cmd.Flags().BoolVar(&useDrift, "drift", false, "only check sections with changes since last check")
	cmd.Flags().BoolVar(&useProfile, "profile", false, "write performance profile to .vex/profile.json")

	return cmd
}

// profStart starts a profiling span if prof is non-nil, otherwise returns a no-op.
func profStart(prof *perf.Profile, name, parent string) func() {
	if prof != nil {
		return prof.Start(name, parent)
	}
	return func() {}
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

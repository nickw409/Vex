package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

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

// newProviderFunc is the factory used to create LLM providers.
// Tests override this to inject mocks.
var newProviderFunc = provider.New

// newProviderWithModelFunc creates a provider with a specific model override.
// Tests override this to inject mocks.
var newProviderWithModelFunc = provider.NewWithModel

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

			for _, name := range ps.OversizedSections() {
				log.Info("warning: section %q exceeds %d behaviors — consider splitting", name, spec.MaxSectionBehaviors)
			}

			pass2Provider, err := newProviderFunc(cfg)
			if err != nil {
				return err
			}

			// Use a cheaper model for pass 1 (triage) to reduce cost.
			var pass1Provider provider.Provider
			if cfg.Pass1Model != cfg.Model {
				pass1Provider, err = newProviderWithModelFunc(cfg, cfg.Pass1Model)
				if err != nil {
					return err
				}
				log.Info("using %s for pass 1, %s for pass 2", cfg.Pass1Model, cfg.Model)
			} else {
				pass1Provider = pass2Provider
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

			var prevReport *diff.PreviousReport
			var skippedSections []spec.Section

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
					prevReport = diff.LoadPreviousReport(cwd)
					var prevChecksums map[string]string
					if prevReport != nil {
						prevChecksums = prevReport.SectionChecksums
					}
					log.Info("checking drift since %s", since.Format("2006-01-02 15:04:05"))
					var drifted []spec.Section
					for _, sec := range sections {
						// Check if the section's spec content changed
						currentSum := spec.SectionChecksum(&sec, ps.ResolveShared(&sec))
						if prevSum, ok := prevChecksums[sec.Name]; ok && prevSum == currentSum {
							// Spec unchanged — check file drift
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
								skippedSections = append(skippedSections, sec)
							}
						} else {
							if prevChecksums == nil {
								log.Info("no stored checksums, checking section %q", sec.Name)
							} else {
								log.Info("spec changed for section %q", sec.Name)
							}
							drifted = append(drifted, sec)
						}
					}
					sections = drifted
					if len(sections) == 0 {
						log.Info("all sections clean, nothing to check")
						endDrift()
						return carryForwardReport(ps, prevReport, skippedSections)
					}
				}
				endDrift()
			}

			// Pre-filter sections: extract covered overrides and skip empty ones.
			// This is cheap (no I/O) so we do it up front to know totalSections.
			type sectionWork struct {
				sec       *spec.Section
				behaviors []spec.Behavior
				paths     []string
				files     []string
			}
			var workItems []sectionWork
			var coveredOverrides []report.Covered
			for i := range sections {
				sec := &sections[i]
				behaviors := ps.AllBehaviors(sec)

				// Extract covered overrides — these skip the LLM entirely.
				if len(sec.Covered) > 0 {
					overrideSet := make(map[string]string, len(sec.Covered))
					for _, co := range sec.Covered {
						overrideSet[co.Behavior] = co.Reason
					}

					var filtered []spec.Behavior
					for _, b := range behaviors {
						if reason, ok := overrideSet[b.Name]; ok {
							coveredOverrides = append(coveredOverrides, report.Covered{
								Behavior: b.Name,
								Detail:   reason,
							})
							log.Info("section %q: %q marked covered (override)", sec.Name, b.Name)
						} else {
							filtered = append(filtered, b)
						}
					}
					behaviors = filtered
				}

				if len(behaviors) == 0 {
					continue
				}

				paths := spec.SectionPaths(sec)
				files := spec.SectionFiles(sec)

				if len(paths) == 0 && len(files) == 0 {
					continue
				}

				workItems = append(workItems, sectionWork{
					sec:       sec,
					behaviors: behaviors,
					paths:     paths,
					files:     files,
				})
			}

			if len(workItems) == 0 {
				return emptyReport(ps)
			}

			log.Info("checking %d section(s)", len(workItems))

			// Build inputs concurrently and stream them into the check engine.
			// LLM calls start as soon as the first section's files are read.
			inputCh := make(chan check.SectionInput, len(workItems))
			var inputWg sync.WaitGroup
			var inputErr error
			var inputErrOnce sync.Once

			// Use a semaphore to limit concurrent file I/O.
			ioSem := make(chan struct{}, cfg.MaxConcurrency)

			endInputs := profStart(prof, "inputs:total", "")
			for _, w := range workItems {
				inputWg.Add(1)
				go func(w sectionWork) {
					defer inputWg.Done()

					ioSem <- struct{}{}
					defer func() { <-ioSem }()

					srcMap := make(map[string]string)
					testMap := make(map[string]string)

					// path: entries — detect languages and find files in a single walk
					for _, dir := range w.paths {
						endDetect := profStart(prof, "inputs:detect_and_find", w.sec.Name)
						_, sourceFiles, testFiles, err := lang.DetectAndFind(dir, cfg.Languages)
						endDetect()
						if err != nil {
							log.Info("warning: skipping path %s: %v", dir, err)
							continue
						}

						// Filter to files relevant to this section when the path is broad.
						beforeSrc := len(sourceFiles)
						beforeTest := len(testFiles)
						sourceFiles = check.FilterRelevantFiles(w.sec, w.behaviors, sourceFiles)
						testFiles = check.FilterRelevantFiles(w.sec, w.behaviors, testFiles)
						if len(sourceFiles) < beforeSrc || len(testFiles) < beforeTest {
							log.Info("section %q: filtered %d/%d source, %d/%d test files",
								w.sec.Name, len(sourceFiles), beforeSrc, len(testFiles), beforeTest)
						}

						endRead := profStart(prof, "inputs:read_files", w.sec.Name)
						for _, f := range sourceFiles {
							data, err := os.ReadFile(f)
							if err != nil {
								endRead()
								inputErrOnce.Do(func() { inputErr = fmt.Errorf("reading %s: %w", f, err) })
								return
							}
							srcMap[f] = string(data)
						}
						for _, f := range testFiles {
							data, err := os.ReadFile(f)
							if err != nil {
								endRead()
								inputErrOnce.Do(func() { inputErr = fmt.Errorf("reading %s: %w", f, err) })
								return
							}
							testMap[f] = string(data)
						}
						endRead()
					}

					// file: entries — classify as source or test, then read
					if len(w.files) > 0 {
						endFiles := profStart(prof, "inputs:read_explicit", w.sec.Name)
						langs, langErr := lang.DetectAll(filepath.Dir(w.files[0]), cfg.Languages)

						for _, f := range w.files {
							data, err := os.ReadFile(f)
							if err != nil {
								endFiles()
								inputErrOnce.Do(func() { inputErr = fmt.Errorf("reading %s: %w", f, err) })
								return
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
						log.Info("warning: no files found for section %q", w.sec.Name)
						return
					}

					inputCh <- check.SectionInput{
						Section:     w.sec,
						Behaviors:   w.behaviors,
						SourceFiles: srcMap,
						TestFiles:   testMap,
					}
				}(w)
			}

			// Close the channel once all input goroutines finish.
			go func() {
				inputWg.Wait()
				endInputs()
				close(inputCh)
			}()

			endCheck := profStart(prof, "check:total", "")
			r, err := check.RunProjectStreaming(cmd.Context(), pass1Provider, pass2Provider, ps, inputCh, len(workItems), cfg.MaxConcurrency, prof)
			endCheck()

			if inputErr != nil {
				return inputErr
			}
			if err != nil {
				log.Info("check done with errors: %v", err)
			}

			// Carry forward previous results for skipped sections.
			if prevReport != nil && len(skippedSections) > 0 {
				mergeSkippedResults(r, prevReport, ps, skippedSections)
			}

			// Inject covered overrides into the report.
			if len(coveredOverrides) > 0 {
				r.Covered = append(r.Covered, coveredOverrides...)
				r.ComputeSummary(r.Summary.TotalBehaviors + len(coveredOverrides))
			}

			// Store per-section checksums for drift detection.
			r.SectionChecksums = make(map[string]string, len(ps.Sections))
			for _, sec := range ps.Sections {
				r.SectionChecksums[sec.Name] = spec.SectionChecksum(&sec, ps.ResolveShared(&sec))
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
	cmd.Flags().BoolVar(&useDrift, "drift", true, "only check sections with changes since last check (default true)")
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

// mergeSkippedResults carries forward gaps and covered entries from the
// previous report for sections that were skipped by drift detection.
func mergeSkippedResults(r *report.Report, prev *diff.PreviousReport, ps *spec.ProjectSpec, skipped []spec.Section) {
	// Build set of behavior names for skipped sections.
	skippedBehaviors := make(map[string]bool)
	extraBehaviors := 0
	for _, sec := range skipped {
		for _, b := range ps.AllBehaviors(&sec) {
			skippedBehaviors[b.Name] = true
			extraBehaviors++
		}
	}

	// Carry forward gaps from previous report for skipped behaviors.
	for _, g := range prev.Gaps {
		if skippedBehaviors[g.Behavior] {
			r.Gaps = append(r.Gaps, report.Gap{
				Behavior:   g.Behavior,
				Detail:     g.Detail,
				Suggestion: g.Suggestion,
			})
		}
	}

	// Carry forward covered entries. Previous report stores covered as
	// a flat list of behavior names. Create Covered entries so
	// ComputeSummary counts them correctly.
	for _, name := range prev.Covered {
		if skippedBehaviors[name] {
			r.Covered = append(r.Covered, report.Covered{
				Behavior: name,
				Detail:   "carried forward from previous check",
			})
		}
	}

	r.ComputeSummary(r.Summary.TotalBehaviors + extraBehaviors)
}

// carryForwardReport produces a report when all sections are clean,
// preserving gaps and covered entries from the previous report.
func carryForwardReport(ps *spec.ProjectSpec, prev *diff.PreviousReport, skipped []spec.Section) error {
	r := &report.Report{
		Spec:    ".vex/vexspec.yaml",
		Gaps:    []report.Gap{},
		Covered: []report.Covered{},
	}
	r.ComputeSummary(0)

	if prev != nil {
		mergeSkippedResults(r, prev, ps, skipped)
	}

	// Store checksums for next drift check.
	r.SectionChecksums = make(map[string]string, len(ps.Sections))
	for _, sec := range ps.Sections {
		r.SectionChecksums[sec.Name] = spec.SectionChecksum(&sec, ps.ResolveShared(&sec))
	}

	return outputReport(r)
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

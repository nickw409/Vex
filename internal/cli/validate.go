package cli

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/nickw409/vex/internal/diff"
	"github.com/nickw409/vex/internal/log"
	"github.com/nickw409/vex/internal/spec"
	"github.com/spf13/cobra"
)

func newValidateCmd() *cobra.Command {
	var specPath string
	var useDrift bool

	cmd := &cobra.Command{
		Use:   "validate [spec-file]",
		Short: "Check if a vexspec is complete",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) > 0 {
				specPath = args[0]
			}

			ps, err := spec.LoadProject(specPath)
			if err != nil {
				return err
			}

			for _, name := range ps.OversizedSections() {
				log.Info("warning: section %q exceeds %d behaviors — consider splitting", name, spec.MaxSectionBehaviors)
			}

			sections := ps.Sections
			var prevValidation *diff.PreviousValidation
			var skippedSuggestions []spec.ValidationSuggestion

			if useDrift {
				cwd, err := os.Getwd()
				if err != nil {
					return fmt.Errorf("getting working directory: %w", err)
				}

				prevValidation = diff.LoadPreviousValidation(cwd)
				if prevValidation == nil || len(prevValidation.SectionChecksums) == 0 {
					log.Info("no previous validation found, validating all sections")
				} else {
					var drifted []spec.Section
					for _, sec := range sections {
						currentSum := spec.SectionChecksum(&sec, ps.ResolveShared(&sec))
						if prevSum, ok := prevValidation.SectionChecksums[sec.Name]; ok && prevSum == currentSum {
							log.Info("skipping clean section %q", sec.Name)
							// Carry forward suggestions for this section.
							for _, s := range prevValidation.Suggestions {
								if s.Section == sec.Name {
									skippedSuggestions = append(skippedSuggestions, spec.ValidationSuggestion{
										Section:      s.Section,
										BehaviorName: s.BehaviorName,
										Description:  s.Description,
										Relation:     s.Relation,
									})
								}
							}
						} else {
							if prevValidation.SectionChecksums == nil {
								log.Info("no stored checksums, validating section %q", sec.Name)
							} else {
								log.Info("spec changed for section %q", sec.Name)
							}
							drifted = append(drifted, sec)
						}
					}
					sections = drifted

					if len(sections) == 0 {
						log.Info("all sections clean, nothing to validate")
						return outputValidation(ps, skippedSuggestions)
					}
				}
			}

			p, err := newProviderFunc(cfg)
			if err != nil {
				return err
			}

			// Build a trimmed ProjectSpec with only drifted sections.
			trimmed := &spec.ProjectSpec{
				Project:     ps.Project,
				Description: ps.Description,
				Shared:      ps.Shared,
				Sections:    sections,
			}

			log.Info("validating %d section(s)", len(sections))
			result, err := spec.ValidateProject(cmd.Context(), p, trimmed, cfg.MaxConcurrency)
			if err != nil {
				return err
			}
			log.Info("validation complete")

			// Merge carried-forward suggestions.
			if len(skippedSuggestions) > 0 {
				result.Suggestions = append(result.Suggestions, skippedSuggestions...)
				if len(result.Suggestions) > 0 {
					result.Complete = false
				}
			}

			// Store checksums for all sections (not just validated ones).
			result.SectionChecksums = make(map[string]string, len(ps.Sections))
			for _, sec := range ps.Sections {
				result.SectionChecksums[sec.Name] = spec.SectionChecksum(&sec, ps.ResolveShared(&sec))
			}

			out, err := json.MarshalIndent(result, "", "  ")
			if err != nil {
				return fmt.Errorf("marshaling result: %w", err)
			}

			fmt.Fprintln(os.Stdout, string(out))

			if err := writeOutput("validation.json", out); err != nil {
				fmt.Fprintln(os.Stderr, err)
			}

			if !result.Complete {
				os.Exit(1)
			}
			return nil
		},
	}

	cmd.Flags().BoolVar(&useDrift, "drift", true, "only validate sections with spec changes since last validation (default true)")

	return cmd
}

func outputValidation(ps *spec.ProjectSpec, suggestions []spec.ValidationSuggestion) error {
	result := &spec.ValidationResult{
		Complete:    len(suggestions) == 0,
		Suggestions: suggestions,
	}

	result.SectionChecksums = make(map[string]string, len(ps.Sections))
	for _, sec := range ps.Sections {
		result.SectionChecksums[sec.Name] = spec.SectionChecksum(&sec, ps.ResolveShared(&sec))
	}

	out, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling result: %w", err)
	}

	fmt.Fprintln(os.Stdout, string(out))

	if err := writeOutput("validation.json", out); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}

	if !result.Complete {
		os.Exit(1)
	}
	return nil
}

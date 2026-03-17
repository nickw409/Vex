package cli

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/nickw409/vex/internal/log"
	"github.com/nickw409/vex/internal/provider"
	"github.com/nickw409/vex/internal/spec"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

func newSpecCmd() *cobra.Command {
	var extend string

	cmd := &cobra.Command{
		Use:   "spec <description>",
		Short: "Generate vexspec sections from a task description",
		Long: `Generates section(s) for .vex/vexspec.yaml from a natural language task description.

Use --extend to add behaviors to an existing section instead of creating a new one.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			p, err := provider.New(cfg)
			if err != nil {
				return err
			}

			specPath := ".vex/vexspec.yaml"

			if extend != "" {
				log.Info("extending section %q", extend)
				return extendSection(cmd, p, specPath, extend, args[0])
			}

			log.Info("generating spec from description")
			sections, err := spec.Generate(cmd.Context(), p, args[0])
			if err != nil {
				return err
			}

			existing, err := spec.LoadProject(specPath)
			if err != nil {
				return createSpec(specPath, sections)
			}

			return appendSections(specPath, existing, sections)
		},
	}

	cmd.Flags().StringVar(&extend, "extend", "", "add behaviors to an existing section by name")

	return cmd
}

func extendSection(cmd *cobra.Command, p provider.Provider, specPath, sectionName, description string) error {
	ps, err := spec.LoadProject(specPath)
	if err != nil {
		return fmt.Errorf("cannot extend: %w", err)
	}

	var target *spec.Section
	var targetIdx int
	for i := range ps.Sections {
		if ps.Sections[i].Name == sectionName {
			target = &ps.Sections[i]
			targetIdx = i
			break
		}
	}
	if target == nil {
		return fmt.Errorf("section %q not found in spec", sectionName)
	}

	result, err := spec.GenerateExtend(cmd.Context(), p, target, description)
	if err != nil {
		return err
	}

	ps.Sections[targetIdx].Behaviors = append(ps.Sections[targetIdx].Behaviors, result.Behaviors...)
	ps.Sections[targetIdx].Subsections = append(ps.Sections[targetIdx].Subsections, result.Subsections...)

	data, err := yaml.Marshal(ps)
	if err != nil {
		return fmt.Errorf("marshaling spec: %w", err)
	}

	if err := os.WriteFile(specPath, data, 0644); err != nil {
		return fmt.Errorf("writing spec: %w", err)
	}

	added := len(result.Behaviors)
	for _, sub := range result.Subsections {
		added += len(sub.Behaviors)
	}

	fmt.Fprintf(os.Stderr, "Extended section %q: added %d behavior(s)\n", sectionName, added)
	return nil
}

func createSpec(path string, sections []spec.Section) error {
	if err := os.MkdirAll(".vex", 0755); err != nil {
		return fmt.Errorf("creating .vex directory: %w", err)
	}

	ps := &spec.ProjectSpec{
		Project:  inferProjectName(),
		Sections: sections,
	}

	data, err := yaml.Marshal(ps)
	if err != nil {
		return fmt.Errorf("marshaling spec: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("writing spec: %w", err)
	}

	fmt.Fprintf(os.Stderr, "Created %s with %d section(s)\n", path, len(sections))
	for _, sec := range sections {
		fmt.Fprintf(os.Stderr, "  - %s (%d behaviors)\n", sec.Name, len(sec.Behaviors))
	}
	return nil
}

func appendSections(path string, existing *spec.ProjectSpec, sections []spec.Section) error {
	existing.Sections = append(existing.Sections, sections...)

	data, err := yaml.Marshal(existing)
	if err != nil {
		return fmt.Errorf("marshaling spec: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("writing spec: %w", err)
	}

	fmt.Fprintf(os.Stderr, "Added %d section(s) to %s\n", len(sections), path)
	for _, sec := range sections {
		fmt.Fprintf(os.Stderr, "  - %s (%d behaviors)\n", sec.Name, len(sec.Behaviors))
	}
	return nil
}

func inferProjectName() string {
	dir, err := os.Getwd()
	if err != nil {
		return "MyProject"
	}
	return filepath.Base(dir)
}

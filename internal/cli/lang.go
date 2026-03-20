package cli

import (
	"fmt"
	"strings"

	"github.com/nickw409/vex/internal/config"
	"github.com/nickw409/vex/internal/lang"
	"github.com/spf13/cobra"
)

func newLangCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "lang",
		Short: "Manage language definitions",
	}

	cmd.AddCommand(newLangAddCmd())
	cmd.AddCommand(newLangListCmd())
	cmd.AddCommand(newLangRemoveCmd())

	return cmd
}

func newLangAddCmd() *cobra.Command {
	var testPatterns string
	var sourcePatterns string

	cmd := &cobra.Command{
		Use:   "add <name>",
		Short: "Add a language to vex.yaml",
		Long:  "Add a custom language with test and source file patterns to vex.yaml.",
		Example: `  vex lang add rust --test-patterns "*_test.rs" --source-patterns "*.rs"
  vex lang add cpp --test-patterns "*_test.cpp,*_test.cc" --source-patterns "*.cpp,*.cc,*.h"`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			name := args[0]

			if testPatterns == "" {
				return fmt.Errorf("--test-patterns is required")
			}
			if sourcePatterns == "" {
				return fmt.Errorf("--source-patterns is required")
			}

			lc := config.LanguageConfig{
				TestPatterns:   splitPatterns(testPatterns),
				SourcePatterns: splitPatterns(sourcePatterns),
			}

			if err := config.AddLanguage("vex.yaml", name, lc); err != nil {
				return err
			}

			fmt.Fprintf(cmd.ErrOrStderr(), "Added language %q to vex.yaml\n", name)
			return nil
		},
	}

	cmd.Flags().StringVar(&testPatterns, "test-patterns", "", "comma-separated glob patterns for test files (required)")
	cmd.Flags().StringVar(&sourcePatterns, "source-patterns", "", "comma-separated glob patterns for source files (required)")

	return cmd
}

func newLangListCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List available languages",
		RunE: func(cmd *cobra.Command, args []string) error {
			w := cmd.OutOrStdout()

			fmt.Fprintln(w, "Built-in languages:")
			for name, l := range lang.BuiltinLanguages() {
				fmt.Fprintf(w, "  %-14s test: %-30s source: %s\n",
					name, strings.Join(l.TestPatterns, ", "), strings.Join(l.SourcePatterns, ", "))
			}

			if cfg != nil && len(cfg.Languages) > 0 {
				fmt.Fprintln(w, "\nConfigured languages (vex.yaml):")
				for name, lc := range cfg.Languages {
					fmt.Fprintf(w, "  %-14s test: %-30s source: %s\n",
						name, strings.Join(lc.TestPatterns, ", "), strings.Join(lc.SourcePatterns, ", "))
				}
			}

			return nil
		},
	}
}

func newLangRemoveCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "remove <name>",
		Short: "Remove a language from vex.yaml",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			name := args[0]
			if err := config.RemoveLanguage("vex.yaml", name); err != nil {
				return err
			}
			fmt.Fprintf(cmd.ErrOrStderr(), "Removed language %q from vex.yaml\n", name)
			return nil
		},
	}
}

func splitPatterns(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

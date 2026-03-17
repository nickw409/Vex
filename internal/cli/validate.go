package cli

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/nickw409/vex/internal/log"
	"github.com/nickw409/vex/internal/provider"
	"github.com/nickw409/vex/internal/spec"
	"github.com/spf13/cobra"
)

func newValidateCmd() *cobra.Command {
	var specPath string

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

			p, err := provider.New(cfg)
			if err != nil {
				return err
			}

			log.Info("validating spec")
			result, err := spec.ValidateProject(cmd.Context(), p, ps)
			if err != nil {
				return err
			}
			log.Info("validation complete")

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

	return cmd
}

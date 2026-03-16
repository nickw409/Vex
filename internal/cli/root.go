package cli

import (
	"github.com/nwiley/vex/internal/config"
	"github.com/spf13/cobra"
)

var cfg *config.Config

func NewRootCmd() *cobra.Command {
	var configPath string

	cmd := &cobra.Command{
		Use:   "vex",
		Short: "Spec-driven test coverage auditor",
		Long:  "Vex verifies that tests fully cover intended behavior described in a spec.",
		SilenceUsage:  true,
		SilenceErrors: true,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			name := cmd.Name()
			if name == "init" || name == "guide" {
				return nil
			}

			var err error
			cfg, err = config.Load(configPath)
			if err != nil {
				cfg = config.Default()
			}
			return nil
		},
	}

	cmd.PersistentFlags().StringVar(&configPath, "config", "", "path to vex.yaml")

	cmd.AddCommand(newInitCmd())
	cmd.AddCommand(newValidateCmd())
	cmd.AddCommand(newCheckCmd())
	cmd.AddCommand(newGuideCmd())
	cmd.AddCommand(newSpecCmd())

	return cmd
}

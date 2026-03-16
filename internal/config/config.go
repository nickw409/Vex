package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Provider       string                    `yaml:"provider"`
	Model          string                    `yaml:"model"`
	MaxConcurrency int                       `yaml:"max_concurrency,omitempty"`
	APIKeyEnv      string                    `yaml:"api_key_env,omitempty"`
	Languages      map[string]LanguageConfig `yaml:"languages,omitempty"`
}

type LanguageConfig struct {
	TestPatterns   []string `yaml:"test_patterns"`
	SourcePatterns []string `yaml:"source_patterns"`
}

func Default() *Config {
	return &Config{
		Provider:       "claude-cli",
		Model:          "opus",
		MaxConcurrency: 4,
	}
}

func Load(path string) (*Config, error) {
	if path == "" {
		var err error
		path, err = find()
		if err != nil {
			return nil, err
		}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}

	if cfg.Provider == "" {
		cfg.Provider = "claude-cli"
	}
	if cfg.Model == "" {
		cfg.Model = "opus"
	}
	if cfg.MaxConcurrency <= 0 {
		cfg.MaxConcurrency = 4
	}

	return &cfg, nil
}

func WriteDefault(path string) error {
	if _, err := os.Stat(path); err == nil {
		return fmt.Errorf("config file already exists: %s", path)
	}

	data, err := yaml.Marshal(Default())
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}

	return os.WriteFile(path, data, 0644)
}

func find() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("getting working directory: %w", err)
	}

	for {
		path := filepath.Join(dir, "vex.yaml")
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("vex.yaml not found (searched up to filesystem root)")
		}
		dir = parent
	}
}

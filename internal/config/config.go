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

// Save writes the config to the given path.
func Save(path string, cfg *Config) error {
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// AddLanguage adds or updates a language in the config file at path.
// If the file does not exist, a default config is created first.
func AddLanguage(path string, name string, lc LanguageConfig) error {
	cfg, err := loadOrDefault(path)
	if err != nil {
		return err
	}

	if cfg.Languages == nil {
		cfg.Languages = make(map[string]LanguageConfig)
	}
	cfg.Languages[name] = lc
	return Save(path, cfg)
}

// RemoveLanguage removes a language from the config file at path.
func RemoveLanguage(path string, name string) error {
	cfg, err := Load(path)
	if err != nil {
		return err
	}

	if _, ok := cfg.Languages[name]; !ok {
		return fmt.Errorf("language %q not found in config", name)
	}
	delete(cfg.Languages, name)
	if len(cfg.Languages) == 0 {
		cfg.Languages = nil
	}
	return Save(path, cfg)
}

func loadOrDefault(path string) (*Config, error) {
	if _, err := os.Stat(path); err != nil {
		return Default(), nil
	}
	return Load(path)
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

package spec

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

type ProjectSpec struct {
	Project     string     `yaml:"project"`
	Description string     `yaml:"description"`
	Shared      []Behavior `yaml:"shared,omitempty"`
	Sections    []Section  `yaml:"sections"`
}

type Section struct {
	Name         string       `yaml:"name"`
	Path         PathList     `yaml:"path,omitempty"`
	File         PathList     `yaml:"file,omitempty"`
	Description  string       `yaml:"description"`
	Shared       []string     `yaml:"shared,omitempty"`
	Behaviors    []Behavior   `yaml:"behaviors,omitempty"`
	Subsections  []Subsection `yaml:"subsections,omitempty"`
}

type Subsection struct {
	Name      string     `yaml:"name"`
	Path      PathList   `yaml:"path,omitempty"`
	File      PathList   `yaml:"file,omitempty"`
	Behaviors []Behavior `yaml:"behaviors,omitempty"`
}

type Behavior struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

// PathList handles path being either a string or a list of strings.
type PathList []string

func (p *PathList) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind == yaml.ScalarNode {
		*p = PathList{value.Value}
		return nil
	}
	if value.Kind == yaml.SequenceNode {
		var list []string
		if err := value.Decode(&list); err != nil {
			return err
		}
		*p = PathList(list)
		return nil
	}
	return fmt.Errorf("path must be a string or list of strings")
}

const defaultSpecPath = ".vex/vexspec.yaml"

func LoadProject(path string) (*ProjectSpec, error) {
	if path == "" {
		path = defaultSpecPath
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading spec: %w", err)
	}

	var ps ProjectSpec
	if err := yaml.Unmarshal(data, &ps); err != nil {
		return nil, fmt.Errorf("parsing spec: %w", err)
	}

	if err := ps.validate(); err != nil {
		return nil, err
	}

	return &ps, nil
}

func (ps *ProjectSpec) validate() error {
	if ps.Project == "" {
		return fmt.Errorf("spec missing required field: project")
	}
	if len(ps.Sections) == 0 {
		return fmt.Errorf("spec must have at least one section")
	}

	sharedNames := make(map[string]bool)
	for i, b := range ps.Shared {
		if b.Name == "" {
			return fmt.Errorf("shared behavior %d missing required field: name", i)
		}
		if b.Description == "" {
			return fmt.Errorf("shared behavior %q missing required field: description", b.Name)
		}
		if sharedNames[b.Name] {
			return fmt.Errorf("duplicate shared behavior: %q", b.Name)
		}
		sharedNames[b.Name] = true
	}

	for _, sec := range ps.Sections {
		if sec.Name == "" {
			return fmt.Errorf("section missing required field: name")
		}

		for _, ref := range sec.Shared {
			if !sharedNames[ref] {
				return fmt.Errorf("section %q references unknown shared behavior: %q", sec.Name, ref)
			}
		}

		for i, b := range sec.Behaviors {
			if err := validateBehavior(b, i, sec.Name); err != nil {
				return err
			}
		}

		for _, sub := range sec.Subsections {
			if sub.Name == "" {
				return fmt.Errorf("subsection in %q missing required field: name", sec.Name)
			}
			if len(sub.Path) > 0 && len(sub.File) > 0 {
				return fmt.Errorf("subsection %q in %q has both path and file; use one or the other", sub.Name, sec.Name)
			}

			for i, b := range sub.Behaviors {
				if err := validateBehavior(b, i, sub.Name); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func validateBehavior(b Behavior, index int, context string) error {
	if b.Name == "" {
		return fmt.Errorf("behavior %d in %q missing required field: name", index, context)
	}
	if b.Description == "" {
		return fmt.Errorf("behavior %q in %q missing required field: description", b.Name, context)
	}
	return nil
}

// AllBehaviors returns all behaviors for a section, including resolved shared behaviors.
func (ps *ProjectSpec) AllBehaviors(sec *Section) []Behavior {
	var all []Behavior

	for _, ref := range sec.Shared {
		for _, sb := range ps.Shared {
			if sb.Name == ref {
				all = append(all, sb)
			}
		}
	}

	all = append(all, sec.Behaviors...)

	for _, sub := range sec.Subsections {
		all = append(all, sub.Behaviors...)
	}

	return all
}

// SectionPaths returns directories from path: entries to walk for file discovery.
func SectionPaths(sec *Section) []string {
	seen := make(map[string]bool)
	var paths []string

	add := func(p string) {
		if !seen[p] {
			paths = append(paths, p)
			seen[p] = true
		}
	}

	for _, p := range sec.Path {
		add(p)
	}

	for _, sub := range sec.Subsections {
		for _, p := range sub.Path {
			add(p)
		}
	}

	return paths
}

// SectionFiles returns explicit file paths from file: entries.
func SectionFiles(sec *Section) []string {
	seen := make(map[string]bool)
	var files []string

	add := func(f string) {
		if !seen[f] {
			files = append(files, f)
			seen[f] = true
		}
	}

	for _, f := range sec.File {
		add(f)
	}

	for _, sub := range sec.Subsections {
		for _, f := range sub.File {
			add(f)
		}
	}

	return files
}

// SectionAllPaths returns all directories relevant to a section, combining
// path: entries and parent directories of file: entries. Used for drift detection.
func SectionAllPaths(sec *Section) []string {
	seen := make(map[string]bool)
	var paths []string

	add := func(p string) {
		if !seen[p] {
			paths = append(paths, p)
			seen[p] = true
		}
	}

	for _, p := range sec.Path {
		add(p)
	}
	for _, f := range sec.File {
		add(filepath.Dir(f))
	}

	for _, sub := range sec.Subsections {
		for _, p := range sub.Path {
			add(p)
		}
		for _, f := range sub.File {
			add(filepath.Dir(f))
		}
	}

	return paths
}

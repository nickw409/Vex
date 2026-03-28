package perf

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestStartAndSpans(t *testing.T) {
	p := New()
	end := p.Start("test-op", "section-a")
	time.Sleep(5 * time.Millisecond)
	end()

	spans := p.Spans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if spans[0].Name != "test-op" {
		t.Errorf("expected name test-op, got %s", spans[0].Name)
	}
	if spans[0].Parent != "section-a" {
		t.Errorf("expected parent section-a, got %s", spans[0].Parent)
	}
	if spans[0].Duration < 1.0 {
		t.Errorf("expected duration >= 1ms, got %.3f ms", spans[0].Duration)
	}
}

func TestConcurrentSpans(t *testing.T) {
	p := New()
	done := make(chan struct{})
	for i := 0; i < 10; i++ {
		go func() {
			end := p.Start("concurrent", "")
			time.Sleep(time.Millisecond)
			end()
			done <- struct{}{}
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
	if len(p.Spans()) != 10 {
		t.Fatalf("expected 10 spans, got %d", len(p.Spans()))
	}
}

func TestWriteFile(t *testing.T) {
	p := New()
	end := p.Start("write-test", "")
	end()

	path := filepath.Join(t.TempDir(), "profile.json")
	if err := p.WriteFile(path); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading profile: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("profile file is empty")
	}
}

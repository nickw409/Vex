VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT  ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
DATE    ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

LDFLAGS = -s -w \
	-X github.com/nickw409/vex/internal/version.Version=$(VERSION) \
	-X github.com/nickw409/vex/internal/version.Commit=$(COMMIT) \
	-X github.com/nickw409/vex/internal/version.Date=$(DATE)

DIST = dist

.PHONY: build install clean test release publish bench bench-diff bench-save

build:
	go build -ldflags "$(LDFLAGS)" -o vex ./cmd/vex/

install:
	go install -ldflags "$(LDFLAGS)" ./cmd/vex/

test:
	go test ./...

release: clean
	@mkdir -p $(DIST)
	@for pair in linux/amd64 linux/arm64 darwin/amd64 darwin/arm64; do \
		os=$${pair%%/*}; arch=$${pair##*/}; \
		name=vex_$${VERSION#v}_$${os}_$${arch}; \
		echo "Building $${name}..."; \
		GOOS=$$os GOARCH=$$arch go build -ldflags "$(LDFLAGS)" -o $(DIST)/vex ./cmd/vex/; \
		tar -czf $(DIST)/$${name}.tar.gz -C $(DIST) vex; \
		rm $(DIST)/vex; \
	done

bench: build
	bench/run.sh

bench-diff: build
	bench/run.sh --diff

bench-save:
	cp bench/profile.json bench/baseline.json
	@echo "Baseline saved."

clean:
	rm -f vex
	rm -rf $(DIST)

# Full release workflow: test, tag, build, publish, update Go proxy.
# Usage: make publish VERSION=v1.5.0 NOTES="Release notes here"
NOTES ?= Release $(VERSION)

publish:
	@if [ "$(VERSION)" = "" ] || echo "$(VERSION)" | grep -q "dev"; then \
		echo "ERROR: set VERSION=vX.Y.Z"; exit 1; \
	fi
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "ERROR: working tree is dirty — commit first"; exit 1; \
	fi
	@echo "==> Running tests..."
	go test ./...
	@echo "==> Tagging $(VERSION)..."
	git tag $(VERSION)
	@echo "==> Pushing to origin..."
	git push origin main $(VERSION)
	@echo "==> Building release binaries..."
	$(MAKE) VERSION=$(VERSION) release
	@echo "==> Creating GitHub release..."
	gh release create $(VERSION) $(DIST)/*.tar.gz --title "$(VERSION)" --notes "$(NOTES)"
	@echo "==> Updating Go module proxy..."
	GOPROXY=proxy.golang.org go list -m github.com/nickw409/vex@$(VERSION)
	@echo "==> Done: $(VERSION) published"

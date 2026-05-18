BUILD_DIR := build
BUILD_TYPE ?= Release

# ── Configuration ─────────────────────────────────────────────────────────────

.PHONY: configure
configure:
	cmake -B $(BUILD_DIR) \
	      -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# ── Build ─────────────────────────────────────────────────────────────────────

.PHONY: build
build:
	cmake --build $(BUILD_DIR) --parallel -- --no-print-directory

# Build a single target, e.g.:  make target TARGET=vector_add
.PHONY: target
target:
	cmake --build $(BUILD_DIR) --target $(TARGET) -- --no-print-directory

# ── Run ───────────────────────────────────────────────────────────────────────

# Run any binary by name, e.g.:  make run EX=vector_add
.PHONY: run
run:
	@bin=$$(find $(BUILD_DIR) -name "$(EX)" -type f | head -1); \
	if [ -z "$$bin" ]; then echo "Binary '$(EX)' not found. Run 'make build' first."; exit 1; fi; \
	echo "Running $$bin"; \
	$$bin

# ── Format ────────────────────────────────────────────────────────────────────

.PHONY: format
format:
	find . -path ./$(BUILD_DIR) -prune -o \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" \) \
	       -print | xargs clang-format -i --style=file

.PHONY: format-check
format-check:
	find . -path ./$(BUILD_DIR) -prune -o \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" \) \
	       -print | xargs clang-format --dry-run --Werror --style=file

# ── Install ───────────────────────────────────────────────────────────────────

# Install device_query to ~/.local/bin so it's available on PATH
.PHONY: install-device-query
install-device-query:
	cmake --build $(BUILD_DIR) --target device_query -- --no-print-directory
	cmake --install $(BUILD_DIR) --component device_query --prefix $(HOME)/.local
	@echo "Installed → $(HOME)/.local/bin/device_query"

# ── Clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) .cache

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make configure          		Configure cmake (BUILD_TYPE=Debug|Release)"
	@echo "  make build              		Build all targets"
	@echo "  make target TARGET=foo  		Build a single target"
	@echo "  make run    EX=foo      		Run a built binary"
	@echo "  make format             		Auto-format all .cu/.h files"
	@echo "  make format-check       		Check formatting without modifying"
	@echo "  make clean              		Remove build directory"
	@echo "  make install-device-query		Build and install the device query utility"
	@echo ""
	@echo "Examples:"
	@echo "  make configure BUILD_TYPE=Release"
	@echo "  make target TARGET=vector_add"
	@echo "  make run EX=vector_add"

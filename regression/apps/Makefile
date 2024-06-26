# SPDX-License-Identifier: MPL-2.0

MAKEFLAGS += --no-builtin-rules # Prevent the implicit rules from compiling ".c" or ".s" files automatically.

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))

INITRAMFS ?= $(CUR_DIR)/../build/initramfs
REGRESSION_BUILD_DIR ?= $(INITRAMFS)/regression

# These test apps are sorted by name
TEST_APPS := \
	execve \
	fork \
	fork_c \
	getpid \
	hello_c \
	hello_pie \
	hello_world \
	mongoose \
	network \
	pthread \
	pty \
	signal_c \

C_SOURCES := $(shell find . -type f \( -name "*.c" -or -name "*.h" \) )

.PHONY: all
all: $(TEST_APPS) scripts

.PHONY: $(TEST_APPS)
$(TEST_APPS):
	@make --no-print-directory -C $@

.PHONY: format
format:
	@echo "Fixing code format for regression tests..."
	@clang-format -i $(C_SOURCES)

.PHONY: check
check:
	@echo "Checking code format for regression tests..."
	@clang-format --dry-run --Werror $(C_SOURCES)

$(REGRESSION_BUILD_DIR):
	@mkdir -p $@

.PHONY: scripts
scripts: | $(REGRESSION_BUILD_DIR)
	@make --no-print-directory BUILD_DIR=$(REGRESSION_BUILD_DIR) -C scripts

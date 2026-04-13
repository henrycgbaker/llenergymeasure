.PHONY: format lint lint-fix typecheck check test test-unit test-integration test-all install dev clean
.PHONY: test-runtime test-runtime-vllm test-runtime-tensorrt test-runtime-all
.PHONY: test-runtime-quick test-runtime-local test-runtime-docker
.PHONY: docker-build docker-build-all docker-build-transformers docker-build-vllm docker-build-tensorrt docker-seed-transformers
.PHONY: docker-build-dev docker-check docker-builder-setup docker-builder-rm
.PHONY: experiment datasets validate docker-shell docker-dev
.PHONY: setup docker-setup lem-clean lem-clean-all lem-clean-state lem-clean-cache lem-clean-trt generate-docs check-docs
.PHONY: package-check docs-check docker-smoke docker-smoke-pytorch docker-smoke-vllm ci ci-all ci-docker
.PHONY: gpu-ci gpu-ci-pytorch gpu-ci-vllm

# PUID/PGID for correct file ownership on bind mounts (LinuxServer.io pattern)
export PUID := $(shell id -u)
export PGID := $(shell id -g)

# Host/container schema handshake stamps — surfaced to docker compose build
# via docker-compose.yml's build.args block so every locally-built image is
# labelled with the same fingerprint llem computes at runtime. Falls back to
# "dev"/"unknown" on any error (e.g. missing venv).
export LLEM_PKG_VERSION := $(shell python3 -c "from llenergymeasure._version import __version__; print(__version__)" 2>/dev/null || echo dev)
export LLEM_EXPCONF_SCHEMA_FINGERPRINT := $(shell python3 scripts/compute_expconf_fingerprint.py 2>/dev/null || echo unknown)

# =============================================================================
# Quick Start
#   Local:  make setup       (pip install + pre-commit)
#   Docker: make docker-setup (above + docker compose build)
#   Dev:    make dev          (poetry install + pre-commit)
# =============================================================================

setup:
	pip install -e ".[dev]"
	pre-commit install
	@echo "Dev environment ready. Run: lem --help"

docker-setup: setup
	docker compose build
	@echo "Docker environment ready. Run: llem run <config.yaml>"
	@echo "Tip: run 'make docker-builder-setup' for a BuildKit builder with larger cache limits"

# =============================================================================
# Local Development
# =============================================================================

format:
	uv run ruff format src/ tests/

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run lint-imports

lint-fix:
	uv run ruff check src/ tests/ --fix
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

check: lint typecheck

test:
	uv run pytest tests/ -m "not gpu and not docker" -x -q --tb=short

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-all:
	uv run pytest tests/ -v --ignore=tests/runtime/

# Runtime tests with Docker container dispatch
# These tests use SSOT introspection to discover ALL params, then dispatch
# each test to the correct engine container (pytorch, vllm, tensorrt).
# Uses the same dispatch pattern as `lem campaign`.

test-runtime:
	python scripts/runtime-test-orchestrator.py --engine pytorch

test-runtime-vllm:
	python scripts/runtime-test-orchestrator.py --engine vllm

test-runtime-tensorrt:
	python scripts/runtime-test-orchestrator.py --engine tensorrt

# Run all engines - discovers params via SSOT, dispatches to correct containers
test-runtime-all:
	python scripts/runtime-test-orchestrator.py --engine all

test-runtime-quick:
	python scripts/runtime-test-orchestrator.py --engine pytorch --quick

# Check Docker setup and list params without running
test-runtime-check:
	python scripts/runtime-test-orchestrator.py --check-docker

test-runtime-list:
	python scripts/runtime-test-orchestrator.py --list-params

# Build missing images automatically before running
test-runtime-docker:
	python scripts/runtime-test-orchestrator.py --engine pytorch --build

install:
	uv sync

dev:
	uv sync --dev
	uv run pre-commit install

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist/ build/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Generate documentation from SSOT sources
generate-docs:
	python scripts/generate_invalid_combos_doc.py
	python scripts/generate_param_matrix.py
	python scripts/generate_config_docs.py
	@echo "Generated docs in docs/generated/"

# Check if generated docs are stale (CI validation)
check-docs: docs-check

docs-check:
	@uv run python scripts/generate_config_docs.py > /dev/null
	@uv run python scripts/generate_cli_reference.py > /dev/null
	@uv run python scripts/generate_invalid_combos_doc.py > /dev/null
	@echo "Generated docs are up to date"

# Build wheel + validate package install + check version consistency
package-check:
	uv build --wheel
	@python3 -m venv /tmp/pkg-check-local 2>/dev/null || true
	@/tmp/pkg-check-local/bin/pip install dist/*.whl --quiet --force-reinstall
	@/tmp/pkg-check-local/bin/python -c "from llenergymeasure import run_experiment, ExperimentConfig, ExperimentResult; print('Package install OK')"
	@PYPROJECT_VER=$$(python3 -c "import tomllib; f=open('pyproject.toml','rb'); print(tomllib.load(f)['project']['version'])"); \
	 VERSION_VER=$$(python3 -c "import re; s=open('src/llenergymeasure/_version.py').read(); print(re.search(r'__version__[^=]*=\s*\"([^\"]+)\"', s).group(1))"); \
	 echo "pyproject.toml: $$PYPROJECT_VER"; \
	 echo "_version.py:    $$VERSION_VER"; \
	 [ "$$PYPROJECT_VER" = "$$VERSION_VER" ] || { echo "ERROR: Version mismatch"; exit 1; }
	@echo "Package validation OK"

# Docker smoke tests — mirrors CI docker-smoke job
docker-smoke: docker-smoke-pytorch docker-smoke-vllm

docker-smoke-pytorch:
	docker build -f docker/Dockerfile.pytorch --build-arg INSTALL_FA3=false . -t smoke-pytorch
	docker run --rm smoke-pytorch llem --version
	docker run --rm smoke-pytorch llem config

docker-smoke-vllm:
	docker build -f docker/Dockerfile.vllm --build-arg INSTALL_FA3=false . -t smoke-vllm
	docker run --rm smoke-vllm llem --version
	docker run --rm smoke-vllm llem config

# CI targets — run the same checks as GitHub Actions
ci: lint typecheck test package-check docs-check

ci-all: ci docker-smoke

# Run CI in a clean container matching GitHub Actions (ubuntu + Python 3.12 + uv)
# Catches "works on my machine" issues before pushing
CI_IMAGE := llenergymeasure-ci-env:local
define CI_DOCKERFILE
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY . .
ENV UV_FROZEN=true UV_NO_PROGRESS=1
RUN uv sync --dev --extra pytorch --extra codecarbon --extra zeus
endef
export CI_DOCKERFILE
ci-docker:
	echo "$$CI_DOCKERFILE" | docker build -t $(CI_IMAGE) -f - .
	docker run --rm $(CI_IMAGE) sh -c '\
		uv run ruff check src/ tests/ && \
		uv run ruff format --check src/ tests/ && \
		uv run lint-imports && \
		uv run mypy src/ && \
		uv run pytest tests/ -m "not gpu and not docker" -x -q --tb=short && \
		echo "=== CI-docker: all checks passed ==="'
	@docker rmi $(CI_IMAGE) 2>/dev/null || true

# GPU CI targets — mirrors .github/workflows/gpu-ci.yml
# Requires: Docker, NVIDIA GPUs, nvidia-container-toolkit
gpu-ci: gpu-ci-pytorch gpu-ci-vllm

gpu-ci-pytorch:
	docker build -f docker/Dockerfile.pytorch -t llenergymeasure-ci:pytorch .
	docker run --name llem-ci-setup llenergymeasure-ci:pytorch pip install --no-cache-dir pytest pytest-xdist
	docker commit llem-ci-setup llenergymeasure-ci:pytorch
	docker rm llem-ci-setup
	mkdir -p results/
	docker run --rm --gpus all \
		-v "$(CURDIR)/tests":/app/tests:ro \
		-v "$(CURDIR)/results":/app/results \
		llenergymeasure-ci:pytorch \
		python3 -m pytest tests/ -v --tb=short -o "addopts="
	docker run --rm --gpus all \
		-v "$(CURDIR)/tests":/app/tests:ro \
		-v "$(CURDIR)/results":/app/results \
		llenergymeasure-ci:pytorch \
		bash tests/integration/sigint_verify.sh
	docker rmi llenergymeasure-ci:pytorch 2>/dev/null || true

gpu-ci-vllm:
	docker build -f docker/Dockerfile.vllm -t llenergymeasure-ci:vllm .
	docker run --name llem-vllm-ci-setup llenergymeasure-ci:vllm pip install --no-cache-dir pytest pytest-xdist
	docker commit llem-vllm-ci-setup llenergymeasure-ci:vllm
	docker rm llem-vllm-ci-setup
	mkdir -p results/
	docker run --rm --gpus all \
		-v "$(CURDIR)/tests":/app/tests:ro \
		-v "$(CURDIR)/results":/app/results \
		llenergymeasure-ci:vllm \
		bash tests/integration/killpg_verify.sh
	docker rmi llenergymeasure-ci:vllm 2>/dev/null || true

# =============================================================================
# Docker Commands (Production)
# =============================================================================


# Builder name used by COMPOSE_BAKE for registry-cached builds
BUILDER_NAME := llem-builder

# Create the BuildKit builder with tuned GC limits (200 GiB).
# Idempotent — skips if builder already exists.
docker-builder-setup:
	@if docker buildx inspect $(BUILDER_NAME) >/dev/null 2>&1; then \
		echo "Builder '$(BUILDER_NAME)' already exists"; \
	else \
		echo "Creating builder '$(BUILDER_NAME)' with 200 GiB cache limit..."; \
		docker buildx create \
			--name $(BUILDER_NAME) \
			--driver docker-container \
			--buildkitd-config docker/buildkitd.toml \
			--bootstrap; \
		echo "Builder '$(BUILDER_NAME)' created. Use with: BUILDX_BUILDER=$(BUILDER_NAME) docker compose build"; \
	fi

# Remove the builder (e.g. to recreate with new config)
docker-builder-rm:
	docker buildx rm $(BUILDER_NAME) 2>/dev/null || true

CACHE_HINT := @echo "First build pulls cache layers from ghcr.io; warm rebuilds < 5 min."
BUILD_WITH_REPORT := scripts/docker_build_with_cache_report.sh

# Build all engines (transformers, vllm, tensorrt) — local images.
# Calls compose directly so all three can build in parallel;
# per-engine cache-import summary is only emitted for single-engine targets
# below. For per-engine diagnostics, run `make docker-build-{engine}`.
docker-build-all:
	$(CACHE_HINT)
	BUILDKIT_PROGRESS=$${BUILDKIT_PROGRESS:-plain} docker compose build transformers vllm tensorrt

# Build Transformers engine (default, recommended for most users)
docker-build-transformers:
	$(CACHE_HINT)
	$(BUILD_WITH_REPORT) transformers

# Build specific engines — local images
docker-build-vllm:
	$(CACHE_HINT)
	$(BUILD_WITH_REPORT) vllm

docker-build-tensorrt:
	$(CACHE_HINT)
	$(BUILD_WITH_REPORT) tensorrt

# Seed GHCR build cache from a local machine with sufficient RAM.
# Intended for seeding the Transformers image cache (FA3 Hopper compile,
# ~30 min but memory-intensive) when the CI hosted runner cannot complete
# the build. Requires: docker login ghcr.io, llem-builder buildx builder.
# Uses Dockerfile default MAX_JOBS=32 — matches local layer cache so FA3
# is not recompiled if already built locally.
docker-seed-transformers:
	@version=$$(python3 -c "from llenergymeasure._version import __version__; print(__version__)" 2>/dev/null || echo "dev"); \
	fingerprint=$$(python3 scripts/compute_expconf_fingerprint.py 2>/dev/null || echo "unknown"); \
	ref=ghcr.io/henrycgbaker/llenergymeasure/transformers; \
	echo "Seeding GHCR cache for transformers (version=$$version)"; \
	docker buildx build \
	  --builder $(BUILDER_NAME) \
	  -f docker/Dockerfile.transformers \
	  --build-arg LLEM_PKG_VERSION=$$version \
	  --build-arg LLEM_EXPCONF_SCHEMA_FINGERPRINT=$$fingerprint \
	  --cache-from type=registry,ref=$$ref:v$$version \
	  --cache-from type=registry,ref=$$ref:latest \
	  --cache-to   type=registry,ref=$$ref:latest,mode=max \
	  --push \
	  --tag $$ref:v$$version \
	  --tag $$ref:latest \
	  .

# Pull versioned registry images (ghcr.io) instead of building locally
docker-pull:
	@version=$$(python3 -c "from llenergymeasure._version import __version__; print(__version__)" 2>/dev/null || echo "latest"); \
	for engine in transformers vllm tensorrt; do \
		echo "Pulling ghcr.io/henrycgbaker/llenergymeasure/$$engine:v$$version"; \
		docker pull "ghcr.io/henrycgbaker/llenergymeasure/$$engine:v$$version"; \
	done

# Show which images llem will use (local vs registry)
docker-images:
	@python3 -c "from llenergymeasure.infra.image_registry import show_image_resolution; show_image_resolution()"

# Validate Docker setup
docker-check:
	@docker compose config -q || (echo "Error: Invalid docker-compose config"; exit 1)
	@echo "Docker config OK"

# Run any lem command in Docker
# Usage: make lem CMD="experiment configs/my_experiment.yaml"
#        make lem CMD="config validate configs/test.yaml"
#        make lem CMD="results list"
CMD ?= --help
lem: docker-check
	docker compose run --rm pytorch lem $(CMD)

# Run experiment (num_processes auto-inferred from config)
# Usage: make experiment CONFIG=test_tiny.yaml DATASET=alpaca SAMPLES=100
CONFIG ?= test_tiny.yaml
DATASET ?= alpaca
SAMPLES ?= 100
experiment: docker-check
	docker compose run --rm pytorch \
		lem experiment /app/configs/$(CONFIG) \
		--dataset $(DATASET) -n $(SAMPLES)

# List available datasets
datasets:
	docker compose run --rm pytorch lem datasets

# Validate a config file
# Usage: make validate CONFIG=test_tiny.yaml
validate: docker-check
	docker compose run --rm pytorch lem config validate /app/configs/$(CONFIG)

# Interactive shell in production container
docker-shell:
	docker compose run --rm pytorch /bin/bash

# =============================================================================
# Docker Commands (Development)
# =============================================================================

# Build the dev Docker image
docker-build-dev:
	docker compose --profile dev build pytorch-dev

# Interactive dev shell with source mounted
docker-dev:
	docker compose --profile dev run --rm pytorch-dev

# =============================================================================
# Volume Management
# =============================================================================

# Clean experiment state volume (preserves caches)
lem-clean-state:
	docker volume rm lem-experiment-state 2>/dev/null || true
	@echo "Cleared experiment state"

# Clean HuggingFace cache volume (will need to re-download models)
lem-clean-cache:
	docker volume rm lem-hf-cache 2>/dev/null || true
	@echo "Cleared HuggingFace cache"

# Clean TensorRT engine cache
lem-clean-trt:
	docker volume rm lem-trt-engine-cache 2>/dev/null || true
	@echo "Cleared TensorRT engine cache"

# Clean all named volumes (state + caches)
lem-clean-all:
	docker volume rm lem-experiment-state lem-hf-cache lem-trt-engine-cache 2>/dev/null || true
	@echo "Cleared all LEM volumes"

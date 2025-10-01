# Developer-friendly shortcuts for model provisioner tests
SHELL := /usr/bin/env bash
ROOT := $(abspath $(CURDIR)/../../..)
CRATE := $(ROOT)/libs/provisioners/model-provisioner
IMAGE := llorch-modelprov-e2e
DOCKERFILE := $(CRATE)/docker/Dockerfile.e2e

.PHONY: docker-build docker-rebuild setup-buildx \
	e2e-real-docker e2e-real-docker-rebuild docker-prune docker-prune-dangling docker-reset

# Build Docker image with BuildKit
docker-build:
	@docker buildx build --file $(DOCKERFILE) -t $(IMAGE) --load $(ROOT)

# Rebuild the image, pulling latest base and without using local cache
docker-rebuild:
	@docker buildx build --pull --no-cache --file $(DOCKERFILE) -t $(IMAGE) --load $(ROOT)

# Setup BuildKit/buildx on host (will use sudo when needed)
setup-buildx:
	@bash $(CRATE)/scripts/setup_buildx.sh

# Real HF E2E inside Docker (downloads tiny GGUF inside container)
# Optional envs: MODEL_REPO, MODEL_FILE
#   make e2e-real-docker MODEL_REPO=TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF MODEL_FILE=tinyllama-1.1b-chat-v1.0.Q2_K.gguf
e2e-real-docker: docker-build
	@bash $(CRATE)/scripts/run_real_e2e.sh "$(IMAGE)"

# Rebuild image (no cache), run E2E, then prune dangling images
e2e-real-docker-rebuild: docker-rebuild
	@bash $(CRATE)/scripts/run_real_e2e.sh "$(IMAGE)"
	@docker image prune -f || true

# Remove dangling images and prune build cache to keep disk usage low
docker-prune-dangling:
	@docker image prune -f || true

docker-prune:
	@docker builder prune -af || true
	@docker image prune -af || true
	@docker container prune -f || true

# Force-remove this crate's test image and prune caches
docker-reset:
	@docker rmi -f $(IMAGE) 2>/dev/null || true
	@$(MAKE) docker-prune

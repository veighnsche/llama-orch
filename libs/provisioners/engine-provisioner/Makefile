# Developer-friendly shortcuts for engine provisioner tests
SHELL := /usr/bin/env bash
ROOT := $(abspath $(CURDIR)/../../..)
CRATE := $(ROOT)/libs/provisioners/engine-provisioner
IMAGE := llorch-llamacpp-e2e-cpu
DOCKERFILE := $(CRATE)/docker/Dockerfile.cpu.e2e

.PHONY: e2e-fixture e2e-real-host docker-build e2e-real-docker setup-buildx \
	docker-rebuild docker-prune docker-prune-dangling docker-reset \
	e2e-real-docker-rebuild

# 1) Hermetic fixture E2E (fast/offline)
e2e-fixture:
	@cd $(ROOT) && \
	LLORCH_E2E_FIXTURE=1 cargo test -p provisioners-engine-provisioner \
	  --test llamacpp_fixture_cpu_e2e -- --ignored --nocapture

# 2) Real llama.cpp E2E on host (requires LLORCH_E2E_MODEL_PATH)
e2e-real-host:
	@cd $(ROOT) && \
	test -n "$$LLORCH_E2E_MODEL_PATH" || (echo "LLORCH_E2E_MODEL_PATH not set"; exit 1)
	@cd $(ROOT) && \
	LLORCH_E2E_REAL=1 cargo test -p provisioners-engine-provisioner \
	  --test llamacpp_source_cpu_real_e2e -- --ignored --nocapture

# 3) Build Docker image with BuildKit
docker-build:
	@docker buildx build --file $(DOCKERFILE) -t $(IMAGE) --load $(ROOT)

# Rebuild the image, pulling latest base and without using local cache
docker-rebuild:
	@docker buildx build --pull --no-cache --file $(DOCKERFILE) -t $(IMAGE) --load $(ROOT)

# 4) Setup BuildKit/buildx on host (will use sudo when needed)
setup-buildx:
	@$(CRATE)/scripts/setup_buildx.sh

# 5) Real llama.cpp E2E inside Docker (downloads tiny GGUF inside container)
# Optional envs: LLAMA_REF, MODEL_REPO, MODEL_PATTERN
#   make e2e-real-docker LLAMA_REF=master MODEL_REPO=TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF MODEL_PATTERN='*Q2_K.gguf'
e2e-real-docker: docker-build
	@$(CRATE)/scripts/run_real_e2e.sh "$(IMAGE)"

# Rebuild image (no cache), run E2E, then prune dangling images
e2e-real-docker-rebuild: docker-rebuild
	@$(CRATE)/scripts/run_real_e2e.sh "$(IMAGE)"
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

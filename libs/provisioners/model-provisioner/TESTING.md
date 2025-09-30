# Model Provisioner: How to test

Two tiers of tests are available:

- Real HF (Docker): run the crate smoke test end-to-end in an isolated HOME and cache; the provisioner will download the model via the Hugging Face CLI.
- Host smoke (optional): run the same ignored test on host if you already have the HF CLI configured.

## 0) Prerequisites

- Rust toolchain installed.
- Docker installed. For BuildKit features, install the buildx plugin on your host.
- On Arch Linux: `sudo pacman -S docker docker-buildx` and ensure the daemon enables BuildKit.

If BuildKit is not enabled, set `/etc/docker/daemon.json` to:

```json
{ "features": { "buildkit": true } }
```

Then `sudo systemctl restart docker`.

You can set up buildx and BuildKit via the helper script:

```bash
libs/provisioners/model-provisioner/scripts/setup_buildx.sh
```

### Preflight requirements (what the provisioner checks)

The model provisioner validates the environment before invoking networked fetches and provides clear errors if missing:

- Hugging Face CLI: prefer `hf`; fallback to `huggingface-cli`.
- For host runs, ensure system CA certificates and Python are sane; otherwise, use the Docker image which provides a venv and CA bundle.

## 1) Real HF (Docker)

- What it does:
  - Builds a slim Rust image with a Python venv containing `huggingface_hub[cli]` (both `hf` and `huggingface-cli` are available).
  - Sets `MODEL_ORCH_SMOKE_REF` to a precise `hf:` ref (with explicit filename) and runs the ignored smoke test end-to-end.
  - The model provisioner performs the actual download using the HF CLI inside the test.

- Build the image (BuildKit):

```bash
# from repo root
docker buildx build \
  -f libs/provisioners/model-provisioner/docker/Dockerfile.e2e \
  -t llorch-modelprov-e2e --load .
```

- Run the test (the provisioner will download the model):

```bash
docker run --rm -it \
  -e MODEL_REPO=TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  -e MODEL_FILE=tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
  -w /workspace \
  llorch-modelprov-e2e \
  bash -lc '
    set -euo pipefail
    export MODEL_ORCH_SMOKE=1
    export MODEL_ORCH_SMOKE_REF="hf:${MODEL_REPO}/${MODEL_FILE}"
    /usr/local/cargo/bin/cargo test -p model-provisioner --test hf_smoke -- --ignored --nocapture
  '
```

- Makefile shortcuts:

```bash
# from this crate directory (libs/provisioners/model-provisioner/)
make setup-buildx
make docker-build
make e2e-real-docker MODEL_REPO=TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF MODEL_FILE=tinyllama-1.1b-chat-v1.0.Q2_K.gguf

# Fast path: rebuild image (no cache), run E2E, then prune dangling layers
make e2e-real-docker-rebuild MODEL_REPO=... MODEL_FILE=tinyllama-1.1b-chat-v1.0.Q2_K.gguf
```

## 2) Host smoke (optional)

If you already have the HF CLI configured locally and want to run without Docker:

```bash
MODEL_ORCH_SMOKE=1 \
MODEL_ORCH_SMOKE_REF=hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat.Q2_K.gguf \
cargo test -p model-provisioner --test hf_smoke -- --ignored --nocapture
```

## 3) Disk hygiene

- Use a fixed image tag (`llorch-modelprov-e2e`) to avoid proliferation.
- Makefile targets:
  - `docker-prune-dangling`: removes dangling layers
  - `docker-prune`: prunes builder cache, images, and containers
  - `docker-reset`: force-removes this crate image and prunes caches

## What we learned (from real runs)

- The HF CLI moved to `hf download`. Docs and scripts prefer `hf`, with compatibility for `huggingface-cli`.
- Tests should not install global tools; the provisioner preflight should explain missing deps clearly. The Docker image provides a known-good environment.
- Cargo visibility inside non-login shells can be flaky; the image symlinks cargo/rustc/rustup into `/usr/local/bin` and sets PATH explicitly.

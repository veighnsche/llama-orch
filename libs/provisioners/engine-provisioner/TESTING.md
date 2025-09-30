# Engine Provisioner: How to test

Two tiers of tests are available:

- Hermetic (fast, offline): build a tiny C fixture and validate end-to-end behavior without network or installs.
- Real llama.cpp (CPU) in Docker: clone/build `ggml-org/llama.cpp`, download a tiny GGUF, launch `llama-server`, validate health + handoff.

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
scripts/docker/setup_buildx.sh
```

## 1) Hermetic CPU fixture (fast/offline)

- What it does:
  - Creates a local git repo with a tiny C HTTP server, builds it via CMake, launches on a free port.
  - Writes handoff to `.runtime/engines/llamacpp.json` and validates the model path is readable.
- Run:

```bash
# from repo root
LLORCH_E2E_FIXTURE=1 \
  cargo test -p provisioners-engine-provisioner \
    --test llamacpp_fixture_cpu_e2e -- --ignored --nocapture
```

- Makefile shortcut:

```bash
# from this crate directory (libs/provisioners/engine-provisioner/)
make e2e-fixture
```

## 2) Real llama.cpp CPU (Docker)

- What it does:
  - Builds a CPU-only environment image with git/cmake/gcc and Hugging Face CLI.
  - Downloads a small GGUF model (TinyLlama Q2_K by default) inside the container.
  - Clones and builds `ggml-org/llama.cpp` at `LLAMA_REF` (default master).
  - Launches server, verifies `/health`, validates handoff JSON.

- Build the image (BuildKit):

```bash
# from repo root
docker buildx build -f Dockerfile.llamacpp.cpu -t llorch-llamacpp-e2e-cpu --load .
```

- Run the test:

```bash
docker run --rm -it \
  -e LLORCH_E2E_REAL=1 \
  -e LLAMA_REF=master \
  -w /workspace \
  llorch-llamacpp-e2e-cpu \
  bash -lc '
    set -euo pipefail
    mkdir -p /models
    hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
      --repo-type model \
      --include "*Q2_K.gguf" \
      -d /models
    export LLORCH_E2E_MODEL_PATH="$(ls /models/*Q2_K.gguf | head -n1)"
    echo "Using model: $LLORCH_E2E_MODEL_PATH"
    cargo test -p provisioners-engine-provisioner \
      --test llamacpp_source_cpu_real_e2e -- --ignored --nocapture
  '
```

- Makefile shortcuts:

```bash
# from this crate directory (libs/provisioners/engine-provisioner/)
make docker-build
make e2e-real-docker LLAMA_REF=master MODEL_REPO=TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF MODEL_PATTERN='*Q2_K.gguf'
```

## 3) Real llama.cpp on host (no Docker)

If you already have a small GGUF locally, you can run the real test on host:

```bash
LLORCH_E2E_REAL=1 \
LLORCH_E2E_MODEL_PATH=/absolute/path/to/model.gguf \
LLAMA_REF=master \
cargo test -p provisioners-engine-provisioner \
  --test llamacpp_source_cpu_real_e2e -- --ignored --nocapture
```

## 4) Troubleshooting

- Buildx "default is reserved": create a differently named builder, e.g. `docker buildx create --use --name llorch`.
- BuildKit deprecation/legacy builder: install docker-buildx and enable BuildKit in `daemon.json` as above.
- PEP 668 error for Python/pip: the Docker image uses a venv under `/opt/hf`; do not install system-wide.

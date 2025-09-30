#!/usr/bin/env bash
set -euo pipefail

# Run the real llama.cpp CPU E2E inside the given image.
# Usage:
#   scripts/docker/run_real_e2e.sh <image>
# Env overrides:
#   LLAMA_REF        (default: master)
#   MODEL_REPO       (default: TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
#   MODEL_PATTERN    (default: *Q2_K.gguf)

IMAGE="${1:-}"
if [[ -z "${IMAGE}" ]]; then
  echo "usage: $0 <image>" >&2
  exit 2
fi

LLAMA_REF="${LLAMA_REF:-master}"
MODEL_REPO="${MODEL_REPO:-TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF}"
MODEL_PATTERN="${MODEL_PATTERN:-*Q2_K.gguf}"

exec docker run --rm -it \
  -e LLORCH_E2E_REAL=1 \
  -e LLAMA_REF="${LLAMA_REF}" \
  -e MODEL_REPO="${MODEL_REPO}" \
  -e MODEL_PATTERN="${MODEL_PATTERN}" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    mkdir -p /models
    echo "[run_real_e2e] downloading from ${MODEL_REPO} pattern ${MODEL_PATTERN}"
    hf download "${MODEL_REPO}" \
      --repo-type model \
      --include "${MODEL_PATTERN}" \
      --local-dir /models
    export LLORCH_E2E_MODEL_PATH="$(ls /models/*.gguf | head -n1)"
    echo "[run_real_e2e] Using model: ${LLORCH_E2E_MODEL_PATH}"
    cargo test -p provisioners-engine-provisioner \
      --test llamacpp_source_cpu_real_e2e -- --ignored --nocapture
  '

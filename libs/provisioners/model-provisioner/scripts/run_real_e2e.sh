#!/usr/bin/env bash
set -euo pipefail

# Run the real HF E2E inside the given image (crate-local).
# Usage:
#   libs/provisioners/model-provisioner/scripts/run_real_e2e.sh <image>
# Env overrides:
#   MODEL_REPO       (default: TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
#   MODEL_FILE       (default: tinyllama-1.1b-chat-v1.0.Q2_K.gguf)

IMAGE="${1:-}"
if [[ -z "${IMAGE}" ]]; then
  echo "usage: $0 <image>" >&2
  exit 2
fi

MODEL_REPO="${MODEL_REPO:-TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF}"
MODEL_FILE="${MODEL_FILE:-tinyllama-1.1b-chat-v1.0.Q2_K.gguf}"

exec docker run --rm -it \
  -e MODEL_REPO="${MODEL_REPO}" \
  -e MODEL_FILE="${MODEL_FILE}" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    export MODEL_ORCH_SMOKE=1
    export MODEL_ORCH_SMOKE_REF="hf:${MODEL_REPO}/${MODEL_FILE}"
    echo "[run_real_e2e] Using MODEL_ORCH_SMOKE_REF=${MODEL_ORCH_SMOKE_REF}"
    /usr/local/cargo/bin/cargo test -p model-provisioner \
      --test hf_smoke -- --ignored --nocapture
  '

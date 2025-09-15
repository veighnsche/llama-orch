#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${HOME}/.cache/models"
MODEL_FILE="${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"

if ! command -v llama-server >/dev/null 2>&1; then
  echo "llama-server not found on PATH" >&2
  exit 1
fi

if [ ! -f "${MODEL_FILE}" ]; then
  echo "Model file not found at ${MODEL_FILE}. Run ci/scripts/fetch_model.sh first." >&2
  exit 1
fi

exec llama-server \
  --model "${MODEL_FILE}" \
  --host "${HOST}" --port "${PORT}" \
  --metrics --no-webui \
  --parallel 1 --no-cont-batching

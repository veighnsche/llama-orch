#!/usr/bin/env bash
set -euo pipefail

: "${HF_HUB_ENABLE_HF_TRANSFER:=1}"
MODEL_DIR="${HOME}/.cache/models"
mkdir -p "${MODEL_DIR}"

# Default model: Qwen2.5-0.5B-Instruct GGUF Q4_K_M
REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"

if command -v huggingface-cli >/dev/null 2>&1; then
  HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER} \
  huggingface-cli download "${REPO}" "${FILE}" \
    --local-dir "${MODEL_DIR}" --local-dir-use-symlinks False
else
  echo "huggingface-cli not found; please install to fetch models" >&2
  exit 0
fi

echo "Model cached at ${MODEL_DIR}/${FILE}"

#!/usr/bin/env bash
# TEAM-023: DEPRECATED - huggingface-cli is deprecated, use `hf` CLI instead!
# If this script uses huggingface-cli, replace with 'hf' command
# Install: pip install huggingface_hub[cli]
set -euo pipefail

: "${HF_HUB_ENABLE_HF_TRANSFER:=1}"
MODEL_DIR="${HOME}/.cache/models"
# TEAM FREE [Review]
# Category: Error handling
# Hypothesis: mkdir -p (line 6) failure not checked; if HOME unset or permission denied, script continues with empty MODEL_DIR.
# Evidence: No check of mkdir exit code; subsequent operations use potentially invalid path.
# Risk: huggingface-cli fails with cryptic error; CI fails without clear cause.
# Confidence: Medium
# Next step: Add after line 6: if [ ! -d "$MODEL_DIR" ]; then echo "Failed to create MODEL_DIR"; exit 1; fi
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
  # TEAM FREE [Review]
  # Category: Error handling
  # Hypothesis: exit 0 (line 18) on missing huggingface-cli signals success; CI thinks model fetched when it wasn't.
  # Evidence: Exit code 0 means success; caller can't distinguish "model already cached" from "tool missing".
  # Risk: CI false positive; tests run with stale/missing model; flaky failures.
  # Confidence: High
  # Next step: Change to exit 1 or add flag file to indicate fetch was skipped.
  exit 0
fi

echo "Model cached at ${MODEL_DIR}/${FILE}"

#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden by env or flags)
INPUTS=${INPUTS:-"/home/vince/Projects/llama-orch/.business/financieelplan_NL_nl/.003-draft/inputs"}
OUT=${OUT:-"/home/vince/Projects/llama-orch/.business/financieelplan_NL_nl/.003-draft/outputs"}
PIPELINES=${PIPELINES:-"public,private"}
SEED=${SEED:-"424242"}
FAIL_FLAGS=${FAIL_FLAGS:---fail-on-warning}
PY=${PY:-python3}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --inputs PATH           Inputs dir (default: ${INPUTS})
  --out PATH              Outputs dir (default: ${OUT})
  --pipelines LIST        Comma list (default: ${PIPELINES})
  --seed N                Seed integer (default: ${SEED})
  --no-fail               Do not fail on warnings
  -h, --help              Show this help and exit

Env overrides: INPUTS, OUT, PIPELINES, SEED, FAIL_FLAGS, PY
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inputs) INPUTS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --pipelines) PIPELINES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --no-fail) FAIL_FLAGS=""; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

exec ${PY} -m d3_engine.cli \
  --inputs "${INPUTS}" \
  --out "${OUT}" \
  --pipelines "${PIPELINES}" \
  --seed "${SEED}" \
  ${FAIL_FLAGS}

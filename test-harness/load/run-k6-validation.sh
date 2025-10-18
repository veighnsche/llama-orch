#!/bin/bash
# Created by: TEAM-107 | 2025-10-18
# Run k6 validation tests with mock server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/validation-results"
mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${RESULTS_DIR}/k6_validation_${TIMESTAMP}.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "🧪 k6 Validation Test Suite"
log "============================"
log ""

# Start mock server in background
log "🚀 Starting mock server..."
python3 "${SCRIPT_DIR}/mock-server.py" > "${RESULTS_DIR}/mock-server.log" 2>&1 &
MOCK_PID=$!

# Wait for server to be ready
log "⏳ Waiting for server to be ready..."
sleep 2

# Check if server is running
if ! curl -f -s http://localhost:8080/health > /dev/null; then
    log "❌ Mock server failed to start"
    kill $MOCK_PID 2>/dev/null || true
    exit 1
fi

log "✅ Mock server is ready (PID: $MOCK_PID)"
log ""

# Cleanup function
cleanup() {
    log ""
    log "🛑 Stopping mock server..."
    kill $MOCK_PID 2>/dev/null || true
    wait $MOCK_PID 2>/dev/null || true
    log "✅ Mock server stopped"
}

trap cleanup EXIT

# Run quick validation test
log "🧪 Running quick k6 validation test (30 seconds)..."
log "----------------------------------------------------"

if k6 run \
    --out json="${RESULTS_DIR}/k6_quick_${TIMESTAMP}.json" \
    --summary-export="${RESULTS_DIR}/k6_summary_${TIMESTAMP}.json" \
    "${SCRIPT_DIR}/test-k6-quick.js" 2>&1 | tee -a "${LOG_FILE}"; then
    log ""
    log "✅ Quick validation test PASSED"
    QUICK_RESULT="PASSED"
else
    log ""
    log "❌ Quick validation test FAILED"
    QUICK_RESULT="FAILED"
fi

log ""
log "============================"
log "📊 k6 Validation Summary"
log "============================"
log "Quick test: ${QUICK_RESULT}"
log ""
log "Results saved to: ${RESULTS_DIR}"
log "Log file: ${LOG_FILE}"
log ""

if [ "${QUICK_RESULT}" = "PASSED" ]; then
    log "✅ k6 is working correctly!"
    log ""
    log "Next steps:"
    log "1. Review results in ${RESULTS_DIR}"
    log "2. Run full load tests when services are ready"
    log "3. Use: ./run-load-tests.sh"
    exit 0
else
    log "❌ k6 validation failed"
    log "Check ${LOG_FILE} for details"
    exit 1
fi

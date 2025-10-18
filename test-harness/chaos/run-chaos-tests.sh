#!/bin/bash
# Created by: TEAM-107 | 2025-10-18
# Main chaos testing orchestration script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${RESULTS_DIR}/chaos_test_${TIMESTAMP}.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "🌪️  CHAOS TESTING SUITE"
log "======================================"
log "Created by: TEAM-107 | 2025-10-18"
log ""

# Start infrastructure
start_infrastructure() {
    log "🚀 Starting chaos testing infrastructure..."
    
    cd "${SCRIPT_DIR}"
    docker-compose -f docker-compose.chaos.yml up -d
    
    log "⏳ Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    local retries=10
    while [ $retries -gt 0 ]; do
        if curl -f -s http://localhost:8474/version > /dev/null 2>&1; then
            log "✅ Toxiproxy is ready"
            break
        fi
        retries=$((retries - 1))
        sleep 5
    done
    
    if [ $retries -eq 0 ]; then
        log "❌ Toxiproxy failed to start"
        return 1
    fi
}

# Run chaos scenarios
run_chaos_scenarios() {
    log ""
    log "🌪️  Running chaos scenarios..."
    
    # Install Python dependencies in chaos-controller
    docker-compose -f docker-compose.chaos.yml exec -T chaos-controller \
        pip install -q requests 2>&1 | tee -a "${LOG_FILE}"
    
    # Run chaos controller
    docker-compose -f docker-compose.chaos.yml exec -T chaos-controller \
        python3 /scripts/chaos_controller.py 2>&1 | tee -a "${LOG_FILE}"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "✅ All chaos scenarios completed successfully"
        return 0
    else
        log "⚠️  Some chaos scenarios failed"
        return 1
    fi
}

# Stop infrastructure
stop_infrastructure() {
    log ""
    log "🛑 Stopping chaos testing infrastructure..."
    
    cd "${SCRIPT_DIR}"
    docker-compose -f docker-compose.chaos.yml down -v
    
    log "✅ Infrastructure stopped"
}

# Copy results from container
copy_results() {
    log ""
    log "📊 Copying test results..."
    
    # Create results directory if it doesn't exist
    mkdir -p "${RESULTS_DIR}"
    
    # Copy results from chaos-controller container
    docker cp $(docker-compose -f docker-compose.chaos.yml ps -q chaos-controller):/results/. "${RESULTS_DIR}/" 2>/dev/null || true
    
    log "✅ Results copied to ${RESULTS_DIR}"
}

# Main execution
main() {
    local exit_code=0
    
    # Trap to ensure cleanup
    trap 'stop_infrastructure' EXIT
    
    # Start infrastructure
    if ! start_infrastructure; then
        log "❌ Failed to start infrastructure"
        exit 1
    fi
    
    # Run chaos scenarios
    if ! run_chaos_scenarios; then
        exit_code=1
    fi
    
    # Copy results
    copy_results
    
    # Summary
    log ""
    log "======================================"
    log "📊 CHAOS TESTING COMPLETE"
    log "======================================"
    log "Results saved to: ${RESULTS_DIR}"
    log "Log file: ${LOG_FILE}"
    
    if [ $exit_code -eq 0 ]; then
        log "✅ All chaos tests passed"
    else
        log "⚠️  Some chaos tests failed - check results"
    fi
    
    return $exit_code
}

main "$@"

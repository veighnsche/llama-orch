#!/bin/bash
# Created by: TEAM-107 | 2025-10-18
# Load testing orchestration script using k6

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${RESULTS_DIR}/load_test_${TIMESTAMP}.log"

# Configuration
QUEEN_URL="${QUEEN_URL:-http://localhost:8080}"
HIVE_URL="${HIVE_URL:-http://localhost:9200}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "📊 LOAD TESTING SUITE"
log "======================================"
log "Created by: TEAM-107 | 2025-10-18"
log "Target: ${QUEEN_URL}"
log ""

# Check if k6 is installed
check_k6() {
    if ! command -v k6 &> /dev/null; then
        log "❌ k6 not found. Installing..."
        
        # Install k6 using package manager or binary
        if command -v apt-get &> /dev/null; then
            sudo gpg -k
            sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
            echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
            sudo apt-get update
            sudo apt-get install k6
        else
            log "❌ Please install k6 manually: https://k6.io/docs/getting-started/installation/"
            return 1
        fi
    fi
    
    log "✅ k6 is available: $(k6 version)"
}

# Check service health
check_services() {
    log "🔍 Checking service health..."
    
    if ! curl -f -s "${QUEEN_URL}/health" > /dev/null; then
        log "❌ Queen-rbee is not healthy at ${QUEEN_URL}"
        return 1
    fi
    log "✅ Queen-rbee is healthy"
    
    if ! curl -f -s "${HIVE_URL}/health" > /dev/null; then
        log "❌ Rbee-hive is not healthy at ${HIVE_URL}"
        return 1
    fi
    log "✅ Rbee-hive is healthy"
    
    return 0
}

# Run inference load test
run_inference_load_test() {
    log ""
    log "🚀 Running inference load test (1000+ concurrent users)..."
    
    local output_file="${RESULTS_DIR}/inference_load_${TIMESTAMP}.json"
    
    k6 run \
        --out json="${output_file}" \
        --summary-export="${RESULTS_DIR}/inference_summary_${TIMESTAMP}.json" \
        -e QUEEN_URL="${QUEEN_URL}" \
        "${SCRIPT_DIR}/inference-load.js" 2>&1 | tee -a "${LOG_FILE}"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "✅ Inference load test passed"
        return 0
    else
        log "❌ Inference load test failed"
        return 1
    fi
}

# Run stress test
run_stress_test() {
    log ""
    log "🔥 Running stress test (finding breaking point)..."
    
    local output_file="${RESULTS_DIR}/stress_test_${TIMESTAMP}.json"
    
    k6 run \
        --out json="${output_file}" \
        --summary-export="${RESULTS_DIR}/stress_summary_${TIMESTAMP}.json" \
        -e QUEEN_URL="${QUEEN_URL}" \
        "${SCRIPT_DIR}/stress-test.js" 2>&1 | tee -a "${LOG_FILE}"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "✅ Stress test completed"
        return 0
    else
        log "⚠️  Stress test found breaking point (expected)"
        return 0
    fi
}

# Run spike test
run_spike_test() {
    log ""
    log "⚡ Running spike test (sudden traffic spikes)..."
    
    local output_file="${RESULTS_DIR}/spike_test_${TIMESTAMP}.json"
    
    k6 run \
        --out json="${output_file}" \
        --summary-export="${RESULTS_DIR}/spike_summary_${TIMESTAMP}.json" \
        -e QUEEN_URL="${QUEEN_URL}" \
        "${SCRIPT_DIR}/spike-test.js" 2>&1 | tee -a "${LOG_FILE}"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "✅ Spike test passed"
        return 0
    else
        log "❌ Spike test failed"
        return 1
    fi
}

# Generate summary report
generate_summary() {
    log ""
    log "📊 Generating summary report..."
    
    local summary_file="${RESULTS_DIR}/load_test_summary_${TIMESTAMP}.md"
    
    cat > "${summary_file}" << EOF
# Load Testing Summary

**Date:** $(date)
**Created by:** TEAM-107

## Test Configuration

- **Target:** ${QUEEN_URL}
- **Duration:** ~30 minutes total
- **Tests Run:** 3 (Inference Load, Stress, Spike)

## Results

### Inference Load Test
- **Max Concurrent Users:** 1000
- **Duration:** 16 minutes
- **Target p95 Latency:** < 500ms
- **Target Error Rate:** < 1%

See: \`inference_summary_${TIMESTAMP}.json\`

### Stress Test
- **Max Concurrent Users:** 5000
- **Duration:** 19 minutes
- **Goal:** Find breaking point

See: \`stress_summary_${TIMESTAMP}.json\`

### Spike Test
- **Max Spike:** 3000 users
- **Duration:** ~7 minutes
- **Goal:** Test recovery from sudden spikes

See: \`spike_summary_${TIMESTAMP}.json\`

## Files Generated

- \`load_test_${TIMESTAMP}.log\` - Full test log
- \`inference_load_${TIMESTAMP}.json\` - Raw k6 data
- \`inference_summary_${TIMESTAMP}.json\` - Summary metrics
- \`stress_test_${TIMESTAMP}.json\` - Raw stress test data
- \`stress_summary_${TIMESTAMP}.json\` - Stress test metrics
- \`spike_test_${TIMESTAMP}.json\` - Raw spike test data
- \`spike_summary_${TIMESTAMP}.json\` - Spike test metrics

## Next Steps

1. Review summary JSON files for detailed metrics
2. Check for threshold violations
3. Investigate any failed scenarios
4. Compare with baseline performance

EOF
    
    log "✅ Summary report generated: ${summary_file}"
}

# Main execution
main() {
    local failed=0
    
    # Check k6 installation
    if ! check_k6; then
        exit 1
    fi
    
    # Check services
    if ! check_services; then
        log "❌ Services are not ready"
        exit 1
    fi
    
    # Run tests
    run_inference_load_test || ((failed++))
    run_stress_test || ((failed++))
    run_spike_test || ((failed++))
    
    # Generate summary
    generate_summary
    
    # Final summary
    log ""
    log "======================================"
    log "📊 LOAD TESTING COMPLETE"
    log "======================================"
    log "Total tests: 3"
    log "Failed: ${failed}"
    log "Passed: $((3 - failed))"
    log ""
    log "Results saved to: ${RESULTS_DIR}"
    log "Log file: ${LOG_FILE}"
    
    if [ ${failed} -eq 0 ]; then
        log "✅ All load tests passed"
        return 0
    else
        log "❌ ${failed} load test(s) failed"
        return 1
    fi
}

main "$@"

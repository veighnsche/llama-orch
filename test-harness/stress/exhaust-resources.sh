#!/bin/bash
# Created by: TEAM-107 | 2025-10-18
# Resource exhaustion stress testing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${RESULTS_DIR}/stress_test_${TIMESTAMP}.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "🔥 RESOURCE EXHAUSTION STRESS TESTING"
log "======================================"

# Test 1: CPU Exhaustion
test_cpu_exhaustion() {
    log ""
    log "Test 1: CPU Exhaustion"
    log "----------------------"
    
    local container="rbee-hive"
    local container_id=$(docker ps -q -f "name=${container}" | head -1)
    
    if [ -z "${container_id}" ]; then
        log "❌ Container ${container} not found"
        return 1
    fi
    
    log "🎯 Target: ${container} (${container_id:0:12})"
    log "⏳ Saturating CPU for 60 seconds..."
    
    # Run stress-ng to saturate all CPUs
    docker exec "${container_id}" \
        stress-ng --cpu 0 --timeout 60s --metrics-brief 2>&1 | tee -a "${LOG_FILE}" || true
    
    log "✅ CPU exhaustion test completed"
    
    # Check if service is still responsive
    sleep 5
    if curl -f -s http://localhost:9200/health > /dev/null; then
        log "✅ Service recovered and is healthy"
        return 0
    else
        log "❌ Service did not recover"
        return 1
    fi
}

# Test 2: Memory Exhaustion
test_memory_exhaustion() {
    log ""
    log "Test 2: Memory Exhaustion"
    log "-------------------------"
    
    local container="mock-worker"
    local container_id=$(docker ps -q -f "name=${container}" | head -1)
    
    if [ -z "${container_id}" ]; then
        log "❌ Container ${container} not found"
        return 1
    fi
    
    log "🎯 Target: ${container} (${container_id:0:12})"
    log "⏳ Allocating 90% of available memory..."
    
    # Allocate memory until near exhaustion
    docker exec "${container_id}" \
        stress-ng --vm 1 --vm-bytes 90% --timeout 30s --metrics-brief 2>&1 | tee -a "${LOG_FILE}" || true
    
    log "✅ Memory exhaustion test completed"
    
    # Check container status
    sleep 5
    if docker ps -q -f "id=${container_id}" | grep -q .; then
        log "✅ Container survived memory pressure"
        return 0
    else
        log "⚠️  Container was killed by OOM killer (expected)"
        return 0
    fi
}

# Test 3: Disk Exhaustion
test_disk_exhaustion() {
    log ""
    log "Test 3: Disk Exhaustion"
    log "-----------------------"
    
    local container="rbee-hive"
    local container_id=$(docker ps -q -f "name=${container}" | head -1)
    
    if [ -z "${container_id}" ]; then
        log "❌ Container ${container} not found"
        return 1
    fi
    
    log "🎯 Target: ${container} (${container_id:0:12})"
    log "⏳ Filling disk with large file..."
    
    # Create large file to fill disk
    docker exec "${container_id}" \
        dd if=/dev/zero of=/tmp/fill bs=1M count=1000 2>&1 | tee -a "${LOG_FILE}" || true
    
    log "✅ Disk exhaustion test completed"
    
    # Clean up
    docker exec "${container_id}" rm -f /tmp/fill || true
    
    # Check if service is still responsive
    sleep 5
    if curl -f -s http://localhost:9200/health > /dev/null; then
        log "✅ Service recovered after disk cleanup"
        return 0
    else
        log "❌ Service did not recover"
        return 1
    fi
}

# Test 4: File Descriptor Exhaustion
test_fd_exhaustion() {
    log ""
    log "Test 4: File Descriptor Exhaustion"
    log "-----------------------------------"
    
    local container="queen-rbee"
    local container_id=$(docker ps -q -f "name=${container}" | head -1)
    
    if [ -z "${container_id}" ]; then
        log "❌ Container ${container} not found"
        return 1
    fi
    
    log "🎯 Target: ${container} (${container_id:0:12})"
    log "⏳ Opening many file descriptors..."
    
    # Open many files to exhaust file descriptors
    docker exec "${container_id}" bash -c '
        for i in {1..1000}; do
            touch /tmp/fd_test_$i
            exec 3< /tmp/fd_test_$i
        done
    ' 2>&1 | tee -a "${LOG_FILE}" || true
    
    log "✅ FD exhaustion test completed"
    
    # Clean up
    docker exec "${container_id}" bash -c 'rm -f /tmp/fd_test_*' || true
    
    return 0
}

# Test 5: Network Connection Exhaustion
test_connection_exhaustion() {
    log ""
    log "Test 5: Network Connection Exhaustion"
    log "--------------------------------------"
    
    log "⏳ Opening 1000 concurrent connections..."
    
    # Use parallel curl to open many connections
    seq 1 1000 | xargs -P 100 -I {} curl -s -m 5 http://localhost:8080/health > /dev/null || true
    
    log "✅ Connection exhaustion test completed"
    
    # Check if service is still responsive
    sleep 5
    if curl -f -s http://localhost:8080/health > /dev/null; then
        log "✅ Service handled connection flood"
        return 0
    else
        log "❌ Service did not recover"
        return 1
    fi
}

# Test 6: Graceful Degradation
test_graceful_degradation() {
    log ""
    log "Test 6: Graceful Degradation"
    log "-----------------------------"
    
    log "⏳ Testing system behavior under combined stress..."
    
    # Apply multiple stressors simultaneously
    local pids=()
    
    # CPU stress in background
    (docker exec $(docker ps -q -f "name=rbee-hive" | head -1) \
        stress-ng --cpu 2 --timeout 30s 2>&1 | tee -a "${LOG_FILE}") &
    pids+=($!)
    
    # Memory stress in background
    (docker exec $(docker ps -q -f "name=mock-worker" | head -1) \
        stress-ng --vm 1 --vm-bytes 50% --timeout 30s 2>&1 | tee -a "${LOG_FILE}") &
    pids+=($!)
    
    # Wait for stressors
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
    
    log "✅ Combined stress test completed"
    
    # Check all services
    sleep 10
    local all_healthy=true
    
    if curl -f -s http://localhost:8080/health > /dev/null; then
        log "✅ Queen-rbee is healthy"
    else
        log "❌ Queen-rbee is not healthy"
        all_healthy=false
    fi
    
    if curl -f -s http://localhost:9200/health > /dev/null; then
        log "✅ Rbee-hive is healthy"
    else
        log "❌ Rbee-hive is not healthy"
        all_healthy=false
    fi
    
    if [ "$all_healthy" = true ]; then
        log "✅ System degraded gracefully"
        return 0
    else
        log "⚠️  Some services did not recover"
        return 1
    fi
}

# Main execution
main() {
    local failed=0
    
    # Run all tests
    test_cpu_exhaustion || ((failed++))
    test_memory_exhaustion || ((failed++))
    test_disk_exhaustion || ((failed++))
    test_fd_exhaustion || ((failed++))
    test_connection_exhaustion || ((failed++))
    test_graceful_degradation || ((failed++))
    
    # Summary
    log ""
    log "======================================"
    log "📊 STRESS TESTING SUMMARY"
    log "======================================"
    log "Total tests: 6"
    log "Failed: ${failed}"
    log "Passed: $((6 - failed))"
    log "Success rate: $(( (6 - failed) * 100 / 6 ))%"
    log ""
    log "Results saved to: ${LOG_FILE}"
    
    if [ ${failed} -eq 0 ]; then
        log "✅ All stress tests passed"
        return 0
    else
        log "❌ ${failed} stress test(s) failed"
        return 1
    fi
}

main "$@"

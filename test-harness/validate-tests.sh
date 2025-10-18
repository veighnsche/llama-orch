#!/bin/bash
# Created by: TEAM-107 | 2025-10-18
# Validation script for chaos & load testing infrastructure
# Tests all components without requiring running services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SCRIPT_DIR}/validation_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "$*" | tee -a "${LOG_FILE}"
}

pass() {
    log "${GREEN}✅ $*${NC}"
}

fail() {
    log "${RED}❌ $*${NC}"
}

warn() {
    log "${YELLOW}⚠️  $*${NC}"
}

info() {
    log "ℹ️  $*"
}

# Track results
total_tests=0
passed_tests=0
failed_tests=0

run_test() {
    local test_name=$1
    shift
    ((total_tests++))
    
    info "Testing: ${test_name}"
    if "$@" >> "${LOG_FILE}" 2>&1; then
        pass "${test_name}"
        ((passed_tests++))
        return 0
    else
        fail "${test_name}"
        ((failed_tests++))
        return 1
    fi
}

log "🧪 TEAM-107 Test Infrastructure Validation"
log "=========================================="
log "Timestamp: ${TIMESTAMP}"
log ""

# Test 1: Check directory structure
log "📁 Checking Directory Structure"
log "--------------------------------"

run_test "chaos/ directory exists" test -d "${SCRIPT_DIR}/chaos"
run_test "load/ directory exists" test -d "${SCRIPT_DIR}/load"
run_test "stress/ directory exists" test -d "${SCRIPT_DIR}/stress"
run_test "bdd/ directory exists" test -d "${SCRIPT_DIR}/bdd"

log ""

# Test 2: Check executable scripts
log "🔧 Checking Executable Scripts"
log "-------------------------------"

run_test "run-all-chaos-load-tests.sh exists and executable" test -x "${SCRIPT_DIR}/run-all-chaos-load-tests.sh"
run_test "chaos/run-chaos-tests.sh exists and executable" test -x "${SCRIPT_DIR}/chaos/run-chaos-tests.sh"
run_test "load/run-load-tests.sh exists and executable" test -x "${SCRIPT_DIR}/load/run-load-tests.sh"
run_test "stress/exhaust-resources.sh exists and executable" test -x "${SCRIPT_DIR}/stress/exhaust-resources.sh"

log ""

# Test 3: Validate shell scripts syntax
log "📝 Validating Shell Script Syntax"
log "----------------------------------"

run_test "run-all-chaos-load-tests.sh syntax" bash -n "${SCRIPT_DIR}/run-all-chaos-load-tests.sh"
run_test "chaos/run-chaos-tests.sh syntax" bash -n "${SCRIPT_DIR}/chaos/run-chaos-tests.sh"
run_test "load/run-load-tests.sh syntax" bash -n "${SCRIPT_DIR}/load/run-load-tests.sh"
run_test "stress/exhaust-resources.sh syntax" bash -n "${SCRIPT_DIR}/stress/exhaust-resources.sh"

log ""

# Test 4: Validate Python scripts
log "🐍 Validating Python Scripts"
log "-----------------------------"

if command -v python3 &> /dev/null; then
    run_test "chaos_controller.py compiles" python3 -m py_compile "${SCRIPT_DIR}/chaos/scripts/chaos_controller.py"
else
    warn "Python3 not found, skipping Python validation"
fi

log ""

# Test 5: Validate JSON scenario files
log "📋 Validating JSON Scenario Files"
log "----------------------------------"

if command -v python3 &> /dev/null; then
    run_test "network-failures.json valid" python3 -c "import json; json.load(open('${SCRIPT_DIR}/chaos/scenarios/network-failures.json'))"
    run_test "worker-crashes.json valid" python3 -c "import json; json.load(open('${SCRIPT_DIR}/chaos/scenarios/worker-crashes.json'))"
    run_test "resource-exhaustion.json valid" python3 -c "import json; json.load(open('${SCRIPT_DIR}/chaos/scenarios/resource-exhaustion.json'))"
else
    warn "Python3 not found, skipping JSON validation"
fi

log ""

# Test 6: Validate k6 JavaScript files
log "📊 Validating k6 Load Test Scripts"
log "-----------------------------------"

if command -v node &> /dev/null; then
    run_test "inference-load.js syntax" node --check "${SCRIPT_DIR}/load/inference-load.js"
    run_test "stress-test.js syntax" node --check "${SCRIPT_DIR}/load/stress-test.js"
    run_test "spike-test.js syntax" node --check "${SCRIPT_DIR}/load/spike-test.js"
else
    warn "Node.js not found, skipping JavaScript validation"
fi

log ""

# Test 7: Check Docker Compose files exist
log "🐳 Checking Docker Compose Files"
log "---------------------------------"

run_test "docker-compose.chaos.yml exists" test -f "${SCRIPT_DIR}/chaos/docker-compose.chaos.yml"
run_test "docker-compose.integration.yml exists" test -f "${SCRIPT_DIR}/bdd/docker-compose.integration.yml"

log ""

# Test 8: Check documentation
log "📚 Checking Documentation"
log "-------------------------"

run_test "test-harness/README.md exists" test -f "${SCRIPT_DIR}/README.md"
run_test "chaos/README.md exists" test -f "${SCRIPT_DIR}/chaos/README.md"
run_test "load/README.md exists" test -f "${SCRIPT_DIR}/load/README.md"
run_test "stress/README.md exists" test -f "${SCRIPT_DIR}/stress/README.md"
run_test "CHAOS_LOAD_TESTING_SUMMARY.md exists" test -f "${SCRIPT_DIR}/CHAOS_LOAD_TESTING_SUMMARY.md"

log ""

# Test 9: Check scenario counts
log "🎯 Validating Scenario Counts"
log "------------------------------"

if command -v python3 &> /dev/null; then
    network_count=$(python3 -c "import json; print(len(json.load(open('${SCRIPT_DIR}/chaos/scenarios/network-failures.json'))['scenarios']))")
    worker_count=$(python3 -c "import json; print(len(json.load(open('${SCRIPT_DIR}/chaos/scenarios/worker-crashes.json'))['scenarios']))")
    resource_count=$(python3 -c "import json; print(len(json.load(open('${SCRIPT_DIR}/chaos/scenarios/resource-exhaustion.json'))['scenarios']))")
    
    info "Network failure scenarios: ${network_count}"
    info "Worker crash scenarios: ${worker_count}"
    info "Resource exhaustion scenarios: ${resource_count}"
    
    total_chaos=$((network_count + worker_count + resource_count))
    if [ ${total_chaos} -eq 15 ]; then
        pass "Total chaos scenarios: ${total_chaos} (expected 15)"
        ((passed_tests++))
    else
        fail "Total chaos scenarios: ${total_chaos} (expected 15)"
        ((failed_tests++))
    fi
    ((total_tests++))
fi

log ""

# Test 10: Check prerequisites
log "🔍 Checking Prerequisites"
log "-------------------------"

if command -v docker &> /dev/null; then
    docker_version=$(docker version --format '{{.Client.Version}}' 2>/dev/null || echo "unknown")
    pass "Docker installed: ${docker_version}"
else
    warn "Docker not found (required for chaos/stress tests)"
fi

if command -v k6 &> /dev/null; then
    k6_version=$(k6 version 2>/dev/null || echo "unknown")
    pass "k6 installed: ${k6_version}"
else
    warn "k6 not found (required for load tests)"
    info "Install: https://k6.io/docs/getting-started/installation/"
fi

if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    pass "Python3 installed: ${python_version}"
else
    warn "Python3 not found (required for chaos controller)"
fi

log ""

# Summary
log "=========================================="
log "📊 VALIDATION SUMMARY"
log "=========================================="
log "Total tests: ${total_tests}"
log "Passed: ${passed_tests}"
log "Failed: ${failed_tests}"
log "Success rate: $(( passed_tests * 100 / total_tests ))%"
log ""
log "Log file: ${LOG_FILE}"
log ""

if [ ${failed_tests} -eq 0 ]; then
    log "${GREEN}✅ ALL VALIDATION TESTS PASSED${NC}"
    log ""
    log "Next steps:"
    log "1. Install missing prerequisites (if any warnings above)"
    log "2. Run full test suite: ./run-all-chaos-load-tests.sh"
    log "3. Review results in test-results/ directory"
    exit 0
else
    log "${RED}❌ SOME VALIDATION TESTS FAILED${NC}"
    log ""
    log "Please fix the failed tests before running the full suite."
    log "Check ${LOG_FILE} for details."
    exit 1
fi

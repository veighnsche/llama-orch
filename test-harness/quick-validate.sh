#!/bin/bash
# Created by: TEAM-107 | 2025-10-18
# Quick validation of test infrastructure

# Don't exit on error - we want to collect all results
set +e

echo "🧪 TEAM-107 Quick Validation"
echo "============================="
echo ""

# Track results
PASS=0
FAIL=0

# Test function
test_item() {
    if eval "$2" > /dev/null 2>&1; then
        echo "✅ $1"
        ((PASS++))
    else
        echo "❌ $1"
        ((FAIL++))
    fi
}

# Directory structure
echo "📁 Directory Structure"
test_item "chaos/ exists" "[ -d chaos ]"
test_item "load/ exists" "[ -d load ]"
test_item "stress/ exists" "[ -d stress ]"
test_item "bdd/ exists" "[ -d bdd ]"
echo ""

# Executable scripts
echo "🔧 Executable Scripts"
test_item "run-all-chaos-load-tests.sh" "[ -x run-all-chaos-load-tests.sh ]"
test_item "chaos/run-chaos-tests.sh" "[ -x chaos/run-chaos-tests.sh ]"
test_item "load/run-load-tests.sh" "[ -x load/run-load-tests.sh ]"
test_item "stress/exhaust-resources.sh" "[ -x stress/exhaust-resources.sh ]"
echo ""

# Shell script syntax
echo "📝 Shell Script Syntax"
test_item "run-all-chaos-load-tests.sh syntax" "bash -n run-all-chaos-load-tests.sh"
test_item "chaos/run-chaos-tests.sh syntax" "bash -n chaos/run-chaos-tests.sh"
test_item "load/run-load-tests.sh syntax" "bash -n load/run-load-tests.sh"
test_item "stress/exhaust-resources.sh syntax" "bash -n stress/exhaust-resources.sh"
echo ""

# Python validation
echo "🐍 Python Scripts"
test_item "chaos_controller.py compiles" "python3 -m py_compile chaos/scripts/chaos_controller.py"
echo ""

# JSON validation
echo "📋 JSON Scenarios"
test_item "network-failures.json" "python3 -c 'import json; json.load(open(\"chaos/scenarios/network-failures.json\"))'"
test_item "worker-crashes.json" "python3 -c 'import json; json.load(open(\"chaos/scenarios/worker-crashes.json\"))'"
test_item "resource-exhaustion.json" "python3 -c 'import json; json.load(open(\"chaos/scenarios/resource-exhaustion.json\"))'"
echo ""

# JavaScript validation
echo "📊 k6 Scripts"
test_item "inference-load.js" "node --check load/inference-load.js"
test_item "stress-test.js" "node --check load/stress-test.js"
test_item "spike-test.js" "node --check load/spike-test.js"
echo ""

# Docker Compose files
echo "🐳 Docker Compose Files"
test_item "docker-compose.chaos.yml" "[ -f chaos/docker-compose.chaos.yml ]"
test_item "docker-compose.integration.yml" "[ -f bdd/docker-compose.integration.yml ]"
echo ""

# Documentation
echo "📚 Documentation"
test_item "test-harness/README.md" "[ -f README.md ]"
test_item "chaos/README.md" "[ -f chaos/README.md ]"
test_item "load/README.md" "[ -f load/README.md ]"
test_item "stress/README.md" "[ -f stress/README.md ]"
test_item "CHAOS_LOAD_TESTING_SUMMARY.md" "[ -f CHAOS_LOAD_TESTING_SUMMARY.md ]"
echo ""

# Summary
echo "============================="
echo "📊 Results"
echo "============================="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "Total: $((PASS + FAIL))"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
    exit 0
else
    echo "⚠️  Some tests failed"
    exit 1
fi

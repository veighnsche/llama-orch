#!/usr/bin/env bash
# BDD Test Runner with Progress Logging and Clean Output
# Created by: TEAM-102
# Usage: ./run-bdd-tests.sh [--tags @auth] [--feature lifecycle] [--verbose]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/.test-logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/bdd-test-$TIMESTAMP.log"
RESULTS_FILE="$LOG_DIR/bdd-results-$TIMESTAMP.txt"

# Default options
TAGS=""
FEATURE=""
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tags)
            TAGS="$2"
            shift 2
            ;;
        --feature)
            FEATURE="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tags TAG        Run tests with specific tag (e.g., @auth, @p0)"
            echo "  --feature NAME    Run specific feature file (e.g., lifecycle, authentication)"
            echo "  --verbose, -v     Show detailed output"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --tags @auth"
            echo "  $0 --feature lifecycle"
            echo "  $0 --tags @p0 --verbose"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Print header
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           BDD Test Runner - llama-orch Test Harness            ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📅 Timestamp:${NC} $TIMESTAMP"
echo -e "${BLUE}📂 Project Root:${NC} $PROJECT_ROOT"
echo -e "${BLUE}📝 Log File:${NC} $LOG_FILE"
echo ""

# Build test command
TEST_CMD="cargo test --test cucumber"

if [[ -n "$TAGS" ]]; then
    TEST_CMD="$TEST_CMD -- --tags $TAGS"
    echo -e "${BLUE}🏷️  Tags:${NC} $TAGS"
fi

if [[ -n "$FEATURE" ]]; then
    TEST_CMD="$TEST_CMD -- $FEATURE"
    echo -e "${BLUE}📋 Feature:${NC} $FEATURE"
fi

echo ""

# Step 1: Check compilation
echo -e "${YELLOW}[1/4]${NC} ${CYAN}Checking compilation...${NC}"
if $VERBOSE; then
    cargo check --lib 2>&1 | tee -a "$LOG_FILE"
    COMPILE_STATUS=${PIPESTATUS[0]}
else
    cargo check --lib > "$LOG_FILE" 2>&1
    COMPILE_STATUS=$?
fi

if [[ $COMPILE_STATUS -ne 0 ]]; then
    echo -e "${RED}❌ Compilation failed!${NC}"
    echo ""
    echo -e "${YELLOW}Compilation errors:${NC}"
    grep "^error" "$LOG_FILE" | head -20
    echo ""
    echo -e "${YELLOW}Full log:${NC} $LOG_FILE"
    exit 1
fi

echo -e "${GREEN}✅ Compilation successful${NC}"
echo ""

# Step 2: Count test scenarios
echo -e "${YELLOW}[2/4]${NC} ${CYAN}Discovering test scenarios...${NC}"

FEATURE_FILES=$(find tests/features -name "*.feature" 2>/dev/null || echo "")
TOTAL_SCENARIOS=0

if [[ -n "$FEATURE_FILES" ]]; then
    TOTAL_SCENARIOS=$(grep -h "^\s*Scenario:" $FEATURE_FILES 2>/dev/null | wc -l || echo "0")
fi

echo -e "${GREEN}📊 Found $TOTAL_SCENARIOS scenarios in feature files${NC}"
echo ""

# Step 3: Run tests
echo -e "${YELLOW}[3/4]${NC} ${CYAN}Running BDD tests...${NC}"
echo -e "${BLUE}Command:${NC} $TEST_CMD"
echo ""

# Create a temporary file for test output
TEMP_OUTPUT=$(mktemp)

# Run tests and capture output
echo -e "${CYAN}⏳ Test execution in progress...${NC}"
echo ""

if $VERBOSE; then
    # Verbose mode: show all output in real-time
    cd "$SCRIPT_DIR"
    $TEST_CMD 2>&1 | tee "$TEMP_OUTPUT" | tee -a "$LOG_FILE"
    TEST_STATUS=${PIPESTATUS[0]}
else
    # Normal mode: show progress indicators
    cd "$SCRIPT_DIR"
    $TEST_CMD > "$TEMP_OUTPUT" 2>&1 &
    TEST_PID=$!
    
    # Show progress while tests run
    SPIN='-\|/'
    i=0
    while kill -0 $TEST_PID 2>/dev/null; do
        i=$(( (i+1) %4 ))
        printf "\r${CYAN}⏳ Running tests... ${SPIN:$i:1}${NC}"
        sleep 0.2
    done
    
    # Get exit status
    wait $TEST_PID
    TEST_STATUS=$?
    
    printf "\r${CYAN}⏳ Running tests... Done!${NC}\n"
    
    # Append to log
    cat "$TEMP_OUTPUT" >> "$LOG_FILE"
fi

echo ""

# Step 4: Parse and display results
echo -e "${YELLOW}[4/4]${NC} ${CYAN}Parsing test results...${NC}"
echo ""

# Extract key information from output
PASSED=$(grep -o "[0-9]* passed" "$TEMP_OUTPUT" | grep -o "[0-9]*" || echo "0")
FAILED=$(grep -o "[0-9]* failed" "$TEMP_OUTPUT" | grep -o "[0-9]*" || echo "0")
SKIPPED=$(grep -o "[0-9]* skipped" "$TEMP_OUTPUT" | grep -o "[0-9]*" || echo "0")

# Extract scenario results
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                        TEST RESULTS                            ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Show summary
if [[ $TEST_STATUS -eq 0 ]]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED${NC}"
else
    echo -e "${RED}❌ TESTS FAILED${NC}"
fi

echo ""
echo -e "${BLUE}📊 Summary:${NC}"
echo -e "   ${GREEN}✅ Passed:${NC}  $PASSED"
echo -e "   ${RED}❌ Failed:${NC}  $FAILED"
echo -e "   ${YELLOW}⏭️  Skipped:${NC} $SKIPPED"
echo ""

# Show failed scenarios if any
if [[ $FAILED -gt 0 ]]; then
    echo -e "${RED}Failed Scenarios:${NC}"
    grep -A 5 "FAILED" "$TEMP_OUTPUT" | head -30 || echo "  (See log file for details)"
    echo ""
fi

# Show compilation warnings summary
WARNINGS=$(grep -c "^warning:" "$LOG_FILE" 2>/dev/null || echo "0")
if [[ $WARNINGS -gt 0 ]]; then
    echo -e "${YELLOW}⚠️  Compilation warnings: $WARNINGS${NC}"
    echo -e "   ${BLUE}(Run with --verbose to see all warnings)${NC}"
    echo ""
fi

# Save results summary
{
    echo "BDD Test Results - $TIMESTAMP"
    echo "================================"
    echo ""
    echo "Command: $TEST_CMD"
    echo "Status: $([ $TEST_STATUS -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
    echo ""
    echo "Summary:"
    echo "  Passed:  $PASSED"
    echo "  Failed:  $FAILED"
    echo "  Skipped: $SKIPPED"
    echo ""
    echo "Full log: $LOG_FILE"
} > "$RESULTS_FILE"

# Show file locations
echo -e "${BLUE}📁 Output Files:${NC}"
echo -e "   ${CYAN}Results:${NC} $RESULTS_FILE"
echo -e "   ${CYAN}Full Log:${NC} $LOG_FILE"
echo ""

# Show quick access commands
echo -e "${BLUE}💡 Quick Commands:${NC}"
echo -e "   ${CYAN}View results:${NC} cat $RESULTS_FILE"
echo -e "   ${CYAN}View full log:${NC} less $LOG_FILE"
echo -e "   ${CYAN}View errors:${NC} grep -E '^error' $LOG_FILE"
echo -e "   ${CYAN}View failures:${NC} grep -A 10 'FAILED' $LOG_FILE"
echo ""

# Cleanup temp file
rm -f "$TEMP_OUTPUT"

# Exit with test status
if [[ $TEST_STATUS -eq 0 ]]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    ✅ SUCCESS ✅                               ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    ❌ FAILED ❌                                ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi

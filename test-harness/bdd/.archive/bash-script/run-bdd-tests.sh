#!/usr/bin/env bash
# ============================================================================
# BDD Test Runner - World-Class Edition
# ============================================================================
# Created by: TEAM-102
# Refactored by: TEAM-111 - Complete architectural overhaul
# 
# Description:
#   Comprehensive BDD test runner with live output, failure-focused reporting,
#   and automatic rerun script generation. Follows engineering-rules.md with
#   no pipeline anti-patterns.
#
# Usage:
#   ./run-bdd-tests.sh [OPTIONS]
#
# Options:
#   --tags TAG        Run tests with specific tag (e.g., @auth, @p0)
#   --feature NAME    Run specific feature file (e.g., lifecycle)
#   --quiet, -q       Suppress live output (only show summary)
#   --help, -h        Show help message
#
# Examples:
#   ./run-bdd-tests.sh                    # Run all tests with live output
#   ./run-bdd-tests.sh --tags @auth       # Run @auth tests
#   ./run-bdd-tests.sh --quiet            # Quiet mode
#
# Exit Codes:
#   0 - All tests passed
#   1 - Tests failed (expected failure)
#   2 - Script error (validation, compilation, etc.)
#   >2 - Unexpected error
# ============================================================================

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Paths
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
readonly LOG_DIR="$SCRIPT_DIR/.test-logs"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Log files
readonly LOG_FILE="$LOG_DIR/bdd-test-$TIMESTAMP.log"
readonly COMPILE_LOG="$LOG_DIR/compile-$TIMESTAMP.log"
readonly TEST_OUTPUT="$LOG_DIR/test-output-$TIMESTAMP.log"
readonly RESULTS_FILE="$LOG_DIR/bdd-results-$TIMESTAMP.txt"
readonly FAILURES_FILE="$LOG_DIR/failures-$TIMESTAMP.txt"
readonly RERUN_SCRIPT="$LOG_DIR/rerun-failures.sh"
readonly RERUN_CMD_FILE="$LOG_DIR/rerun-failures-cmd.txt"

# Options (will be set by argument parser)
TAGS=""
FEATURE=""
QUIET=false

# Test results (will be set by parsers)
TEST_CMD=""
TEST_STATUS=0
PASSED=0
FAILED=0
SKIPPED=0
TOTAL_SCENARIOS=0
WARNINGS=0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Print colored header
print_header() {
    local message="$1"
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    printf "${CYAN}║%-64s║${NC}\n" "$(printf '%*s' $(((64+${#message})/2)) "$message" | sed 's/ *$//' | sed 's/^/                                /' | cut -c1-64)"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
}

# Print separator line
print_separator() {
    local color="${1:-$CYAN}"
    echo -e "${color}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Print step indicator
print_step() {
    local step="$1"
    local message="$2"
    echo -e "${YELLOW}[$step]${NC} ${CYAN}$message${NC}"
}

# Print info line
print_info() {
    local label="$1"
    local value="$2"
    echo -e "${BLUE}$label:${NC} $value"
}

# Print success message
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Print error message
print_error() {
    echo -e "${RED}❌ $1${NC}" >&2
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Cleanup temporary files
cleanup_temp_files() {
    rm -f "$LOG_DIR"/*.tmp 2>/dev/null || true
}

# Safe number extraction from file
safe_extract_number() {
    local file="$1"
    local default="${2:-0}"
    
    if [[ -f "$file" && -s "$file" ]]; then
        head -1 "$file" 2>/dev/null || echo "$default"
    else
        echo "$default"
    fi
}

# ============================================================================
# TRAP HANDLER
# ============================================================================

cleanup_on_exit() {
    local exit_code=$?
    
    # Clean up temp files
    cleanup_temp_files
    
    # Report unexpected errors
    if [[ $exit_code -ne 0 ]] && [[ $exit_code -ne 1 ]]; then
        echo ""
        print_separator "$RED"
        print_error "Script terminated unexpectedly with exit code: $exit_code"
        print_separator "$RED"
        if [[ -n "${LOG_FILE:-}" ]] && [[ -f "$LOG_FILE" ]]; then
            echo -e "${YELLOW}Check logs at: $LOG_FILE${NC}" >&2
        fi
    fi
}

trap cleanup_on_exit EXIT

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

validate_cargo_project() {
    if [[ ! -f "$SCRIPT_DIR/Cargo.toml" ]]; then
        print_error "Cannot find Cargo.toml in $SCRIPT_DIR"
        echo -e "${YELLOW}This script must be run from the test-harness/bdd directory${NC}" >&2
        exit 2
    fi
}

validate_features_directory() {
    if [[ ! -d "$SCRIPT_DIR/tests/features" ]]; then
        print_warning "No tests/features directory found"
        echo -e "${YELLOW}Expected at: $SCRIPT_DIR/tests/features${NC}" >&2
        echo ""
    fi
}

validate_environment() {
    validate_cargo_project
    validate_features_directory
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --tags TAG        Run tests with specific tag (e.g., @auth, @p0)
  --feature NAME    Run specific feature file (e.g., lifecycle, authentication)
  --quiet, -q       Suppress live output (only show summary)
  --help, -h        Show this help message

Examples:
  $0                      # Run all tests with live output
  $0 --tags @auth         # Run @auth tests with live output
  $0 --feature lifecycle  # Run lifecycle feature with live output
  $0 --tags @p0 --quiet   # Run @p0 tests quietly (summary only)

Note: By default, ALL stdout/stderr is shown in real-time!
EOF
    exit 0
}

parse_arguments() {
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
            --quiet|-q)
                QUIET=true
                shift
                ;;
            --help|-h)
                show_help
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

setup_logging() {
    mkdir -p "$LOG_DIR"
}

build_test_command() {
    TEST_CMD="cargo test --test cucumber"
    
    if [[ -n "$TAGS" ]]; then
        TEST_CMD="$TEST_CMD -- --tags $TAGS"
    fi
    
    if [[ -n "$FEATURE" ]]; then
        TEST_CMD="$TEST_CMD -- $FEATURE"
    fi
}

print_banner() {
    print_header "BDD Test Runner - llama-orch Test Harness"
    echo ""
    print_info "📅 Timestamp" "$TIMESTAMP"
    print_info "📂 Project Root" "$PROJECT_ROOT"
    print_info "📝 Log Directory" "$LOG_DIR"
    echo ""
    
    # Show output mode
    if $QUIET; then
        echo -e "${YELLOW}🔇 Output Mode:${NC} QUIET (summary only)"
    else
        echo -e "${GREEN}📺 Output Mode:${NC} LIVE (all stdout/stderr shown in real-time)"
    fi
    echo ""
    
    # Show filters if any
    [[ -n "$TAGS" ]] && print_info "🏷️  Tags" "$TAGS"
    [[ -n "$FEATURE" ]] && print_info "📋 Feature" "$FEATURE"
    [[ -n "$TAGS" || -n "$FEATURE" ]] && echo ""
}

# ============================================================================
# COMPILATION FUNCTIONS
# ============================================================================

run_compilation_check() {
    print_step "1/4" "Checking compilation..."
    echo ""
    
    local status=0
    
    if $QUIET; then
        cargo check --lib > "$COMPILE_LOG" 2>&1 || status=$?
    else
        cargo check --lib 2>&1 | tee "$COMPILE_LOG" || status=$?
    fi
    
    # Append to main log
    cat "$COMPILE_LOG" >> "$LOG_FILE"
    
    return $status
}

handle_compilation_failure() {
    echo ""
    print_error "Compilation failed!"
    echo ""
    
    if $QUIET; then
        echo -e "${YELLOW}Compilation errors:${NC}"
        grep "^error" "$COMPILE_LOG" > "$LOG_DIR/errors.tmp" 2>&1 || true
        head -20 "$LOG_DIR/errors.tmp" || echo "  (No error markers found, check log)"
        echo ""
    fi
    
    echo -e "${YELLOW}Full log:${NC} $COMPILE_LOG"
    exit 1
}

check_compilation() {
    if run_compilation_check; then
        echo ""
        print_success "Compilation successful"
        echo ""
    else
        handle_compilation_failure
    fi
}

# ============================================================================
# DISCOVERY FUNCTIONS
# ============================================================================

discover_test_scenarios() {
    print_step "2/4" "Discovering test scenarios..."
    
    # Find feature files
    local scenarios_tmp="$LOG_DIR/scenarios.tmp"
    find tests/features -name "*.feature" > "$scenarios_tmp" 2>/dev/null || echo "" > "$scenarios_tmp"
    
    TOTAL_SCENARIOS=0
    if [[ -s "$scenarios_tmp" ]]; then
        # Count scenarios
        grep -h "^\s*Scenario:" $(cat "$scenarios_tmp") > "$LOG_DIR/scenario-lines.tmp" 2>/dev/null || echo "" > "$LOG_DIR/scenario-lines.tmp"
        TOTAL_SCENARIOS=$(wc -l < "$LOG_DIR/scenario-lines.tmp" 2>/dev/null || echo "0")
    fi
    
    echo -e "${GREEN}📊 Found $TOTAL_SCENARIOS scenarios in feature files${NC}"
    echo ""
}

# ============================================================================
# TEST EXECUTION FUNCTIONS
# ============================================================================

run_tests_quiet() {
    $TEST_CMD > "$TEST_OUTPUT" 2>&1 &
    local test_pid=$!
    
    # Show spinner
    local spin='-\|/'
    local i=0
    while kill -0 $test_pid 2>/dev/null; do
        i=$(( (i+1) %4 ))
        printf "\r${CYAN}⏳ Running tests... ${spin:$i:1}${NC}"
        sleep 0.2
    done
    
    # Get exit status
    wait $test_pid
    TEST_STATUS=$?
    
    printf "\r${CYAN}⏳ Running tests... Done!${NC}\n"
}

run_tests_live() {
    echo -e "${GREEN}📺 LIVE OUTPUT MODE - You will see ALL test output below:${NC}"
    echo ""
    
    $TEST_CMD 2>&1 | tee "$TEST_OUTPUT"
    TEST_STATUS=${PIPESTATUS[0]}
}

run_tests() {
    print_step "3/4" "Running BDD tests..."
    print_info "Command" "$TEST_CMD"
    echo ""
    
    print_separator "$CYAN"
    echo -e "${CYAN}                    🧪 TEST EXECUTION START 🧪${NC}"
    print_separator "$CYAN"
    echo ""
    
    cd "$SCRIPT_DIR"
    
    if $QUIET; then
        run_tests_quiet
    else
        run_tests_live
    fi
    
    # Append to main log
    cat "$TEST_OUTPUT" >> "$LOG_FILE"
    
    echo ""
    print_separator "$CYAN"
    echo -e "${CYAN}                     🧪 TEST EXECUTION END 🧪${NC}"
    print_separator "$CYAN"
    echo ""
}

# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

parse_test_counts() {
    # Extract passed count
    grep -o "[0-9]* passed" "$TEST_OUTPUT" > "$LOG_DIR/passed.tmp" 2>/dev/null || echo "0 passed" > "$LOG_DIR/passed.tmp"
    grep -o "[0-9]*" "$LOG_DIR/passed.tmp" > "$LOG_DIR/passed-num.tmp" 2>/dev/null || echo "0" > "$LOG_DIR/passed-num.tmp"
    PASSED=$(safe_extract_number "$LOG_DIR/passed-num.tmp" "0")
    
    # Extract failed count
    grep -o "[0-9]* failed" "$TEST_OUTPUT" > "$LOG_DIR/failed.tmp" 2>/dev/null || echo "0 failed" > "$LOG_DIR/failed.tmp"
    grep -o "[0-9]*" "$LOG_DIR/failed.tmp" > "$LOG_DIR/failed-num.tmp" 2>/dev/null || echo "0" > "$LOG_DIR/failed-num.tmp"
    FAILED=$(safe_extract_number "$LOG_DIR/failed-num.tmp" "0")
    
    # Extract skipped count
    grep -o "[0-9]* skipped" "$TEST_OUTPUT" > "$LOG_DIR/skipped.tmp" 2>/dev/null || echo "0 skipped" > "$LOG_DIR/skipped.tmp"
    grep -o "[0-9]*" "$LOG_DIR/skipped.tmp" > "$LOG_DIR/skipped-num.tmp" 2>/dev/null || echo "0" > "$LOG_DIR/skipped-num.tmp"
    SKIPPED=$(safe_extract_number "$LOG_DIR/skipped-num.tmp" "0")
}

parse_warnings() {
    grep "^warning:" "$LOG_FILE" > "$LOG_DIR/warnings.tmp" 2>/dev/null || echo "" > "$LOG_DIR/warnings.tmp"
    WARNINGS=$(wc -l < "$LOG_DIR/warnings.tmp" 2>/dev/null || echo "0")
}

parse_test_results() {
    print_step "4/4" "Parsing test results..."
    echo ""
    
    parse_test_counts
    parse_warnings
}

# ============================================================================
# FAILURE EXTRACTION FUNCTIONS
# ============================================================================

extract_failure_patterns() {
    local output_file="$1"
    
    # Pattern 1: "FAILED" lines with context
    grep -B 2 -A 10 "FAILED" "$TEST_OUTPUT" > "$output_file" 2>/dev/null || true
    
    # Pattern 2: "Error:" lines with context
    grep -B 2 -A 5 "Error:" "$TEST_OUTPUT" >> "$output_file" 2>/dev/null || true
    
    # Pattern 3: "assertion" failures
    grep -B 2 -A 5 "assertion" "$TEST_OUTPUT" >> "$output_file" 2>/dev/null || true
    
    # Pattern 4: "panicked at" messages
    grep -B 2 -A 5 "panicked at" "$TEST_OUTPUT" >> "$output_file" 2>/dev/null || true
    
    # Pattern 5: Stack traces
    grep "^\s*at " "$TEST_OUTPUT" >> "$output_file" 2>/dev/null || true
}

extract_failed_test_names() {
    grep "test .* \.\.\. FAILED" "$TEST_OUTPUT" > "$LOG_DIR/failed-tests.tmp" 2>/dev/null || echo "" > "$LOG_DIR/failed-tests.tmp"
    
    if [[ -s "$LOG_DIR/failed-tests.tmp" ]]; then
        sed 's/test \(.*\) \.\.\. FAILED/\1/' "$LOG_DIR/failed-tests.tmp" > "$LOG_DIR/failed-test-names.tmp"
        return 0
    else
        return 1
    fi
}

# ============================================================================
# REPORTING FUNCTIONS
# ============================================================================

display_test_summary() {
    print_header "TEST RESULTS"
    echo ""
    
    if [[ $TEST_STATUS -eq 0 ]]; then
        print_success "ALL TESTS PASSED"
    else
        print_error "TESTS FAILED"
    fi
    
    echo ""
    echo -e "${BLUE}📊 Summary:${NC}"
    echo -e "   ${GREEN}✅ Passed:${NC}  $PASSED"
    echo -e "   ${RED}❌ Failed:${NC}  $FAILED"
    echo -e "   ${YELLOW}⏭️  Skipped:${NC} $SKIPPED"
    echo ""
}

display_failure_details() {
    print_separator "$RED"
    echo -e "${RED}                    ❌ FAILURE DETAILS ❌${NC}"
    print_separator "$RED"
    echo ""
    
    # Extract and display failures
    local failures_tmp="$LOG_DIR/failures-full.tmp"
    extract_failure_patterns "$failures_tmp"
    
    if [[ -s "$failures_tmp" ]]; then
        cat "$failures_tmp"
    else
        print_warning "No specific failure patterns found. Showing last 50 lines of output:"
        echo ""
        tail -50 "$TEST_OUTPUT"
    fi
    
    echo ""
    print_separator "$RED"
    echo ""
}

display_warnings() {
    if [[ $WARNINGS -gt 0 ]]; then
        print_warning "Compilation warnings: $WARNINGS"
        echo -e "   ${BLUE}(Check $COMPILE_LOG for details)${NC}"
        echo ""
    fi
}

# ============================================================================
# FILE GENERATION FUNCTIONS
# ============================================================================

generate_failures_file() {
    {
        echo "FAILURE DETAILS - $TIMESTAMP"
        echo "========================================"
        echo ""
        echo "Failed Tests: $FAILED"
        echo "Command: $TEST_CMD"
        echo ""
        echo "========================================"
        echo ""
        grep -B 2 -A 10 "FAILED" "$TEST_OUTPUT" 2>/dev/null || echo "No FAILED markers found"
        echo ""
        echo "========================================"
        echo "Errors:"
        echo "========================================"
        grep -B 2 -A 5 "Error:" "$TEST_OUTPUT" 2>/dev/null || echo "No Error: markers found"
        echo ""
        echo "========================================"
        echo "Panics:"
        echo "========================================"
        grep -B 2 -A 5 "panicked at" "$TEST_OUTPUT" 2>/dev/null || echo "No panics found"
    } > "$FAILURES_FILE"
    
    echo -e "${BLUE}💾 Detailed failures saved to:${NC} $FAILURES_FILE"
    echo ""
}

generate_rerun_script() {
    {
        echo "#!/usr/bin/env bash"
        echo "# Auto-generated script to re-run ONLY failed tests"
        echo "# Generated: $TIMESTAMP"
        echo "# Failed tests: $FAILED"
        echo ""
        echo "set -euo pipefail"
        echo ""
        echo "cd \"$SCRIPT_DIR\""
        echo ""
        echo "# Re-run only the failed tests:"
        
        while IFS= read -r test_name; do
            if [[ -n "$test_name" ]]; then
                echo "cargo test --test cucumber '$test_name' -- --nocapture"
            fi
        done < "$LOG_DIR/failed-test-names.tmp"
    } > "$RERUN_SCRIPT"
    
    chmod +x "$RERUN_SCRIPT"
}

generate_rerun_command() {
    local failed_tests_list=$(tr '\n' ' ' < "$LOG_DIR/failed-test-names.tmp" | sed 's/ $//')
    
    {
        echo "# Re-run failed tests from $TIMESTAMP"
        echo "# Copy and paste the command below:"
        echo ""
        echo "cd $SCRIPT_DIR"
        
        if [[ -n "$failed_tests_list" ]]; then
            echo "cargo test --test cucumber $failed_tests_list -- --nocapture"
        fi
    } > "$RERUN_CMD_FILE"
}

generate_rerun_files() {
    if extract_failed_test_names; then
        generate_rerun_script
        generate_rerun_command
        
        echo -e "${GREEN}🔄 Rerun script generated:${NC}"
        echo -e "   ${CYAN}Executable:${NC}  $RERUN_SCRIPT"
        echo -e "   ${CYAN}Command:${NC}     $RERUN_CMD_FILE"
        echo ""
        echo -e "${YELLOW}💡 To re-run ONLY the failed tests:${NC}"
        echo -e "   ${GREEN}$RERUN_SCRIPT${NC}"
        echo -e "   ${BLUE}or${NC}"
        echo -e "   ${GREEN}bash $RERUN_SCRIPT${NC}"
        echo ""
    else
        print_warning "Could not extract test names for rerun (check output format)"
        echo ""
    fi
}

handle_test_failures() {
    display_failure_details
    generate_failures_file
    generate_rerun_files
}

generate_results_summary() {
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
}

# ============================================================================
# OUTPUT DISPLAY FUNCTIONS
# ============================================================================

display_output_files() {
    echo -e "${BLUE}📁 Output Files:${NC}"
    echo -e "   ${CYAN}Summary:${NC}      $RESULTS_FILE"
    
    if [[ $FAILED -gt 0 ]]; then
        echo -e "   ${RED}Failures:${NC}     $FAILURES_FILE  ${YELLOW}⭐ START HERE${NC}"
        if [[ -f "$RERUN_SCRIPT" ]]; then
            echo -e "   ${GREEN}Rerun Script:${NC} $RERUN_SCRIPT  ${YELLOW}🔄 EXECUTABLE${NC}"
            echo -e "   ${GREEN}Rerun Cmd:${NC}    $RERUN_CMD_FILE  ${YELLOW}📋 COPY-PASTE${NC}"
        fi
    fi
    
    echo -e "   ${CYAN}Test Output:${NC}  $TEST_OUTPUT"
    echo -e "   ${CYAN}Compile Log:${NC}  $COMPILE_LOG"
    echo -e "   ${CYAN}Full Log:${NC}     $LOG_FILE"
    echo ""
}

display_quick_commands() {
    echo -e "${BLUE}💡 Quick Commands (respecting engineering-rules.md):${NC}"
    
    if [[ $FAILED -gt 0 ]]; then
        echo -e "   ${RED}View failures:${NC}   less $FAILURES_FILE  ${YELLOW}⭐ DEBUG${NC}"
        if [[ -f "$RERUN_SCRIPT" ]]; then
            echo -e "   ${GREEN}Rerun failed:${NC}    $RERUN_SCRIPT  ${YELLOW}🔄 FIX & RETRY${NC}"
        fi
    fi
    
    echo -e "   ${CYAN}View summary:${NC}    cat $RESULTS_FILE"
    echo -e "   ${CYAN}View test log:${NC}   less $TEST_OUTPUT"
    echo -e "   ${CYAN}View full log:${NC}   less $LOG_FILE"
    echo ""
}

display_final_banner() {
    if [[ $TEST_STATUS -eq 0 ]]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                    ✅ SUCCESS ✅                               ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    else
        echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║                    ❌ FAILED ❌                                ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    fi
}

# ============================================================================
# MAIN EXECUTION FLOW
# ============================================================================

main() {
    # Phase 1: Initialization
    parse_arguments "$@"
    setup_logging
    validate_environment
    build_test_command
    print_banner
    
    # Phase 2: Compilation
    check_compilation
    
    # Phase 3: Discovery
    discover_test_scenarios
    
    # Phase 4: Execution
    run_tests
    
    # Phase 5: Parsing
    parse_test_results
    
    # Phase 6: Reporting
    display_test_summary
    
    # Phase 7: Handle failures (if any)
    if [[ $FAILED -gt 0 ]]; then
        handle_test_failures
    fi
    
    # Phase 8: Display warnings
    display_warnings
    
    # Phase 9: Generate summary file
    generate_results_summary
    
    # Phase 10: Display output information
    display_output_files
    display_quick_commands
    
    # Phase 11: Final banner
    display_final_banner
    
    # Phase 12: Exit with appropriate code
    exit $TEST_STATUS
}

# ============================================================================
# ENTRY POINT
# ============================================================================

main "$@"

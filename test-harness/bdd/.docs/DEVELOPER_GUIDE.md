# Developer Guide - BDD Test Runner

**TEAM-111** - For developers who want to modify or extend the script  
**Date:** 2025-10-18

---

## 🎯 Quick Start for Developers

### Understanding the Structure

The script is organized into **clear sections** with **well-defined functions**. Each section handles a specific responsibility.

```
Header → Config → Utils → Validation → Init → Compile → Discover → Execute → Parse → Report → Generate → Display → Main
```

---

## 📚 Function Reference

### Utility Functions

#### `print_header(message)`
Print a colored header box.
```bash
print_header "My Header"
# Output:
# ╔════════════════════════════════════════════════════════════════╗
# ║                          My Header                             ║
# ╚════════════════════════════════════════════════════════════════╝
```

#### `print_separator([color])`
Print a separator line.
```bash
print_separator          # Cyan (default)
print_separator "$RED"   # Red
print_separator "$GREEN" # Green
```

#### `print_step(step, message)`
Print a step indicator.
```bash
print_step "1/4" "Checking compilation..."
# Output: [1/4] Checking compilation...
```

#### `print_info(label, value)`
Print an info line.
```bash
print_info "📅 Timestamp" "$TIMESTAMP"
# Output: 📅 Timestamp: 20251018_213000
```

#### `print_success(message)`
Print a success message.
```bash
print_success "Compilation successful"
# Output: ✅ Compilation successful
```

#### `print_error(message)`
Print an error message to stderr.
```bash
print_error "Compilation failed!"
# Output: ❌ Compilation failed!
```

#### `print_warning(message)`
Print a warning message.
```bash
print_warning "No features directory found"
# Output: ⚠️  No features directory found
```

#### `cleanup_temp_files()`
Remove all .tmp files from LOG_DIR.
```bash
cleanup_temp_files
```

#### `safe_extract_number(file, [default])`
Safely extract a number from a file.
```bash
PASSED=$(safe_extract_number "$LOG_DIR/passed-num.tmp" "0")
```

---

## 🔧 How to Add a New Feature

### Example: Add JSON Output

**Step 1:** Create the function in the appropriate section
```bash
# ============================================================================
# FILE GENERATION FUNCTIONS
# ============================================================================

# ... existing functions ...

generate_json_output() {
    local json_file="$LOG_DIR/results-$TIMESTAMP.json"
    
    {
        echo "{"
        echo "  \"timestamp\": \"$TIMESTAMP\","
        echo "  \"command\": \"$TEST_CMD\","
        echo "  \"status\": \"$([ $TEST_STATUS -eq 0 ] && echo 'passed' || echo 'failed')\","
        echo "  \"results\": {"
        echo "    \"passed\": $PASSED,"
        echo "    \"failed\": $FAILED,"
        echo "    \"skipped\": $SKIPPED"
        echo "  }"
        echo "}"
    } > "$json_file"
    
    echo -e "${BLUE}📄 JSON output saved to:${NC} $json_file"
}
```

**Step 2:** Call it from main()
```bash
main() {
    # ... existing code ...
    
    # Phase 9: Generate summary file
    generate_results_summary
    generate_json_output  # <-- Add here
    
    # ... rest of code ...
}
```

**Step 3:** Update display_output_files() if needed
```bash
display_output_files() {
    echo -e "${BLUE}📁 Output Files:${NC}"
    echo -e "   ${CYAN}Summary:${NC}      $RESULTS_FILE"
    echo -e "   ${CYAN}JSON:${NC}         $LOG_DIR/results-$TIMESTAMP.json"  # <-- Add
    # ... rest of function ...
}
```

**Done!** That's it. Clean and simple.

---

## 🎨 Adding a New Failure Pattern

Want to detect a new type of failure?

**Edit:** `extract_failure_patterns()`
```bash
extract_failure_patterns() {
    local output_file="$1"
    
    # Existing patterns...
    grep -B 2 -A 10 "FAILED" "$TEST_OUTPUT" > "$output_file" 2>/dev/null || true
    grep -B 2 -A 5 "Error:" "$TEST_OUTPUT" >> "$output_file" 2>/dev/null || true
    
    # Add your new pattern here
    grep -B 2 -A 5 "YOUR_PATTERN" "$TEST_OUTPUT" >> "$output_file" 2>/dev/null || true
}
```

**Also update:** `generate_failures_file()`
```bash
generate_failures_file() {
    {
        # ... existing sections ...
        
        echo "========================================"
        echo "Your New Section:"
        echo "========================================"
        grep -B 2 -A 5 "YOUR_PATTERN" "$TEST_OUTPUT" 2>/dev/null || echo "No YOUR_PATTERN found"
    } > "$FAILURES_FILE"
}
```

---

## 🚦 Adding a New Execution Mode

Want to add a "verbose" mode?

**Step 1:** Add the option to configuration
```bash
# Default options
TAGS=""
FEATURE=""
QUIET=false
VERBOSE=false  # <-- Add this
```

**Step 2:** Update argument parser
```bash
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            # ... existing cases ...
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            # ... rest of cases ...
        esac
    done
}
```

**Step 3:** Create execution function
```bash
run_tests_verbose() {
    echo -e "${GREEN}📺 VERBOSE MODE - Extra details enabled${NC}"
    echo ""
    
    # Add verbose flags to command
    local verbose_cmd="$TEST_CMD --verbose"
    
    $verbose_cmd 2>&1 | tee "$TEST_OUTPUT"
    TEST_STATUS=${PIPESTATUS[0]}
}
```

**Step 4:** Update run_tests()
```bash
run_tests() {
    # ... setup code ...
    
    if $QUIET; then
        run_tests_quiet
    elif $VERBOSE; then
        run_tests_verbose  # <-- Add this
    else
        run_tests_live
    fi
    
    # ... rest of function ...
}
```

---

## 📊 Adding Custom Metrics

Want to track test duration?

**Step 1:** Add variables
```bash
# Test results (will be set by parsers)
TEST_CMD=""
TEST_STATUS=0
PASSED=0
FAILED=0
SKIPPED=0
TOTAL_SCENARIOS=0
WARNINGS=0
TEST_DURATION=0  # <-- Add this
```

**Step 2:** Capture start/end times
```bash
run_tests() {
    # ... setup code ...
    
    local start_time=$(date +%s)  # <-- Add this
    
    if $QUIET; then
        run_tests_quiet
    else
        run_tests_live
    fi
    
    local end_time=$(date +%s)  # <-- Add this
    TEST_DURATION=$((end_time - start_time))  # <-- Add this
    
    # ... rest of function ...
}
```

**Step 3:** Display in summary
```bash
display_test_summary() {
    # ... existing code ...
    
    echo ""
    echo -e "${BLUE}📊 Summary:${NC}"
    echo -e "   ${GREEN}✅ Passed:${NC}  $PASSED"
    echo -e "   ${RED}❌ Failed:${NC}  $FAILED"
    echo -e "   ${YELLOW}⏭️  Skipped:${NC} $SKIPPED"
    echo -e "   ${CYAN}⏱️  Duration:${NC} ${TEST_DURATION}s"  # <-- Add this
    echo ""
}
```

---

## 🔍 Debugging Tips

### Enable Bash Debugging
```bash
# Add to top of script (after shebang)
set -x  # Print each command before executing
```

### Test Individual Functions
```bash
# Source the script
source run-bdd-tests.sh

# Call functions directly
print_header "Test Header"
print_separator "$RED"
```

### Check Variable Values
```bash
# Add debug output
echo "DEBUG: FAILED=$FAILED" >&2
echo "DEBUG: TEST_OUTPUT=$TEST_OUTPUT" >&2
```

### Validate File Operations
```bash
# Check if file exists and has content
if [[ -s "$TEST_OUTPUT" ]]; then
    echo "File exists and has content"
else
    echo "File missing or empty"
fi
```

---

## 🎯 Best Practices

### 1. **Keep Functions Small**
Each function should do ONE thing.
```bash
# Good
parse_test_counts() {
    parse_passed_count
    parse_failed_count
    parse_skipped_count
}

# Bad
parse_everything() {
    # 200 lines of mixed parsing, display, and file generation
}
```

### 2. **Use Descriptive Names**
Function names should explain what they do.
```bash
# Good
extract_failed_test_names()
generate_rerun_script()
display_test_summary()

# Bad
process_data()
do_stuff()
handle_things()
```

### 3. **Avoid Global State**
Pass parameters to functions.
```bash
# Good
extract_failure_patterns "$output_file"

# Bad (relies on global variable)
extract_failure_patterns()  # Uses $OUTPUT_FILE from global scope
```

### 4. **Return Status Codes**
Use return codes for success/failure.
```bash
extract_failed_test_names() {
    # ... extraction logic ...
    
    if [[ -s "$LOG_DIR/failed-tests.tmp" ]]; then
        return 0  # Success
    else
        return 1  # Failure
    fi
}

# Usage
if extract_failed_test_names; then
    generate_rerun_files
fi
```

### 5. **Clean Up After Yourself**
Always clean up temp files.
```bash
my_function() {
    local temp_file="$LOG_DIR/my-temp.tmp"
    
    # Do work
    process_data > "$temp_file"
    
    # Clean up
    rm -f "$temp_file"
}
```

### 6. **Use Readonly for Constants**
Prevent accidental modification.
```bash
readonly RED='\033[0;31m'
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
```

### 7. **Follow the Pattern**
Look at existing functions and follow the same style.

---

## 📝 Code Review Checklist

Before submitting changes:

- [ ] Function has a clear, descriptive name
- [ ] Function does ONE thing
- [ ] Function is in the appropriate section
- [ ] Local variables are declared with `local`
- [ ] Temp files are cleaned up
- [ ] Error cases are handled
- [ ] Return codes are used appropriately
- [ ] Code follows existing style
- [ ] No pipeline anti-patterns (grep | grep)
- [ ] Comments explain WHY, not WHAT
- [ ] Function is called from main() if needed
- [ ] Help text updated if new option added
- [ ] Documentation updated if behavior changed

---

## 🚀 Common Modifications

### Change Color Scheme
Edit the color constants:
```bash
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
# ... etc
```

### Add New Log File
```bash
# Add to configuration
readonly MY_LOG="$LOG_DIR/my-log-$TIMESTAMP.log"

# Use in functions
echo "data" > "$MY_LOG"

# Display in output
display_output_files() {
    # ... existing files ...
    echo -e "   ${CYAN}My Log:${NC}       $MY_LOG"
}
```

### Change Output Format
Edit the display functions:
```bash
display_test_summary() {
    # Modify this function to change summary format
}
```

### Add Validation
Add to `validate_environment()`:
```bash
validate_environment() {
    validate_cargo_project
    validate_features_directory
    validate_my_new_thing  # <-- Add your validation
}
```

---

## 🎓 Learning Resources

### Bash Best Practices
- Use `set -euo pipefail`
- Quote variables: `"$VAR"`
- Use `readonly` for constants
- Use `local` for function variables
- Check return codes: `if command; then`

### Function Design
- Single Responsibility Principle
- DRY (Don't Repeat Yourself)
- KISS (Keep It Simple, Stupid)
- Composition over complexity

### Testing
- Test each function independently
- Test edge cases
- Test error conditions
- Test with different inputs

---

## 💡 Pro Tips

### 1. **Use ShellCheck**
```bash
shellcheck run-bdd-tests.sh
```

### 2. **Test in Isolation**
```bash
# Create test script
#!/usr/bin/env bash
source run-bdd-tests.sh

# Test specific function
print_header "Test"
```

### 3. **Use Bash Debugger**
```bash
bash -x run-bdd-tests.sh --help
```

### 4. **Profile Performance**
```bash
time ./run-bdd-tests.sh
```

### 5. **Check for Leaks**
```bash
# Before
ls -la .test-logs/*.tmp

# Run script
./run-bdd-tests.sh

# After (should be clean)
ls -la .test-logs/*.tmp
```

---

## 🎉 Summary

The refactored script is:
- ✅ Easy to understand
- ✅ Easy to modify
- ✅ Easy to extend
- ✅ Well-organized
- ✅ Professional-grade

**Happy coding!** 🚀

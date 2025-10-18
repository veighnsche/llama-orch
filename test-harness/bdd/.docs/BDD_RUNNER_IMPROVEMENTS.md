# BDD Test Runner Improvements (TEAM-111)

**Date:** 2025-10-18  
**Modified by:** TEAM-111  
**Original by:** TEAM-102

## 🎯 Primary Goals Achieved

1. **LIVE OUTPUT BY DEFAULT** - User sees ALL stdout/stderr in real-time
2. **FAILURE-FOCUSED REPORTING** - Final view shows ONLY failing test details (default)
3. **SELECTIVE CAPTURE** - Programmatic data still captured to files
4. **PIPELINE ANTI-PATTERN COMPLIANCE** - Follows `engineering-rules.md` strictly
5. **ROBUSTNESS** - Better error handling, validation, and cleanup

---

## 🔄 Major Changes

### 1. Inverted Verbosity Logic

**Before:**
- Default: Quiet mode with spinner
- `--verbose`: Show output

**After:**
- Default: **LIVE mode** - shows ALL output in real-time
- `--quiet`: Suppress output (only summary)

**Rationale:** User explicitly requested "I WANT TO REALLY SEE THE STDOUT AND STDERR"

### 2. Pipeline Anti-Pattern Elimination

**Before:**
```bash
PASSED=$(grep -o "[0-9]* passed" "$TEMP_OUTPUT" | grep -o "[0-9]*" || echo "0")
WARNINGS=$(grep -c "^warning:" "$LOG_FILE" 2>/dev/null || echo "0")
```

**After:**
```bash
# Step 1: Extract to file
grep -o "[0-9]* passed" "$TEST_OUTPUT" > "$LOG_DIR/passed.tmp" 2>/dev/null

# Step 2: Process file
grep -o "[0-9]*" "$LOG_DIR/passed.tmp" > "$LOG_DIR/passed-num.tmp"

# Step 3: Read result
PASSED=$(head -1 "$LOG_DIR/passed-num.tmp" || echo "0")

# Step 4: Cleanup
rm -f "$LOG_DIR/passed.tmp" "$LOG_DIR/passed-num.tmp"
```

**Rationale:** Follows `engineering-rules.md` lines 68-93 - no piping into interactive tools

### 3. Live Output Implementation

**Compilation Check:**
```bash
if $QUIET; then
    cargo check --lib > "$COMPILE_LOG" 2>&1
else
    # Show ALL output AND capture to file
    cargo check --lib 2>&1 | tee "$COMPILE_LOG"
fi
```

**Test Execution:**
```bash
if $QUIET; then
    $TEST_CMD > "$TEST_OUTPUT" 2>&1 &
    # Show spinner...
else
    # LIVE MODE: Show everything in real-time
    $TEST_CMD 2>&1 | tee "$TEST_OUTPUT"
fi
```

**Key Point:** `tee` is safe because it writes to a file, not an interactive tool

### 4. Better File Organization

**Before:**
- `$LOG_FILE` - main log
- `$RESULTS_FILE` - summary
- `$TEMP_OUTPUT` - temporary (deleted)

**After:**
- `$LOG_FILE` - consolidated log (all steps)
- `$COMPILE_LOG` - compilation output only
- `$TEST_OUTPUT` - test execution output only
- `$RESULTS_FILE` - summary
- Temp files properly cleaned up

### 5. Enhanced Error Handling

**Added:**
- Trap handler for unexpected exits
- Directory validation (checks for `Cargo.toml`)
- Feature directory warning
- Better error messages with proper stderr redirection

**Example:**
```bash
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]] && [[ $exit_code -ne 1 ]]; then
        echo "⚠️  Script terminated unexpectedly" >&2
        echo "Check logs at: $LOG_FILE" >&2
    fi
}
trap cleanup EXIT
```

### 6. Comprehensive Failure Reporting (NEW!)

**The final view now shows ONLY failure details by default** - exactly what you need to debug!

**Failure patterns detected:**
1. `FAILED` markers with context (2 lines before, 10 after)
2. `Error:` messages with context
3. `assertion` failures
4. `panicked at` messages
5. Stack traces

**Output structure when tests fail:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ❌ FAILURE DETAILS ❌
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[All failure information extracted from test output]
- FAILED test names
- Error messages
- Assertion failures
- Panic messages
- Stack traces

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💾 Detailed failures saved to: .test-logs/failures-TIMESTAMP.txt
```

**Dedicated failures file includes:**
- Failed test count
- Command that was run
- All FAILED markers with context
- All Error messages
- All panic messages
- Organized into sections for easy navigation

### 8. Auto-Generated Rerun Scripts (NEW!)

**Automatically creates executable scripts to re-run ONLY failed tests!**

When tests fail, the script generates:

1. **`rerun-failures.sh`** - Executable script
   ```bash
   #!/usr/bin/env bash
   # Auto-generated script to re-run ONLY failed tests
   # Generated: 20251018_213500
   # Failed tests: 3
   
   set -euo pipefail
   
   cd "/home/vince/Projects/llama-orch/test-harness/bdd"
   
   # Re-run only the failed tests:
   cargo test --test cucumber 'lifecycle::worker_shutdown' -- --nocapture
   cargo test --test cucumber 'auth::token_validation' -- --nocapture
   cargo test --test cucumber 'scheduling::priority_ordering' -- --nocapture
   ```

2. **`rerun-failures-cmd.txt`** - Copy-paste command
   ```bash
   # Re-run failed tests from 20251018_213500
   # Copy and paste the command below:
   
   cd /home/vince/Projects/llama-orch/test-harness/bdd
   cargo test --test cucumber lifecycle::worker_shutdown auth::token_validation scheduling::priority_ordering -- --nocapture
   ```

**Usage:**
```bash
# Just run the script!
.test-logs/rerun-failures.sh

# Or copy-paste from the command file
cat .test-logs/rerun-failures-cmd.txt
```

**Benefits:**
- ✅ No need to manually type test names
- ✅ Instant iteration on failures
- ✅ Both script and copy-paste options
- ✅ Includes `--nocapture` for full output
- ✅ Automatically extracts test names from output

### 7. Visual Improvements

**Added:**
- Clear output mode indicator in header
- Visual separators for test execution
- Better structured output sections
- Helpful quick commands that respect engineering rules
- Highlighted failures file in output list

**Example:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    🧪 TEST EXECUTION START 🧪
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📺 LIVE OUTPUT MODE - You will see ALL test output below:

[... all test output streams here ...]
```

---

## 📋 Usage Examples

### Default (Live Output)
```bash
./run-bdd-tests.sh
# Shows ALL compilation + test output in real-time
```

### With Filters (Still Live)
```bash
./run-bdd-tests.sh --tags @auth
./run-bdd-tests.sh --feature lifecycle
./run-bdd-tests.sh --tags @p0 --feature authentication
```

### Quiet Mode (For CI/Automation)
```bash
./run-bdd-tests.sh --quiet
./run-bdd-tests.sh --tags @p0 --quiet
```

---

## 🔍 For AI Agents

When you (AI) need to analyze test results:

1. **Run in quiet mode** to avoid overwhelming your context:
   ```bash
   ./run-bdd-tests.sh --quiet
   ```

2. **Read selective files** (no pipelines!):
   ```bash
   # Read summary
   cat .test-logs/bdd-results-*.txt
   
   # Extract errors to file first
   grep "^error" .test-logs/compile-*.log > errors.out 2>&1
   cat errors.out
   
   # Extract failures to file first
   grep -A 10 "FAILED" .test-logs/test-output-*.log > failures.out 2>&1
   cat failures.out
   ```

3. **Never use pipelines** - always write to file first, then read

---

## 🎓 Engineering Rules Compliance

This script now fully complies with `.windsurf/rules/engineering-rules.md`:

✅ **No CLI piping into interactive tools**
- All `grep | grep` replaced with `grep > file; grep file`
- All `grep -c` replaced with `grep > file; wc -l < file`

✅ **Proper team signatures**
- Added `TEAM-111` signatures to all modifications

✅ **Foreground execution with logging**
- Uses `tee` for live output + capture
- All logs preserved in timestamped files

---

## 📊 Output Files

All files are timestamped and stored in `.test-logs/`:

| File | Purpose | When to Use |
|------|---------|-------------|
| `failures-TIMESTAMP.txt` | **⭐ Extracted failure details** | **Start here when tests fail!** |
| `rerun-failures.sh` | **🔄 Executable rerun script** | **Run to retry only failed tests** |
| `rerun-failures-cmd.txt` | **📋 Copy-paste command** | **Alternative to script** |
| `bdd-results-TIMESTAMP.txt` | Human-readable summary | Quick status check |
| `test-output-TIMESTAMP.log` | Raw test execution output | Full test output review |
| `compile-TIMESTAMP.log` | Compilation output | Debug compile errors |
| `bdd-test-TIMESTAMP.log` | Consolidated log (all steps) | Full execution trace |

**Note:** The `failures-*.txt` and `rerun-failures.*` files are only created when tests fail!

---

## 🚀 Benefits

### For Users (Interactive)
- ✅ See everything happening in real-time
- ✅ **Final view shows ONLY failure details** - no scrolling through passed tests
- ✅ Dedicated failures file for easy debugging
- ✅ **Auto-generated rerun script** - instant retry of failed tests
- ✅ No manual test name typing needed
- ✅ No surprises - full visibility
- ✅ Can Ctrl+C if something looks wrong
- ✅ Immediate feedback on progress

### For AI Agents (Programmatic)
- ✅ Selective data extraction from files
- ✅ No pipeline hangs
- ✅ Deterministic output parsing
- ✅ Clear separation of concerns

### For CI/CD
- ✅ `--quiet` mode for clean logs
- ✅ Proper exit codes (0=pass, 1=fail, 2=error)
- ✅ Timestamped artifacts
- ✅ Structured output files

---

## 🔧 Technical Details

### Exit Codes
- `0` - All tests passed
- `1` - Tests failed (expected failure)
- `2` - Script error (missing Cargo.toml, etc.)
- Other - Unexpected error (trapped and reported)

### File Cleanup
- Temp files (`*.tmp`) are cleaned up immediately after use
- Log files are preserved with timestamps
- No orphaned processes (proper signal handling)

### Performance
- Live mode: Minimal overhead (just `tee`)
- Quiet mode: Background execution with spinner
- No unnecessary file operations

---

## 📝 Maintenance Notes

### Adding New Parsing Logic

**DON'T:**
```bash
RESULT=$(cat file | grep pattern | awk '{print $1}')
```

**DO:**
```bash
# Step 1: Extract to file
grep pattern file > pattern.tmp 2>/dev/null || echo "default" > pattern.tmp

# Step 2: Process file
awk '{print $1}' pattern.tmp > result.tmp

# Step 3: Read result
RESULT=$(cat result.tmp)

# Step 4: Cleanup
rm -f pattern.tmp result.tmp
```

### Testing Changes

1. Test in live mode: `./run-bdd-tests.sh`
2. Test in quiet mode: `./run-bdd-tests.sh --quiet`
3. Test with failures: `./run-bdd-tests.sh --tags @nonexistent`
4. Test error handling: Remove `Cargo.toml` temporarily

---

## 🎉 Summary

This script is now:
- **User-friendly** - Shows everything by default
- **AI-friendly** - Selective data extraction without pipelines
- **Robust** - Proper error handling and validation
- **Compliant** - Follows all engineering rules
- **Maintainable** - Clear patterns and documentation

**Bottom line:** You (the user) get to see everything. We (AI agents) get clean data. Everyone wins! 🎊
